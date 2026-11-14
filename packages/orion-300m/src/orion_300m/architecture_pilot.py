"""Reproducible local pilot comparing decoder, prefix-LM, and seq2seq designs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.model.encoder_decoder import Orion300MEncoderDecoderPilot
from orion_core.hashing import sha256_file
from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything

ArchitectureName = Literal["decoder", "prefix_lm", "encoder_decoder"]


class ArchitecturePilotConfig(BaseModel):
    """Small-scale, same-data settings for the plan's E-300-1 comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion300MConfig = Field(
        default_factory=lambda: Orion300MConfig(
            num_layers=12,
            hidden_size=512,
            num_heads=8,
            num_kv_heads=4,
            intermediate_size=1365,
            pretraining_length=512,
            context_length=1024,
            max_generation_tokens=256,
        )
    )
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=500,
            checkpoint_every_steps=100,
            checkpoint_dir="experiments/tutor/e-300-1",
        )
    )
    candidates: tuple[ArchitectureName, ...] = ("decoder", "prefix_lm", "encoder_decoder")
    encoder_decoder_layers: tuple[int, int] | None = None
    sequence_length: int = Field(default=512, ge=3)
    batch_size: int = Field(default=4, ge=1, le=1024)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    scheduler: Literal["wsd", "trapezoidal"] = "wsd"

    @model_validator(mode="after")
    def validate_pilot(self) -> ArchitecturePilotConfig:
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must contain unique architecture names")
        if "encoder_decoder" in self.candidates and self.model.num_layers < 2:
            raise ValueError("encoder-decoder candidate needs at least two total pilot layers")
        if self.sequence_length > self.model.context_length:
            raise ValueError("sequence_length exceeds model context_length")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if self.encoder_decoder_layers is not None and (
            len(self.encoder_decoder_layers) != 2
            or any(layer_count <= 0 for layer_count in self.encoder_decoder_layers)
        ):
            raise ValueError("encoder_decoder_layers must contain two positive layer counts")
        if (
            self.encoder_decoder_layers is not None
            and sum(self.encoder_decoder_layers) != self.model.num_layers
        ):
            raise ValueError("encoder-decoder pilot depth must equal model.num_layers")
        return self

    @property
    def encoder_decoder_layer_split(self) -> tuple[int, int]:
        """Return an explicit or balanced split of the configured pilot depth."""
        if self.encoder_decoder_layers is not None:
            return self.encoder_decoder_layers
        encoder_layers = max(1, self.model.num_layers // 2)
        decoder_layers = max(1, self.model.num_layers - encoder_layers)
        return encoder_layers, decoder_layers


class PairedTextDataset(Dataset[dict[str, list[int]]]):
    """Indexed tokenized source-target pairs for the architecture comparison."""

    def __init__(self, path: str | Path, *, config: ArchitecturePilotConfig) -> None:
        self.path = Path(path)
        self.config = config
        self.sha256 = sha256_file(self.path)
        self.offsets: list[int] = []
        byte_offset = 0
        with self.path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    byte_offset += len(line)
                    continue
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError(f"architecture pilot row {line_number} exceeds 4 MiB")
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid architecture pilot JSONL row {line_number}") from exc
                self._validate(row, line_number)
                self.offsets.append(byte_offset)
                byte_offset += len(line)
        if not self.offsets:
            raise ValueError("architecture pilot JSONL contains no examples")

    def _validate(self, row: Any, line_number: int) -> None:
        if not isinstance(row, dict) or set(row) != {"source_ids", "target_ids"}:
            raise ValueError(f"architecture pilot row {line_number} has an invalid schema")
        for name in ("source_ids", "target_ids"):
            ids = row[name]
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    type(token) is not int
                    or not 0 <= token < self.config.model.vocab_size
                    or token == self.config.model.pad_token_id
                    for token in ids
                )
            ):
                raise ValueError(f"architecture pilot row {line_number} has invalid {name}")
        if row["target_ids"][-1] != self.config.model.eos_token_id:
            raise ValueError(f"architecture pilot row {line_number} target must end with EOS")
        if len(row["source_ids"]) + len(row["target_ids"]) > self.config.sequence_length:
            raise ValueError(f"architecture pilot row {line_number} exceeds sequence_length")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


def _collate_pairs(
    rows: list[dict[str, list[int]]], *, pad_token_id: int, bos_token_id: int
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    source_width = max(len(row["source_ids"]) for row in rows)
    target_width = max(len(row["target_ids"]) for row in rows)
    combined_width = max(len(row["source_ids"]) + len(row["target_ids"]) for row in rows)
    source_ids = torch.full((batch_size, source_width), pad_token_id, dtype=torch.long)
    source_mask = torch.zeros((batch_size, source_width), dtype=torch.bool)
    decoder_input_ids = torch.full((batch_size, target_width), pad_token_id, dtype=torch.long)
    target_mask = torch.zeros((batch_size, target_width), dtype=torch.bool)
    target_labels = torch.full((batch_size, target_width), -100, dtype=torch.long)
    combined_ids = torch.full((batch_size, combined_width), pad_token_id, dtype=torch.long)
    combined_mask = torch.zeros((batch_size, combined_width), dtype=torch.bool)
    combined_labels = torch.full((batch_size, combined_width), -100, dtype=torch.long)
    prefix_lengths = torch.empty((batch_size,), dtype=torch.long)
    for row_index, row in enumerate(rows):
        source = torch.tensor(row["source_ids"], dtype=torch.long)
        target = torch.tensor(row["target_ids"], dtype=torch.long)
        source_length = len(source)
        target_length = len(target)
        source_ids[row_index, :source_length] = source
        source_mask[row_index, :source_length] = True
        decoder_input = torch.cat(
            (torch.tensor([bos_token_id], dtype=torch.long), target[:-1]), dim=0
        )
        decoder_input_ids[row_index, :target_length] = decoder_input
        target_mask[row_index, :target_length] = True
        target_labels[row_index, :target_length] = target
        combined = torch.cat((source, target), dim=0)
        combined_ids[row_index, : len(combined)] = combined
        combined_mask[row_index, : len(combined)] = True
        combined_labels[row_index, source_length : len(combined)] = target
        prefix_lengths[row_index] = source_length
    return {
        "source_ids": source_ids,
        "source_attention_mask": source_mask,
        "decoder_input_ids": decoder_input_ids,
        "target_attention_mask": target_mask,
        "target_labels": target_labels,
        "input_ids": combined_ids,
        "attention_mask": combined_mask,
        "labels": combined_labels,
        "prefix_lengths": prefix_lengths,
    }


def _model_for_candidate(
    candidate: ArchitectureName, config: ArchitecturePilotConfig
) -> torch.nn.Module:
    if candidate in {"decoder", "prefix_lm"}:
        return Orion300MDecoder(config.model)
    encoder_layers, decoder_layers = config.encoder_decoder_layer_split
    return Orion300MEncoderDecoderPilot(
        config.model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )


def _loss_for_candidate(
    candidate: ArchitectureName,
    network: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if candidate == "encoder_decoder":
        if not isinstance(network, Orion300MEncoderDecoderPilot):
            raise TypeError("encoder-decoder candidate received the wrong model")
        output = network(
            batch["source_ids"],
            batch["decoder_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["target_labels"],
        )
    else:
        if not isinstance(network, Orion300MDecoder):
            raise TypeError("decoder candidate received the wrong model")
        output = network(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prefix_lengths=batch["prefix_lengths"] if candidate == "prefix_lm" else None,
        )
    if output.loss is None:
        raise RuntimeError("architecture pilot model did not compute a loss")
    return output.loss


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_architecture_pilot(
    training_jsonl: str | Path,
    validation_jsonl: str | Path,
    output_dir: str | Path,
    config: ArchitecturePilotConfig,
    *,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train comparable local pilot variants and emit measured validation losses.

    The caller must explicitly invoke this experiment after the plan's data,
    hardware, and architecture review gates. This function never downloads data.
    """
    output = Path(output_dir)
    report_path = output / "architecture-pilot-report.json"
    if report_path.exists() and not resume:
        raise FileExistsError("Architecture report already exists; pass resume=True to replace it")
    train_set = PairedTextDataset(training_jsonl, config=config)
    validation_set = PairedTextDataset(validation_jsonl, config=config)
    if train_set.sha256 == validation_set.sha256:
        raise ValueError("architecture pilot training and validation files must be different")
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for candidate in config.candidates:
        seed_everything(config.trainer.seed)
        candidate_dir = output / candidate
        manager = CheckpointManager(candidate_dir / "checkpoints")
        sampler = StatefulRandomSampler(train_set, seed=config.trainer.seed)

        def collate(rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
            return _collate_pairs(
                rows,
                pad_token_id=config.model.pad_token_id,
                bos_token_id=config.model.bos_token_id,
            )

        def loss_function(
            network: torch.nn.Module,
            batch: dict[str, torch.Tensor],
            selected_candidate: ArchitectureName = candidate,
        ) -> torch.Tensor:
            return _loss_for_candidate(selected_candidate, network, batch)

        def data_state_provider(
            current_sampler: StatefulRandomSampler = sampler,
        ) -> dict[str, Any]:
            return {"sampler": current_sampler.state_dict()}

        def data_state_restorer(
            state: dict[str, Any], current_sampler: StatefulRandomSampler = sampler
        ) -> None:
            current_sampler.load_state_dict(state["sampler"])

        training_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        model = _model_for_candidate(candidate, config)
        optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = build_lr_scheduler(
            optimizer,
            kind=config.scheduler,
            total_steps=config.trainer.max_steps,
            warmup_steps=config.warmup_steps,
        )
        identity = {
            "model": "orion-300m-architecture-pilot",
            "architecture": candidate,
            "training_config": config.model_dump(mode="json"),
            "training_data_sha256": train_set.sha256,
        }
        trainer = Trainer(
            model,
            optimizer,
            config=config.trainer,
            scheduler=scheduler,
            device=device,
            loss_function=loss_function,
            checkpoint_manager=manager,
            data_state_provider=data_state_provider,
            data_state_restorer=data_state_restorer,
            checkpoint_metadata=identity,
        )
        if resume:
            if manager.latest() is not None:
                trainer.restore_checkpoint()
        elif manager.latest() is not None:
            raise FileExistsError(f"Checkpoint already exists for candidate {candidate}")
        started = time.perf_counter()
        observed_losses = trainer.fit(training_loader)
        latest = manager.latest()
        latest_step = -1
        if latest is not None:
            state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
            latest_step = int(state.get("step", -1))
        if latest_step != trainer.state.step:
            trainer.save_checkpoint()
        validation_loss = trainer.evaluate(validation_loader)
        elapsed = time.perf_counter() - started
        results[candidate] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "steps": trainer.state.step,
            "last_training_loss": observed_losses[-1] if observed_losses else None,
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed,
            "checkpoint": str(manager.latest()),
        }
    report = {
        "experiment": "E-300-1",
        "status": "measured_pilot_complete",
        "training_data_sha256": train_set.sha256,
        "validation_data_sha256": validation_set.sha256,
        "config": config.model_dump(mode="json"),
        "results": results,
        "interpretation": (
            "Pilot metrics compare these configurations on the supplied splits; "
            "they are not production quality or hardware-scale results."
        ),
    }
    _write_json_atomic(report_path, report)
    return report


__all__ = [
    "ArchitecturePilotConfig",
    "PairedTextDataset",
    "run_architecture_pilot",
]
