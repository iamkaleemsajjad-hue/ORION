"""Local paired-text data and Matryoshka contrastive training for ORION-100M."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import Orion100MEncoder
from orion_100m.model.losses import matryoshka_info_nce
from orion_core.hashing import sha256_file
from orion_train.checkpoint import CheckpointManager, load_checkpoint_model_weights
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomBatchSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything


class ContrastiveTrainingConfig(BaseModel):
    """Validated sequence, dimension, and optimization settings for paired-text training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion100MConfig = Field(default_factory=Orion100MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=100,
            checkpoint_dir="checkpoints/orion-100m-contrastive",
        )
    )
    sequence_length: int = Field(default=512, ge=2)
    batch_size: int = Field(default=64, ge=2, le=1024)
    learning_rate: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    scheduler: Literal["wsd", "trapezoidal"] = "wsd"
    dimensions: tuple[int, ...] = (640, 256, 128)
    dimension_weights: tuple[float, ...] | None = None
    temperature: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ContrastiveTrainingConfig:
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length exceeds the encoder maximum")
        if not self.dimensions or any(
            dimension <= 0 or dimension > self.model.hidden_size for dimension in self.dimensions
        ):
            raise ValueError("contrastive dimensions must be positive prefixes of hidden_size")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("contrastive dimensions must be unique")
        if self.dimension_weights is not None and (
            len(self.dimension_weights) != len(self.dimensions)
            or any(weight < 0 for weight in self.dimension_weights)
            or not any(self.dimension_weights)
        ):
            raise ValueError("dimension_weights must be non-negative and match dimensions")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        return self


class ContrastivePairDataset(Dataset[dict[str, list[int]]]):
    """Validate and offset-index a local JSONL file of tokenized positive pairs."""

    def __init__(
        self,
        path: str | Path,
        *,
        config: ContrastiveTrainingConfig,
    ) -> None:
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
                    raise ValueError(f"contrastive JSONL row {line_number} exceeds 4 MiB")
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid contrastive JSONL row {line_number}") from exc
                self._validate(record, line_number)
                self.offsets.append(byte_offset)
                byte_offset += len(line)
        if len(self.offsets) < 2:
            raise ValueError("contrastive training needs at least two positive pairs")

    def _validate(self, record: Any, line_number: int) -> None:
        if not isinstance(record, dict) or set(record) != {"anchor_ids", "positive_ids"}:
            raise ValueError(f"contrastive JSONL row {line_number} has an invalid schema")
        for name in ("anchor_ids", "positive_ids"):
            values = record[name]
            if (
                not isinstance(values, list)
                or not values
                or len(values) > self.config.sequence_length
                or any(
                    type(value) is not int
                    or not 0 <= value < self.config.model.vocab_size
                    or value == self.config.model.pad_token_id
                    for value in values
                )
            ):
                raise ValueError(f"contrastive JSONL row {line_number} has invalid {name}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        self._validate(record, index + 1)
        return cast(dict[str, list[int]], record)


def _collate_pairs(
    records: list[dict[str, list[int]]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    if len(records) < 2:
        raise ValueError("contrastive batches need at least two positive pairs")
    width = max(max(len(record["anchor_ids"]), len(record["positive_ids"])) for record in records)
    batch: dict[str, torch.Tensor] = {}
    for name, mask_name in (("anchor_ids", "anchor_mask"), ("positive_ids", "positive_mask")):
        ids = torch.full((len(records), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(records), width), dtype=torch.bool)
        for row, record in enumerate(records):
            values = torch.tensor(record[name], dtype=torch.long)
            ids[row, : len(values)] = values
            mask[row, : len(values)] = True
        batch[name] = ids
        batch[mask_name] = mask
    return batch


def _make_trainer(
    data_jsonl: str | Path,
    output_dir: str | Path,
    config: ContrastiveTrainingConfig,
    *,
    initial_checkpoint: str | Path | None,
    device: str,
) -> tuple[
    Trainer[Orion100MEncoder, dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    ContrastivePairDataset,
]:
    dataset = ContrastivePairDataset(data_jsonl, config=config)
    batch_size = min(config.batch_size, len(dataset))
    sampler = StatefulRandomBatchSampler(dataset, batch_size=batch_size, seed=config.trainer.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: _collate_pairs(rows, pad_token_id=config.model.pad_token_id),
    )
    seed_everything(config.trainer.seed)
    model = Orion100MEncoder(config.model)
    initial_checkpoint_sha256 = None
    if initial_checkpoint is not None:
        state, initial_checkpoint_sha256 = load_checkpoint_model_weights(
            initial_checkpoint,
            expected_model="orion-100m",
            expected_model_config=config.model.model_dump(mode="json"),
        )
        encoder_state = {
            name.removeprefix("discriminator."): value
            for name, value in state.items()
            if name.startswith("discriminator.")
        }
        if not encoder_state:
            raise ValueError("initial checkpoint has no ORION-100M RTD discriminator encoder")
        model.load_state_dict(encoder_state, strict=True)
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )

    def loss_fn(network: Orion100MEncoder, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = network(batch["anchor_ids"], attention_mask=batch["anchor_mask"]).pooled_output
        positive = network(
            batch["positive_ids"], attention_mask=batch["positive_mask"]
        ).pooled_output
        return matryoshka_info_nce(
            anchor,
            positive,
            dimensions=config.dimensions,
            weights=config.dimension_weights,
            temperature=config.temperature,
        )

    identity = {
        "model": "orion-100m-contrastive",
        "training_config": config.model_dump(mode="json"),
        "data_sha256": dataset.sha256,
    }
    if initial_checkpoint_sha256 is not None:
        identity["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    manager = CheckpointManager(Path(output_dir) / "checkpoints")
    trainer = Trainer(
        model,
        optimizer,
        config=config.trainer,
        scheduler=scheduler,
        device=device,
        loss_function=loss_fn,
        checkpoint_manager=manager,
        data_state_provider=lambda: {"sampler": sampler.state_dict()},
        data_state_restorer=lambda state: sampler.load_state_dict(state["sampler"]),
        checkpoint_metadata=identity,
    )
    return trainer, cast(DataLoader[dict[str, torch.Tensor]], loader), dataset


def train_contrastive(
    data_jsonl: str | Path,
    output_dir: str | Path,
    config: ContrastiveTrainingConfig,
    *,
    initial_checkpoint: str | Path,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train nested encoder embeddings from a local paired-text JSONL shard."""
    trainer, loader, dataset = _make_trainer(
        data_jsonl,
        output_dir,
        config,
        initial_checkpoint=initial_checkpoint,
        device=device,
    )
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("checkpoint manager was not configured")
    if resume:
        trainer.restore_checkpoint()
    elif manager.latest() is not None:
        raise FileExistsError("A checkpoint already exists; pass resume=True to continue")
    losses = trainer.fit(loader)
    latest = manager.latest()
    latest_step = -1
    if latest is not None:
        state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        latest_step = int(state.get("step", -1))
    if latest_step != trainer.state.step:
        trainer.save_checkpoint()
    return {
        "status": "training_completed",
        "objective": "matryoshka_contrastive",
        "step": trainer.state.step,
        "epoch": trainer.state.epoch,
        "observed_losses": losses,
        "data_sha256": dataset.sha256,
    }


def evaluate_contrastive(
    data_jsonl: str | Path,
    checkpoint_dir: str | Path,
    config: ContrastiveTrainingConfig,
    *,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate local contrastive loss without restoring the training sampler state."""
    trainer, loader, dataset = _make_trainer(
        data_jsonl,
        checkpoint_dir,
        config,
        initial_checkpoint=None,
        device=device,
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value for key, value in trainer.checkpoint_metadata.items() if key != "data_sha256"
        }
    )
    loss = trainer.evaluate(loader, max_batches=max_batches)
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("evaluation checkpoint manager was not configured")
    return {
        "status": "evaluated",
        "objective": "matryoshka_contrastive",
        "loss": loss,
        "data_sha256": dataset.sha256,
        "checkpoint": str(manager.latest()),
    }


__all__ = [
    "ContrastivePairDataset",
    "ContrastiveTrainingConfig",
    "evaluate_contrastive",
    "train_contrastive",
]
