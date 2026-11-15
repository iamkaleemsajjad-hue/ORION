"""Local JSONL data and resumable supervised multi-task training for ORION-100M."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset

from orion_100m.config import Orion100MConfig
from orion_100m.model.document_model import Orion100MDocumentModel, TaskLabelMap
from orion_100m.model.tasks import multi_task_loss
from orion_core.hashing import sha256_file
from orion_train.checkpoint import CheckpointManager, load_checkpoint_model_weights
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything

_CLASSIFICATION_KEYS = ("subject", "document_type", "topic")
_TARGET_KEYS = frozenset((*_CLASSIFICATION_KEYS, "multilabel", "metadata", "confidence"))


class FineTuningConfig(BaseModel):
    """Validated optimizer and sequence settings for supervised document training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion100MConfig = Field(default_factory=Orion100MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=100,
            checkpoint_dir="checkpoints/orion-100m-finetune",
        )
    )
    sequence_length: int = Field(default=512, ge=2)
    batch_size: int = Field(default=8, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    scheduler: Literal["wsd", "trapezoidal"] = "wsd"
    task_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def dimensions_fit(self) -> FineTuningConfig:
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length exceeds the encoder maximum")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if set(self.task_weights) - _TARGET_KEYS:
            raise ValueError("task_weights contains an unsupported task")
        if any(weight < 0 for weight in self.task_weights.values()):
            raise ValueError("task weights must be non-negative")
        return self


class JsonlDocumentDataset(Dataset[dict[str, Any]]):
    """Index and validate a local, tokenized multi-task JSONL file without loading it all."""

    def __init__(
        self,
        path: str | Path,
        *,
        labels: TaskLabelMap,
        config: Orion100MConfig,
        sequence_length: int,
    ) -> None:
        self.path = Path(path)
        labels.validate()
        self.labels = labels
        self.config = config
        self.sequence_length = sequence_length
        self.sha256 = sha256_file(self.path)
        self.offsets: list[int] = []
        self.target_keys: frozenset[str] | None = None
        byte_offset = 0
        with self.path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    byte_offset += len(line)
                    continue
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError(f"JSONL row {line_number} exceeds 4 MiB")
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSONL row {line_number}") from exc
                keys = self._validate_record(record, line_number)
                if self.target_keys is None:
                    self.target_keys = keys
                elif keys != self.target_keys:
                    raise ValueError(
                        f"JSONL row {line_number} has a different set of target fields; "
                        "use one consistent task set per shard"
                    )
                self.offsets.append(byte_offset)
                byte_offset += len(line)
        if not self.offsets:
            raise ValueError("supervised JSONL contains no records")
        if self.target_keys is None:
            raise ValueError("supervised JSONL contains no task targets")

    def _validate_record(self, record: Any, line_number: int) -> frozenset[str]:
        if not isinstance(record, dict) or set(record) - (set(_TARGET_KEYS) | {"input_ids"}):
            raise ValueError(f"JSONL row {line_number} has an invalid schema")
        tokens = record.get("input_ids")
        if (
            not isinstance(tokens, list)
            or not tokens
            or len(tokens) > self.sequence_length
            or any(
                type(token) is not int or not 0 <= token < self.config.vocab_size
                for token in tokens
            )
        ):
            raise ValueError(f"JSONL row {line_number} has invalid input_ids")
        keys = frozenset(record) - {"input_ids"}
        if not keys:
            raise ValueError(f"JSONL row {line_number} has no supervised targets")
        ranges = {
            "subject": len(self.labels.subjects),
            "document_type": len(self.labels.document_types),
            "topic": len(self.labels.topics),
        }
        for key, count in ranges.items():
            if key in keys and (type(record[key]) is not int or not 0 <= record[key] < count):
                raise ValueError(f"JSONL row {line_number} has an invalid {key} label")
        if "multilabel" in keys:
            values = record["multilabel"]
            if (
                not isinstance(values, list)
                or len(values) != len(self.labels.tags)
                or any(value not in (0, 1, 0.0, 1.0) for value in values)
            ):
                raise ValueError(f"JSONL row {line_number} has invalid multilabel targets")
        if "metadata" in keys:
            values = record["metadata"]
            if (
                not isinstance(values, list)
                or len(values) != len(tokens)
                or any(
                    type(value) is not int or not 0 <= value < len(self.labels.bio_tags)
                    for value in values
                )
            ):
                raise ValueError(f"JSONL row {line_number} has invalid metadata tag IDs")
        if "confidence" in keys:
            value = record["confidence"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"JSONL row {line_number} has invalid confidence target")
        return keys

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        self._validate_record(record, index + 1)
        return cast(dict[str, Any], record)


def collate_document_records(
    records: Sequence[dict[str, Any]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Pad aligned JSONL records and build explicit masks for every model task."""
    if not records:
        raise ValueError("cannot collate an empty batch")
    keys = set(records[0]) - {"input_ids"}
    if any(set(record) - {"input_ids"} != keys for record in records):
        raise ValueError("all records in a batch must have the same target fields")
    width = max(len(record["input_ids"]) for record in records)
    input_ids = torch.full((len(records), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), width), dtype=torch.bool)
    for row, record in enumerate(records):
        values = torch.tensor(record["input_ids"], dtype=torch.long)
        input_ids[row, : len(values)] = values
        attention_mask[row, : len(values)] = True
    batch: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in keys:
        if key in _CLASSIFICATION_KEYS:
            batch[key] = torch.tensor([record[key] for record in records], dtype=torch.long)
        elif key == "multilabel":
            batch[key] = torch.tensor([record[key] for record in records], dtype=torch.float32)
        elif key == "metadata":
            batch[key] = torch.full((len(records), width), -100, dtype=torch.long)
            for row, record in enumerate(records):
                values = torch.tensor(record[key], dtype=torch.long)
                batch[key][row, : len(values)] = values
        elif key == "confidence":
            batch[key] = torch.tensor([record[key] for record in records], dtype=torch.float32)
    return batch


def _make_trainer(
    path: str | Path,
    output_dir: str | Path,
    labels: TaskLabelMap,
    config: FineTuningConfig,
    *,
    initial_checkpoint: str | Path | None,
    device: str,
) -> tuple[
    Trainer[Orion100MDocumentModel, dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    JsonlDocumentDataset,
]:
    dataset = JsonlDocumentDataset(
        path,
        labels=labels,
        config=config.model,
        sequence_length=config.sequence_length,
    )
    sampler = StatefulRandomSampler(dataset, seed=config.trainer.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: collate_document_records(
            rows, pad_token_id=config.model.pad_token_id
        ),
    )
    seed_everything(config.trainer.seed)
    model = Orion100MDocumentModel(labels, config.model)
    initial_checkpoint_sha256 = None
    if initial_checkpoint is not None:
        source_state, initial_checkpoint_sha256 = load_checkpoint_model_weights(
            initial_checkpoint,
            expected_model="orion-100m",
            expected_model_config=config.model.model_dump(mode="json"),
        )
        encoder_state = {
            name.removeprefix("discriminator."): tensor
            for name, tensor in source_state.items()
            if name.startswith("discriminator.")
        }
        if not encoder_state:
            raise ValueError("initial checkpoint has no ORION-100M RTD discriminator encoder")
        model.tasks.encoder.load_state_dict(encoder_state, strict=True)
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )

    def loss_fn(network: Orion100MDocumentModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = network.tasks(batch["input_ids"], attention_mask=batch["attention_mask"])
        target = {key: batch[key] for key in batch if key not in {"input_ids", "attention_mask"}}
        loss, _components = multi_task_loss(
            output,
            target,
            weights=config.task_weights,
            attention_mask=batch["attention_mask"],
        )
        return loss

    manager = CheckpointManager(Path(output_dir) / "checkpoints")
    run_identity = {
        "model": "orion-100m-multitask",
        "training_config": config.model_dump(mode="json"),
        "labels": {
            "subjects": list(labels.subjects),
            "document_types": list(labels.document_types),
            "topics": list(labels.topics),
            "tags": list(labels.tags),
            "bio_tags": list(labels.bio_tags),
        },
        "data_sha256": dataset.sha256,
    }
    if initial_checkpoint_sha256 is not None:
        run_identity["initial_checkpoint_sha256"] = initial_checkpoint_sha256
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
        checkpoint_metadata=run_identity,
    )
    return trainer, loader, dataset


def train_multitask(
    train_jsonl: str | Path,
    output_dir: str | Path,
    labels: TaskLabelMap,
    config: FineTuningConfig,
    *,
    initial_checkpoint: str | Path,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train configured document heads from an explicit local tokenized JSONL shard."""
    trainer, loader, dataset = _make_trainer(
        train_jsonl,
        output_dir,
        labels,
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
        latest_state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        latest_step = int(latest_state.get("step", -1))
    if latest_step != trainer.state.step:
        trainer.save_checkpoint()
    return {
        "status": "training_completed",
        "objective": "multitask_document_finetuning",
        "step": trainer.state.step,
        "epoch": trainer.state.epoch,
        "observed_losses": losses,
        "data_sha256": dataset.sha256,
    }


def evaluate_multitask(
    eval_jsonl: str | Path,
    checkpoint_dir: str | Path,
    labels: TaskLabelMap,
    config: FineTuningConfig,
    *,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate the configured multi-task loss on a local held-out JSONL shard."""
    trainer, training_loader, dataset = _make_trainer(
        eval_jsonl, checkpoint_dir, labels, config, initial_checkpoint=None, device=device
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value for key, value in trainer.checkpoint_metadata.items() if key != "data_sha256"
        }
    )
    loss = trainer.evaluate(training_loader, max_batches=max_batches)
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("evaluation checkpoint manager was not configured")
    return {
        "status": "evaluated",
        "objective": "multitask_document_finetuning",
        "loss": loss,
        "data_sha256": dataset.sha256,
        "checkpoint": str(manager.latest()),
    }


__all__ = [
    "FineTuningConfig",
    "JsonlDocumentDataset",
    "collate_document_records",
    "evaluate_multitask",
    "train_multitask",
]
