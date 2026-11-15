"""Local registry-prototype training with explicit open-set course examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from orion_100m.config import Orion100MConfig
from orion_100m.model.document_model import Orion100MDocumentModel, TaskLabelMap
from orion_100m.model.losses import course_open_set_loss
from orion_core.errors import ArtifactIntegrityError
from orion_core.hashing import sha256_file
from orion_train.checkpoint import CheckpointManager, load_checkpoint_model_weights
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomBatchSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything


class CourseMatchingTrainingConfig(BaseModel):
    """Validated settings for registry-prototype course matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion100MConfig = Field(default_factory=Orion100MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=100,
            checkpoint_dir="checkpoints/orion-100m-course-matching",
        )
    )
    sequence_length: int = Field(default=256, ge=2)
    batch_size: int = Field(default=32, ge=2, le=1024)
    negative_courses: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    scheduler: str = "wsd"
    unknown_margin: float = Field(default=0.3, ge=-1, le=1)
    unknown_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> CourseMatchingTrainingConfig:
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length exceeds the encoder maximum")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if self.scheduler not in {"wsd", "trapezoidal"}:
            raise ValueError("scheduler must be wsd or trapezoidal")
        return self


class CourseRegistry:
    """Validated, tokenized course profiles held locally for one training run."""

    def __init__(self, path: str | Path, *, config: CourseMatchingTrainingConfig) -> None:
        self.path = Path(path)
        if self.path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("course registry exceeds the 128 MiB safety limit")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("course registry must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict) or set(value) != {"courses"}:
            raise ValueError("course registry schema must contain only a courses array")
        courses = value["courses"]
        if not isinstance(courses, list) or not 2 <= len(courses) <= 10000:
            raise ValueError("course registry must contain between 2 and 10000 courses")
        course_ids: list[str] = []
        token_ids: list[list[int]] = []
        for row_number, item in enumerate(courses, start=1):
            if not isinstance(item, dict) or set(item) != {"course_id", "input_ids"}:
                raise ValueError(f"course registry row {row_number} has an invalid schema")
            course_id = item["course_id"]
            ids = item["input_ids"]
            if not isinstance(course_id, str) or not course_id.strip():
                raise ValueError(f"course registry row {row_number} has an invalid course_id")
            if (
                not isinstance(ids, list)
                or not ids
                or len(ids) > config.sequence_length
                or any(
                    type(token) is not int
                    or not 0 <= token < config.model.vocab_size
                    or token == config.model.pad_token_id
                    for token in ids
                )
            ):
                raise ValueError(f"course registry row {row_number} has invalid input_ids")
            course_ids.append(course_id)
            token_ids.append(ids)
        if len(set(course_ids)) != len(course_ids):
            raise ValueError("course registry course_id values must be unique")
        self.course_ids = tuple(course_ids)
        self.token_ids = tuple(token_ids)
        self.index_by_course_id = {course_id: index for index, course_id in enumerate(course_ids)}
        self.sha256 = sha256_file(self.path)


class CourseDocumentDataset(Dataset[dict[str, Any]]):
    """Indexed tokenized documents labeled with a registry course or explicit null."""

    def __init__(
        self,
        path: str | Path,
        *,
        registry: CourseRegistry,
        config: CourseMatchingTrainingConfig,
    ) -> None:
        self.path = Path(path)
        self.registry = registry
        self.config = config
        self.sha256 = sha256_file(self.path)
        self.offsets: list[int] = []
        self.labels: list[int] = []
        byte_offset = 0
        saw_known = False
        saw_unknown = False
        with self.path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    byte_offset += len(line)
                    continue
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError(f"course matching JSONL row {line_number} exceeds 4 MiB")
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid course matching JSONL row {line_number}") from exc
                label = self._validate_row(row, line_number)
                self.offsets.append(byte_offset)
                self.labels.append(label)
                saw_known |= label >= 0
                saw_unknown |= label == -1
                byte_offset += len(line)
        if len(self.offsets) < 2:
            raise ValueError("course matching training needs at least two document examples")
        if not saw_known:
            raise ValueError("course matching dataset needs at least one known-course example")
        if not saw_unknown:
            raise ValueError(
                "open-set training needs explicit unknown examples with course_id null"
            )

    def _validate_row(self, row: Any, line_number: int) -> int:
        if not isinstance(row, dict) or set(row) != {"document_ids", "course_id"}:
            raise ValueError(f"course matching JSONL row {line_number} has an invalid schema")
        ids = row["document_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > self.config.sequence_length
            or any(
                type(token) is not int
                or not 0 <= token < self.config.model.vocab_size
                or token == self.config.model.pad_token_id
                for token in ids
            )
        ):
            raise ValueError(f"course matching JSONL row {line_number} has invalid document_ids")
        course_id = row["course_id"]
        if course_id is None:
            return -1
        if not isinstance(course_id, str) or course_id not in self.registry.index_by_course_id:
            raise ValueError(
                f"course matching JSONL row {line_number} references an unknown course"
            )
        return self.registry.index_by_course_id[course_id]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return {"document_ids": row["document_ids"], "course_label": self.labels[index]}


def _collate_documents(rows: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    if len(rows) < 2:
        raise ValueError("course-matching batches need at least two examples")
    width = max(len(row["document_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        values = torch.tensor(row["document_ids"], dtype=torch.long)
        ids[index, : len(values)] = values
        mask[index, : len(values)] = True
    return {
        "document_ids": ids,
        "document_mask": mask,
        "course_labels": torch.tensor([row["course_label"] for row in rows], dtype=torch.long),
    }


def _candidate_indices(
    course_labels: torch.Tensor, *, registry_size: int, negative_courses: int
) -> tuple[list[int], dict[int, int]]:
    positive = sorted({int(label) for label in course_labels.tolist() if int(label) >= 0})
    positive_set = set(positive)
    negative_pool = [index for index in range(registry_size) if index not in positive_set]
    sample_count = min(negative_courses, len(negative_pool))
    if sample_count:
        selection = torch.randperm(len(negative_pool))[:sample_count].tolist()
        selected_negative = [negative_pool[index] for index in selection]
    else:
        selected_negative = []
    candidates = positive + selected_negative
    if len(candidates) < 2:
        remaining = next(index for index in range(registry_size) if index not in candidates)
        candidates.append(remaining)
    return candidates, {
        registry_index: local_index for local_index, registry_index in enumerate(candidates)
    }


def _make_training(
    registry_json: str | Path,
    data_jsonl: str | Path,
    output_dir: str | Path,
    labels: TaskLabelMap,
    config: CourseMatchingTrainingConfig,
    *,
    initial_checkpoint: str | Path | None,
    device: str,
) -> tuple[
    Trainer[Orion100MDocumentModel, dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    CourseRegistry,
    CourseDocumentDataset,
]:
    registry = CourseRegistry(registry_json, config=config)
    dataset = CourseDocumentDataset(data_jsonl, registry=registry, config=config)
    batch_size = min(config.batch_size, len(dataset))
    sampler = StatefulRandomBatchSampler(dataset, batch_size=batch_size, seed=config.trainer.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: _collate_documents(rows, pad_token_id=config.model.pad_token_id),
    )
    seed_everything(config.trainer.seed)
    labels.validate()
    model = Orion100MDocumentModel(labels, config.model)
    label_map = {
        "subjects": list(labels.subjects),
        "document_types": list(labels.document_types),
        "topics": list(labels.topics),
        "tags": list(labels.tags),
        "bio_tags": list(labels.bio_tags),
    }
    initial_checkpoint_sha256 = None
    if initial_checkpoint is not None:
        state, initial_checkpoint_sha256 = load_checkpoint_model_weights(
            initial_checkpoint,
            expected_model="orion-100m-multitask",
            expected_model_config=config.model.model_dump(mode="json"),
        )
        checkpoint_path = Path(initial_checkpoint)
        if checkpoint_path.is_file() and checkpoint_path.name == "model.safetensors":
            checkpoint_path = checkpoint_path.parent
        elif checkpoint_path.is_dir() and (checkpoint_path / "latest.json").is_file():
            latest_path = CheckpointManager(checkpoint_path).latest()
            if latest_path is None:
                raise ArtifactIntegrityError("Initial multi-task checkpoint is empty")
            checkpoint_path = latest_path
        try:
            initial_state = json.loads(
                (checkpoint_path / "training_state.json").read_text(encoding="utf-8")
            )
            initial_labels = initial_state["metadata"]["run_identity"]["labels"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactIntegrityError("Initial checkpoint has no multi-task label map") from exc
        if initial_labels != label_map:
            raise ArtifactIntegrityError("Initial multi-task checkpoint label map does not match")
        model.load_state_dict(state, strict=True)
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )

    def loss_fn(network: Orion100MDocumentModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidates, remap = _candidate_indices(
            batch["course_labels"],
            registry_size=len(registry.course_ids),
            negative_courses=config.negative_courses,
        )
        target_device = next(network.parameters()).device
        width = max(len(registry.token_ids[index]) for index in candidates)
        course_ids = torch.full(
            (len(candidates), width), config.model.pad_token_id, dtype=torch.long
        )
        course_mask = torch.zeros((len(candidates), width), dtype=torch.bool)
        for row_index, registry_index in enumerate(candidates):
            values = torch.tensor(registry.token_ids[registry_index], dtype=torch.long)
            course_ids[row_index, : len(values)] = values
            course_mask[row_index, : len(values)] = True
        course_ids = course_ids.to(target_device)
        course_mask = course_mask.to(target_device)
        doc_embeddings = network.tasks.encoder(
            batch["document_ids"], attention_mask=batch["document_mask"]
        ).pooled_output
        course_embeddings = network.tasks.encoder(
            course_ids, attention_mask=course_mask
        ).pooled_output
        logits = network.prototype_matcher(doc_embeddings, course_embeddings)
        cosine_scores = (
            F.normalize(doc_embeddings.float(), dim=-1)
            @ F.normalize(course_embeddings.float(), dim=-1).T
        )
        labels = batch["course_labels"].to(target_device)
        remapped = torch.tensor(
            [remap[int(label)] if int(label) >= 0 else -1 for label in labels.tolist()],
            dtype=torch.long,
            device=target_device,
        )
        loss, _ = course_open_set_loss(
            logits,
            remapped,
            cosine_scores=cosine_scores,
            unknown_margin=config.unknown_margin,
            unknown_weight=config.unknown_weight,
        )
        return loss

    identity: dict[str, Any] = {
        "model": "orion-100m-course-matching",
        "training_config": config.model_dump(mode="json"),
        "labels": label_map,
        "registry_sha256": registry.sha256,
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
    return trainer, loader, registry, dataset


def train_course_matching(
    registry_json: str | Path,
    data_jsonl: str | Path,
    output_dir: str | Path,
    labels: TaskLabelMap,
    config: CourseMatchingTrainingConfig,
    *,
    initial_checkpoint: str | Path,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train local course prototypes and unknown rejection from tokenized examples."""
    trainer, loader, registry, dataset = _make_training(
        registry_json,
        data_jsonl,
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
        state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        latest_step = int(state.get("step", -1))
    if latest_step != trainer.state.step:
        trainer.save_checkpoint()
    return {
        "status": "training_completed",
        "objective": "registry_prototype_open_set",
        "step": trainer.state.step,
        "epoch": trainer.state.epoch,
        "observed_losses": losses,
        "registry_sha256": registry.sha256,
        "data_sha256": dataset.sha256,
    }


def evaluate_course_matching(
    registry_json: str | Path,
    data_jsonl: str | Path,
    checkpoint_dir: str | Path,
    labels: TaskLabelMap,
    config: CourseMatchingTrainingConfig,
    *,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate known-course and unknown-margin losses against local registry data."""
    trainer, loader, registry, dataset = _make_training(
        registry_json,
        data_jsonl,
        checkpoint_dir,
        labels,
        config,
        initial_checkpoint=None,
        device=device,
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value
            for key, value in trainer.checkpoint_metadata.items()
            if key not in {"data_sha256", "initial_checkpoint_sha256"}
        }
    )
    loss = trainer.evaluate(loader, max_batches=max_batches)
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("evaluation checkpoint manager was not configured")
    return {
        "status": "evaluated",
        "objective": "registry_prototype_open_set",
        "loss": loss,
        "registry_sha256": registry.sha256,
        "data_sha256": dataset.sha256,
        "checkpoint": str(manager.latest()),
    }


__all__ = [
    "CourseMatchingTrainingConfig",
    "CourseRegistry",
    "CourseDocumentDataset",
    "evaluate_course_matching",
    "train_course_matching",
]
