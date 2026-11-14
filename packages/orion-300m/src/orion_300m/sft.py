"""Chat rendering and resumable assistant-only supervised fine-tuning for ORION-300M."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.model.losses import assistant_only_sft_loss
from orion_core.hashing import sha256_file
from orion_tokenizer.spec import SPECIAL_TOKEN_IDS
from orion_train.checkpoint import CheckpointManager, load_checkpoint_model_weights
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything

_ROLE_IDS = {
    "system": SPECIAL_TOKEN_IDS["<|system|>"],
    "user": SPECIAL_TOKEN_IDS["<|user|>"],
    "assistant": SPECIAL_TOKEN_IDS["<|assistant|>"],
}


def render_chat_messages(
    messages: Sequence[dict[str, str]], tokenizer: Any, *, max_length: int
) -> dict[str, list[int]]:
    """Render validated chat turns and mark assistant content plus end tokens as targets."""
    if max_length < 4 or not messages:
        raise ValueError("messages must be non-empty and max_length must be at least four")
    input_ids = [SPECIAL_TOKEN_IDS["<|bos|>"]]
    assistant_mask = [False]
    assistant_target_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain exactly role and content")
        role, content = message["role"], message["content"]
        if role not in _ROLE_IDS:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be non-empty text")
        is_assistant = role == "assistant"
        input_ids.append(_ROLE_IDS[role])
        assistant_mask.append(False)
        content_ids = list(tokenizer.encode(content).input_ids)
        input_ids.extend(content_ids)
        assistant_mask.extend([is_assistant] * len(content_ids))
        input_ids.append(SPECIAL_TOKEN_IDS["<|end|>"])
        assistant_mask.append(is_assistant)
        if is_assistant:
            assistant_target_count += len(content_ids) + 1
        if len(input_ids) > max_length:
            raise ValueError(f"rendered conversation exceeds max_length={max_length}")
    if assistant_target_count == 0:
        raise ValueError("conversation must contain at least one assistant response")
    return {"input_ids": input_ids, "assistant_mask": assistant_mask}


def write_sft_jsonl(
    conversations: Sequence[Sequence[dict[str, str]]],
    tokenizer: Any,
    path: str | Path,
    *,
    max_length: int,
) -> Path:
    """Render local conversations into a new tokenized JSONL shard without overwriting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for messages in conversations:
            stream.write(
                json.dumps(
                    render_chat_messages(messages, tokenizer, max_length=max_length),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return destination


class SFTTrainingConfig(BaseModel):
    """Validated settings for local assistant-only SFT."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion300MConfig = Field(default_factory=Orion300MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=100,
            checkpoint_dir="checkpoints/orion-300m-sft",
        )
    )
    sequence_length: int = Field(default=2048, ge=2)
    batch_size: int = Field(default=2, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    scheduler: str = "wsd"

    @model_validator(mode="after")
    def dimensions_fit(self) -> SFTTrainingConfig:
        if self.sequence_length > self.model.context_length:
            raise ValueError("sequence_length exceeds decoder context_length")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if self.scheduler not in {"wsd", "trapezoidal"}:
            raise ValueError("scheduler must be wsd or trapezoidal")
        return self


class SFTJsonlDataset(Dataset[dict[str, Any]]):
    """Validated offset-indexed dataset of input IDs and assistant target masks."""

    def __init__(self, path: str | Path, *, config: SFTTrainingConfig) -> None:
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
                    raise ValueError(f"SFT JSONL row {line_number} exceeds 4 MiB")
                try:
                    item = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid SFT JSONL row {line_number}") from exc
                self._validate(item, line_number)
                self.offsets.append(byte_offset)
                byte_offset += len(line)
        if not self.offsets:
            raise ValueError("SFT JSONL contains no conversations")

    def _validate(self, item: Any, line_number: int) -> None:
        if not isinstance(item, dict) or set(item) != {"input_ids", "assistant_mask"}:
            raise ValueError(
                f"SFT JSONL row {line_number} must contain input_ids and assistant_mask"
            )
        ids, mask = item["input_ids"], item["assistant_mask"]
        if (
            not isinstance(ids, list)
            or not isinstance(mask, list)
            or len(ids) != len(mask)
            or len(ids) < 2
            or len(ids) > self.config.sequence_length
            or any(
                type(token) is not int or not 0 <= token < self.config.model.vocab_size
                for token in ids
            )
            or any(type(value) is not bool for value in mask)
            or not any(mask[1:])
        ):
            raise ValueError(f"SFT JSONL row {line_number} has invalid tokens or target mask")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


def collate_sft_records(
    records: Sequence[dict[str, Any]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad SFT records and keep padding excluded from both attention and loss."""
    if not records:
        raise ValueError("cannot collate an empty batch")
    width = max(len(record["input_ids"]) for record in records)
    ids = torch.full((len(records), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(records), width), dtype=torch.bool)
    assistant = torch.zeros((len(records), width), dtype=torch.bool)
    for row, record in enumerate(records):
        length = len(record["input_ids"])
        ids[row, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
        attention[row, :length] = True
        assistant[row, :length] = torch.tensor(record["assistant_mask"], dtype=torch.bool)
    if not assistant[:, 1:].any():
        raise ValueError("batch has no assistant next-token targets")
    return {"input_ids": ids, "attention_mask": attention, "assistant_mask": assistant}


def _make_trainer(
    path: str | Path,
    output_dir: str | Path,
    config: SFTTrainingConfig,
    *,
    initial_checkpoint: str | Path | None,
    device: str,
) -> tuple[Trainer, DataLoader, SFTJsonlDataset]:
    dataset = SFTJsonlDataset(path, config=config)
    sampler = StatefulRandomSampler(dataset, seed=config.trainer.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: collate_sft_records(rows, pad_token_id=config.model.pad_token_id),
    )
    seed_everything(config.trainer.seed)
    model = Orion300MDecoder(config.model)
    initial_checkpoint_sha256 = None
    if initial_checkpoint is not None:
        source_state, initial_checkpoint_sha256 = load_checkpoint_model_weights(
            initial_checkpoint,
            expected_model="orion-300m",
            expected_model_config=config.model.model_dump(mode="json"),
        )
        model.load_state_dict(source_state, strict=True)
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )

    def loss_fn(network: Orion300MDecoder, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = network(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, _z_loss = assistant_only_sft_loss(
            output.logits,
            batch["input_ids"],
            batch["assistant_mask"],
            attention_mask=batch["attention_mask"],
            z_loss_coefficient=config.model.z_loss_coefficient,
        )
        return loss

    manager = CheckpointManager(Path(output_dir) / "checkpoints")
    run_identity = {
        "model": "orion-300m-sft",
        "training_config": config.model_dump(mode="json"),
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


def train_sft_jsonl(
    train_jsonl: str | Path,
    output_dir: str | Path,
    config: SFTTrainingConfig,
    *,
    initial_checkpoint: str | Path,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Run assistant-only supervised fine-tuning on explicit local tokenized data."""
    trainer, loader, dataset = _make_trainer(
        train_jsonl,
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
        "objective": "assistant_only_sft",
        "step": trainer.state.step,
        "epoch": trainer.state.epoch,
        "observed_losses": losses,
        "data_sha256": dataset.sha256,
    }


def evaluate_sft_jsonl(
    eval_jsonl: str | Path,
    checkpoint_dir: str | Path,
    config: SFTTrainingConfig,
    *,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate assistant-only loss on a local held-out SFT JSONL shard."""
    trainer, loader, dataset = _make_trainer(
        eval_jsonl, checkpoint_dir, config, initial_checkpoint=None, device=device
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value for key, value in trainer.checkpoint_metadata.items() if key != "data_sha256"
        }
    )
    loss = trainer.evaluate(loader, max_batches=max_batches)
    return {
        "status": "evaluated",
        "objective": "assistant_only_sft",
        "loss": loss,
        "data_sha256": dataset.sha256,
        "checkpoint": str(trainer.checkpoint_manager.latest()),
    }


__all__ = [
    "SFTJsonlDataset",
    "SFTTrainingConfig",
    "collate_sft_records",
    "evaluate_sft_jsonl",
    "render_chat_messages",
    "train_sft_jsonl",
    "write_sft_jsonl",
]
