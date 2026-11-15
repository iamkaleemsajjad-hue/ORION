"""Explicit local-shard causal-LM training and evaluation for ORION-300M."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_data.dataset import TokenBlockDataset
from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything


class CausalTrainingConfig(BaseModel):
    """Validated hyperparameters for future local ORION-300M LM training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion300MConfig = Field(default_factory=Orion300MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=1000,
            checkpoint_dir="checkpoints/orion-300m",
        )
    )
    sequence_length: int = Field(default=1024, ge=2)
    batch_size: int = Field(default=4, ge=1, le=1024)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    scheduler: str = "wsd"

    @model_validator(mode="after")
    def dimensions_fit(self) -> CausalTrainingConfig:
        if self.sequence_length > self.model.context_length:
            raise ValueError("sequence_length exceeds decoder context_length")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if self.scheduler not in {"wsd", "trapezoidal"}:
            raise ValueError("scheduler must be wsd or trapezoidal")
        return self


def _make_training(
    token_shard: str | Path,
    output_dir: str | Path,
    config: CausalTrainingConfig,
    *,
    device: str,
) -> tuple[
    Trainer[Orion300MDecoder, dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    StatefulRandomSampler,
    TokenBlockDataset,
]:
    dataset = TokenBlockDataset(token_shard, sequence_length=config.sequence_length)
    sampler = StatefulRandomSampler(dataset, seed=config.trainer.seed)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, num_workers=0
    )
    seed_everything(config.trainer.seed)
    model = Orion300MDecoder(config.model)
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )
    manager = CheckpointManager(Path(output_dir) / "checkpoints")
    trainer: Trainer[Orion300MDecoder, dict[str, torch.Tensor]] = Trainer(
        model,
        optimizer,
        config=config.trainer,
        scheduler=scheduler,
        device=device,
        loss_function=lambda network, batch: (
            network(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            ).loss
        ),
        checkpoint_manager=manager,
        data_state_provider=lambda: {"sampler": sampler.state_dict()},
        data_state_restorer=lambda state: sampler.load_state_dict(state["sampler"]),
        checkpoint_metadata={
            "model": "orion-300m",
            "training_config": config.model_dump(mode="json"),
            "token_shard_sha256": dataset.sha256,
        },
    )
    return trainer, loader, sampler, dataset


def train_causal_lm(
    token_shard: str | Path,
    output_dir: str | Path,
    config: CausalTrainingConfig,
    *,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train causal LM weights on an explicit local pretokenized shard."""
    trainer, loader, _sampler, dataset = _make_training(
        token_shard, output_dir, config, device=device
    )
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("training checkpoint manager was not created")
    if resume:
        trainer.restore_checkpoint()
    elif manager.latest() is not None:
        raise FileExistsError("A checkpoint already exists; pass resume=True to continue it")
    losses = trainer.fit(loader)
    latest = manager.latest()
    latest_state = None
    if latest is not None:
        latest_state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
    if latest_state is None or int(latest_state.get("step", -1)) != trainer.state.step:
        trainer.save_checkpoint(metadata={"training_config": config.model_dump(mode="json")})
    return {
        "status": "training_completed",
        "step": trainer.state.step,
        "epoch": trainer.state.epoch,
        "tokens_seen": trainer.state.tokens_seen,
        "observed_losses": losses,
        "token_shard_sha256": dataset.sha256,
    }


def evaluate_causal_lm(
    token_shard: str | Path,
    checkpoint_dir: str | Path,
    config: CausalTrainingConfig,
    *,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a saved causal-LM checkpoint on explicit local tokens."""
    trainer, _training_loader, _sampler, dataset = _make_training(
        token_shard, checkpoint_dir, config, device=device
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value
            for key, value in trainer.checkpoint_metadata.items()
            if key != "token_shard_sha256"
        }
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    loss = trainer.evaluate(loader, max_batches=max_batches)
    manager = trainer.checkpoint_manager
    if manager is None:
        raise RuntimeError("evaluation checkpoint manager was not configured")
    return {
        "status": "evaluated",
        "loss": loss,
        "batches": min(len(loader), max_batches or len(loader)),
        "checkpoint": str(manager.latest()),
        "token_shard_sha256": dataset.sha256,
    }


__all__ = ["CausalTrainingConfig", "evaluate_causal_lm", "train_causal_lm"]
