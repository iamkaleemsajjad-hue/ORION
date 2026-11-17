"""Explicit local-shard RTD training and evaluation entry points for ORION-100M."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader

from orion_100m.config import Orion100MConfig
from orion_100m.model.pretraining import Orion100MRTDPretrainer
from orion_data.dataset import TokenBlockDataset
from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.optim import build_adamw
from orion_train.sampler import StatefulRandomSampler
from orion_train.schedulers import build_lr_scheduler
from orion_train.trainer import Trainer, seed_everything


class RTDTrainingConfig(BaseModel):
    """Validated hyperparameters for future local ORION-100M RTD training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Orion100MConfig = Field(default_factory=Orion100MConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            max_steps=1000,
            checkpoint_every_steps=1000,
            checkpoint_dir="checkpoints/orion-100m",
        )
    )
    sequence_length: int = Field(default=512, ge=2)
    batch_size: int = Field(default=8, ge=1, le=1024)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    scheduler: str = "wsd"
    masking_granularity: Literal["whole_word", "token"] = "whole_word"
    rtd_weight: float = Field(default=50.0, gt=0)
    mask_rate: float = Field(default=0.2, gt=0, le=1)
    generator_layers: int = Field(default=8, ge=1)
    generator_hidden_size: int = Field(default=384, ge=8)
    generator_heads: int = Field(default=6, ge=1)
    generator_intermediate_size: int = Field(default=1024, ge=8)

    @model_validator(mode="after")
    def dimensions_fit(self) -> RTDTrainingConfig:
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length exceeds the encoder maximum")
        if self.generator_hidden_size % self.generator_heads:
            raise ValueError("generator_hidden_size must be divisible by generator_heads")
        if self.warmup_steps >= self.trainer.max_steps:
            raise ValueError("warmup_steps must be less than trainer.max_steps")
        if self.scheduler not in {"wsd", "trapezoidal"}:
            raise ValueError("scheduler must be wsd or trapezoidal")
        return self


def _make_training(
    token_shard: str | Path,
    word_ids_path: str | Path | None,
    output_dir: str | Path,
    config: RTDTrainingConfig,
    *,
    device: str,
) -> tuple[
    Trainer[Orion100MRTDPretrainer, dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    StatefulRandomSampler,
    TokenBlockDataset,
]:
    dataset = TokenBlockDataset(
        token_shard, sequence_length=config.sequence_length, word_ids_path=word_ids_path
    )
    sampler = StatefulRandomSampler(dataset, seed=config.trainer.seed)
    loader_options = config.trainer.data_loader_options
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=loader_options.num_workers,
        prefetch_factor=loader_options.prefetch_factor,
        persistent_workers=loader_options.persistent_workers,
    )
    seed_everything(config.trainer.seed)
    model = Orion100MRTDPretrainer(
        config.model,
        generator_layers=config.generator_layers,
        generator_hidden_size=config.generator_hidden_size,
        generator_heads=config.generator_heads,
        generator_intermediate_size=config.generator_intermediate_size,
        rtd_weight=config.rtd_weight,
        mask_rate=config.mask_rate,
        pad_token_id=config.model.pad_token_id,
    )
    optimizer = build_adamw(model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        kind=config.scheduler,
        total_steps=config.trainer.max_steps,
        warmup_steps=config.warmup_steps,
    )
    manager = CheckpointManager(Path(output_dir) / "checkpoints")
    trainer: Trainer[Orion100MRTDPretrainer, dict[str, torch.Tensor]] = Trainer(
        model,
        optimizer,
        config=config.trainer,
        scheduler=scheduler,
        device=device,
        loss_function=lambda network, batch: (
            network(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_ids=(
                    batch.get("word_ids") if config.masking_granularity == "whole_word" else None
                ),
            ).loss
        ),
        checkpoint_manager=manager,
        data_state_provider=lambda: {"sampler": sampler.state_dict()},
        data_state_restorer=lambda state: sampler.load_state_dict(state["sampler"]),
        checkpoint_metadata={
            "model": "orion-100m",
            "training_config": config.model_dump(mode="json"),
            "token_shard_sha256": dataset.sha256,
            "word_ids_sha256": dataset.word_ids_sha256,
        },
    )
    return trainer, loader, sampler, dataset


def train_rtd(
    token_shard: str | Path,
    output_dir: str | Path,
    config: RTDTrainingConfig,
    *,
    word_ids_path: str | Path | None = None,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Run the configured RTD objective on explicit pretokenized local data."""
    if config.masking_granularity == "whole_word" and word_ids_path is None:
        raise ValueError("whole-word RTD training requires a tokenizer-aligned word_ids.i32 shard")
    trainer, loader, _sampler, dataset = _make_training(
        token_shard, word_ids_path, output_dir, config, device=device
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
        import json

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
        "word_ids_sha256": dataset.word_ids_sha256,
    }


def evaluate_rtd(
    token_shard: str | Path,
    checkpoint_dir: str | Path,
    config: RTDTrainingConfig,
    *,
    word_ids_path: str | Path | None = None,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a saved RTD checkpoint on an explicit local token shard."""
    if config.masking_granularity == "whole_word" and word_ids_path is None:
        raise ValueError("whole-word RTD evaluation requires its aligned word_ids.i32 shard")
    trainer, _training_loader, _sampler, dataset = _make_training(
        token_shard, word_ids_path, checkpoint_dir, config, device=device
    )
    trainer.data_state_restorer = None
    trainer.restore_checkpoint(
        expected_run_identity={
            key: value
            for key, value in trainer.checkpoint_metadata.items()
            if key not in {"token_shard_sha256", "word_ids_sha256"}
        }
    )
    loader_options = config.trainer.data_loader_options
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=loader_options.num_workers,
        prefetch_factor=loader_options.prefetch_factor,
        persistent_workers=loader_options.persistent_workers,
    )
    seed_everything(config.trainer.seed)
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


__all__ = ["RTDTrainingConfig", "evaluate_rtd", "train_rtd"]
