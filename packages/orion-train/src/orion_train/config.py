"""Validated trainer configuration shared by the two independent model pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True, slots=True)
class DataLoaderOptions:
    """Validated worker settings passed to PyTorch's ``DataLoader``."""

    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool


class TrainerConfig(BaseModel):
    """Runtime limits and optimizer-independent settings for one training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(gt=0)
    grad_accum_steps: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    log_every_steps: int = Field(default=1, gt=0)
    checkpoint_every_steps: int = Field(default=1000, gt=0)
    checkpoint_dir: str | None = None
    num_workers: int = Field(default=0, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = False
    seed: int = 1729
    distributed_strategy: Literal["none", "ddp"] = "none"
    distributed_backend: Literal["auto", "gloo", "nccl"] = "auto"
    distributed_timeout_seconds: int = Field(default=1800, gt=0)
    loss_spike_factor: float = Field(default=5.0, gt=1)
    loss_ema_decay: float = Field(default=0.98, ge=0, lt=1)
    rollback_on_spike: bool = False
    lower_lr_on_rollback: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_precision(self) -> TrainerConfig:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers to be greater than zero")
        return self

    @property
    def data_loader_options(self) -> DataLoaderOptions:
        """Return only the worker options valid for the configured worker count."""
        return DataLoaderOptions(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers if self.num_workers else False,
        )
