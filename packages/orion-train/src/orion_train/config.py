"""Validated trainer configuration shared by the two independent model pipelines."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainerConfig(BaseModel):
    """Runtime limits and optimizer-independent settings for one training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(gt=0)
    grad_accum_steps: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    precision: str = "fp32"
    log_every_steps: int = Field(default=1, gt=0)
    checkpoint_every_steps: int = Field(default=1000, gt=0)
    checkpoint_dir: str | None = None
    seed: int = 1729
    loss_spike_factor: float = Field(default=5.0, gt=1)
    loss_ema_decay: float = Field(default=0.98, ge=0, lt=1)
    rollback_on_spike: bool = False
    lower_lr_on_rollback: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_precision(self) -> TrainerConfig:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        return self
