"""Warmup, stable, trapezoidal, and square-root decay learning-rate schedules."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _validate_schedule(warmup_steps: int, total_steps: int, min_ratio: float) -> None:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    if not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio must be in [0, 1]")


class WarmupStableDecaySchedule(LambdaLR):
    """Linear warmup, stable phase, then linear/cosine decay over the final fraction."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        decay_fraction: float = 0.15,
        min_ratio: float = 0.1,
        curve: str = "linear",
    ) -> None:
        _validate_schedule(warmup_steps, total_steps, min_ratio)
        if not 0 < decay_fraction <= 1:
            raise ValueError("decay_fraction must be in (0, 1]")
        if curve not in {"linear", "cosine", "sqrt"}:
            raise ValueError("curve must be linear, cosine, or sqrt")
        decay_steps = max(1, int(total_steps * decay_fraction))
        decay_start = total_steps - decay_steps

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            if step < decay_start:
                return 1.0
            progress = min(1.0, max(0.0, (step - decay_start) / max(1, total_steps - decay_start)))
            if curve == "cosine":
                decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif curve == "sqrt":
                decayed = 1.0 - math.sqrt(progress)
            else:
                decayed = 1.0 - progress
            return min_ratio + (1.0 - min_ratio) * decayed

        self.schedule_config = {
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "decay_fraction": decay_fraction,
            "min_ratio": min_ratio,
            "curve": curve,
        }
        super().__init__(optimizer, multiplier)

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["schedule_config"] = self.schedule_config
        return state


class TrapezoidalSchedule(WarmupStableDecaySchedule):
    """The encoder plan's warmup, stable-to-80%, and square-root tail schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        stable_until: float = 0.8,
        min_ratio: float = 1e-5,
    ) -> None:
        if not 0 < stable_until < 1:
            raise ValueError("stable_until must be in (0, 1)")
        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_fraction=1 - stable_until,
            min_ratio=min_ratio,
            curve="sqrt",
        )
        self.schedule_config["stable_until"] = stable_until
