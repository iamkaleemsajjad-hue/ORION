"""Trapezoidal and warmup-stable-decay LR schedules."""

from __future__ import annotations

import math

import torch


def trapezoidal_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_start_fraction: float = 0.8,
    final_multiplier: float = 0.0125,
) -> float:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("total_steps must be positive and warmup_steps must be in [0,total_steps)")
    if not 0 <= decay_start_fraction < 1 or not 0 <= final_multiplier <= 1:
        raise ValueError("invalid trapezoidal schedule fractions")
    if step < warmup_steps:
        return max(0.0, step / max(1, warmup_steps))
    decay_start = max(warmup_steps, int(total_steps * decay_start_fraction))
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return final_multiplier + (1.0 - final_multiplier) * (1.0 - math.sqrt(progress))


def wsd_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_start_fraction: float = 0.85,
    final_multiplier: float = 0.1,
) -> float:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("total_steps must be positive and warmup_steps must be in [0,total_steps)")
    if not 0 <= decay_start_fraction < 1 or not 0 <= final_multiplier <= 1:
        raise ValueError("invalid WSD schedule fractions")
    if step < warmup_steps:
        return max(0.0, step / max(1, warmup_steps))
    decay_start = max(warmup_steps, int(total_steps * decay_start_fraction))
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return final_multiplier + (1 - final_multiplier) * cosine


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    total_steps: int,
    warmup_steps: int,
    **schedule_kwargs,
) -> torch.optim.lr_scheduler.LambdaLR:
    if kind not in {"trapezoidal", "wsd"}:
        raise ValueError("kind must be 'trapezoidal' or 'wsd'")
    function = trapezoidal_multiplier if kind == "trapezoidal" else wsd_multiplier
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: function(
            step, total_steps=total_steps, warmup_steps=warmup_steps, **schedule_kwargs
        ),
    )
