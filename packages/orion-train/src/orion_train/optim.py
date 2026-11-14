"""AdamW parameter grouping with decay omitted for norms and embeddings."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def build_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Separate matrix weights from bias, norm, and embedding parameters."""
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            name.endswith("bias")
            or "norm" in lowered
            or "embedding" in lowered
            or lowered.endswith(".emb.weight")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def build_adamw(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    extra_groups: Iterable[dict[str, object]] | None = None,
) -> torch.optim.AdamW:
    """Create AdamW with explicit groups and independently configurable betas/epsilon."""
    if lr <= 0 or eps <= 0 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError("invalid AdamW hyperparameters")
    groups: list[dict[str, object]] = build_parameter_groups(model, weight_decay=weight_decay)
    if extra_groups is not None:
        groups.extend(extra_groups)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
