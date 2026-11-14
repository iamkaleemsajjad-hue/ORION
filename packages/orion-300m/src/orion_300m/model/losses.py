"""Causal language modeling and assistant-only SFT objective helpers."""

from __future__ import annotations

import torch

from orion_300m.model.decoder import causal_lm_loss


def assistant_only_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    z_loss_coefficient: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply next-token loss only where target tokens are inside assistant turns."""
    if assistant_mask.shape != input_ids.shape:
        raise ValueError("assistant_mask must match input_ids")
    target = input_ids.masked_fill(
        ~assistant_mask.to(dtype=torch.bool, device=input_ids.device), -100
    )
    return causal_lm_loss(
        logits,
        input_ids,
        labels=target,
        attention_mask=attention_mask,
        z_loss_coefficient=z_loss_coefficient,
    )
