"""Core objective functions specified for the two ORION model families."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


def replaced_token_loss(
    logits: torch.Tensor, replaced: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1).float(), replaced.float(), reduction="none"
    )
    weights = attention_mask.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def electra_loss(
    generator_logits: torch.Tensor,
    generator_labels: torch.Tensor,
    discriminator_logits: torch.Tensor,
    replaced: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    discriminator_weight: float = 50.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator_loss = masked_lm_loss(generator_logits, generator_labels)
    discriminator_loss = replaced_token_loss(discriminator_logits, replaced, attention_mask)
    total = generator_loss + discriminator_weight * discriminator_loss
    return total, {"generator_mlm": generator_loss, "discriminator_rtd": discriminator_loss}


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    z_loss_weight: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    log_partition = torch.logsumexp(shifted_logits, dim=-1)
    valid = shifted_labels.ne(-100)
    z_loss = (log_partition.square() * valid).sum() / valid.sum().clamp_min(1)
    return ce + z_loss_weight * z_loss, {"next_token_ce": ce, "z_loss": z_loss}


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.05
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(embeddings.float(), dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positives = same & ~eye
    valid = ~eye
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(~valid, -torch.inf), dim=-1, keepdim=True
    )
    positive_count = positives.sum(dim=-1)
    usable = positive_count > 0
    per_example = -(log_prob.masked_fill(~positives, 0).sum(dim=-1) / positive_count.clamp_min(1))
    return per_example[usable].mean() if usable.any() else embeddings.sum() * 0.0
