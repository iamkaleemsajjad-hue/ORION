"""Loss functions for RTD pretraining, contrastive learning, and task heads."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_mean(losses: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return losses.mean()
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def replaced_token_detection_loss(
    logits: torch.Tensor,
    replaced_labels: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy over valid positions, where one means replaced."""
    if logits.shape != replaced_labels.shape:
        raise ValueError("RTD logits and labels must have the same shape")
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), replaced_labels.to(logits.dtype), reduction="none"
    )
    return _masked_mean(losses, attention_mask)


def masked_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy for selected masked tokens; unselected labels use ignore_index."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("MLM logits must have one extra vocabulary dimension")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index
    )


def contrastive_info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE for aligned positive pairs."""
    if anchors.shape != positives.shape or anchors.ndim != 2:
        raise ValueError("anchors and positives must have the same [batch, dimension] shape")
    if anchors.shape[0] < 2:
        raise ValueError("in-batch contrastive loss requires at least two pairs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left = F.normalize(anchors, dim=-1)
    right = F.normalize(positives, dim=-1)
    scores = left @ right.T / temperature
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))


def matryoshka_info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    *,
    dimensions: tuple[int, ...] = (640, 256, 128),
    weights: tuple[float, ...] | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Apply symmetric in-batch InfoNCE at each configured embedding prefix."""
    if anchors.shape != positives.shape or anchors.ndim != 2:
        raise ValueError("anchors and positives must have the same [batch, dimension] shape")
    if not dimensions or any(
        dimension <= 0 or dimension > anchors.shape[-1] for dimension in dimensions
    ):
        raise ValueError("Matryoshka dimensions must be positive prefixes of the embedding width")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Matryoshka dimensions must be unique")
    coefficients = weights or tuple(1.0 for _ in dimensions)
    if len(coefficients) != len(dimensions) or any(weight < 0 for weight in coefficients):
        raise ValueError("Matryoshka weights must be non-negative and match dimensions")
    if not any(coefficients):
        raise ValueError("at least one Matryoshka dimension weight must be positive")
    losses = [
        weight
        * contrastive_info_nce(
            anchors[:, :dimension], positives[:, :dimension], temperature=temperature
        )
        for dimension, weight in zip(dimensions, coefficients, strict=True)
    ]
    return torch.stack(losses).sum() / sum(coefficients)


def course_open_set_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    *,
    cosine_scores: torch.Tensor,
    unknown_margin: float = 0.3,
    unknown_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train known-course ranking and explicitly labeled unknown rejection.

    ``labels`` contains candidate indices for known courses and ``-1`` for
    unknown documents. The margin is applied to unscaled cosine scores so the
    learned matcher temperature cannot redefine the open-set operating point.
    """
    if similarities.ndim != 2 or similarities.shape[-1] < 2:
        raise ValueError(
            "similarities must have shape [batch, candidates] with at least two courses"
        )
    if cosine_scores.shape != similarities.shape:
        raise ValueError("cosine_scores must match similarities")
    if labels.shape != (similarities.shape[0],):
        raise ValueError("labels must have one course index per document")
    if not -1.0 <= unknown_margin <= 1.0:
        raise ValueError("unknown_margin must be between -1 and 1")
    if unknown_weight < 0:
        raise ValueError("unknown_weight must be non-negative")
    labels = labels.to(device=similarities.device, dtype=torch.long)
    if torch.any(labels < -1) or torch.any(labels >= similarities.shape[-1]):
        raise ValueError("course labels must be candidate indices or -1 for unknown")
    known = labels >= 0
    unknown = ~known
    losses: dict[str, torch.Tensor] = {}
    if known.any():
        losses["known_course"] = F.cross_entropy(similarities[known].float(), labels[known])
    if unknown.any():
        maximum_known_similarity = cosine_scores[unknown].float().max(dim=-1).values
        losses["unknown_margin"] = F.relu(maximum_known_similarity - unknown_margin).square().mean()
    if not losses:
        raise ValueError("course matching batch contains no labeled examples")
    total = losses.get("known_course", similarities.sum() * 0.0)
    total = total + unknown_weight * losses.get("unknown_margin", similarities.sum() * 0.0)
    return total, losses


def confidence_loss(logits: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:
    """Binary loss for a confidence head trained on held-out correctness labels."""
    if logits.shape != correctness.shape:
        raise ValueError("confidence logits and correctness labels must match")
    return F.binary_cross_entropy_with_logits(logits.float(), correctness.to(logits.dtype))
