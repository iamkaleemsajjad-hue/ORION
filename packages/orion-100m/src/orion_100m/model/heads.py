"""Classification, token extraction, confidence, and registry matching heads."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class SequenceClassificationHead(nn.Module):
    """Linear classifier over the concatenated CLS and masked-mean features."""

    def __init__(self, input_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        if num_labels < 2:
            raise ValueError("num_labels must be at least two")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(self.dropout(pooled)))


class MultiLabelHead(nn.Module):
    """Independent tag logits with optional per-label positive weights."""

    def __init__(self, input_size: int, num_labels: int) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(pooled))


class BIOExtractionHead(nn.Module):
    """Token-level BIO tagger; span decoding is kept deterministic in the SDK."""

    def __init__(self, hidden_size: int, num_tags: int = 25) -> None:
        super().__init__()
        if num_tags < 3:
            raise ValueError("BIO extraction requires at least three tags")
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(hidden_states))


class ConfidenceHead(nn.Module):
    """Predict probability that the selected task prediction is correct."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(pooled)).squeeze(-1)


class OpenSetMatcher(nn.Module):
    """Cosine matching against caller-supplied registry prototype vectors."""

    def __init__(self, embedding_size: int, initial_temperature: float = 0.07) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    def forward(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("embeddings and prototypes must be rank-two tensors")
        if embeddings.shape[-1] != prototypes.shape[-1]:
            raise ValueError("embedding and prototype dimensions must match")
        if embeddings.shape[-1] <= 0:
            raise ValueError("embedding dimensions must be non-empty")
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        return F.normalize(embeddings, dim=-1) @ F.normalize(prototypes, dim=-1).T / temperature

    @staticmethod
    def predict_unknown(
        similarities: torch.Tensor,
        *,
        min_similarity: float,
        energy_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class indices and an unknown flag from explicitly tuned thresholds."""
        if similarities.ndim != 2 or similarities.shape[-1] == 0:
            raise ValueError("similarities must have shape [batch, classes] with classes > 0")
        scores, indices = similarities.max(dim=-1)
        unknown = scores < min_similarity
        if energy_threshold is not None:
            energy = -torch.logsumexp(similarities, dim=-1)
            unknown |= energy > energy_threshold
        return indices, unknown


def temperature_scale(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Apply a positive scalar calibration temperature to logits."""
    value = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    if value.numel() != 1 or torch.any(value <= 0):
        raise ValueError("temperature must be a positive scalar")
    return logits / value
