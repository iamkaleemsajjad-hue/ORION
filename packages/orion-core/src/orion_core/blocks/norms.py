"""Normalization layers with the exact trainable parameters specified by ORION."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNormNoBias(nn.Module):
    """Layer normalization with a learned scale and no learned offset."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with one learned scale vector."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(input_dtype) * self.weight.to(input_dtype)
