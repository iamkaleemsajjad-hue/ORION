"""Bias-free gated feed-forward blocks used by each independently trained model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class GeGLU(nn.Module):
    """GeGLU feed-forward network: GELU(gate(x)) multiplied by up(x)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.gate_proj(x), approximate="none") * cast(torch.Tensor, self.up_proj(x))
        return cast(torch.Tensor, self.down_proj(self.dropout(gated)))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: SiLU(gate(x)) multiplied by up(x)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * cast(torch.Tensor, self.up_proj(x))
        return cast(torch.Tensor, self.down_proj(self.dropout(gated)))
