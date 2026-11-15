"""Rotary positional embeddings for attention query and key vectors."""

from __future__ import annotations

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Generate rotary cosine and sine tables for explicit position indices."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even integer")
        if theta <= 0:
            raise ValueError("RoPE theta must be positive")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.theta = theta

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.to(self.inv_freq.device, torch.float32).unsqueeze(-1)
        phases = frequencies * self.inv_freq
        angles = torch.repeat_interleave(phases, repeats=2, dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``[batch, heads, length, head_dim]`` attention tensors."""
    while cos.ndim < tensor.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return tensor * cos + _rotate_half(tensor) * sin
