"""Multi-head and grouped-query self-attention using PyTorch SDPA."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from orion_core.blocks.rotary import RotaryEmbedding, apply_rotary


class KeyValue(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor


class StaticKVCache:
    """Lazy, preallocated K/V buffers for incremental autoregressive decoding."""

    def __init__(self, num_layers: int, max_batch_size: int, max_length: int) -> None:
        if min(num_layers, max_batch_size, max_length) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.sequence_length = 0
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write current key/value vectors and return the populated prefix view."""
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("cache layer index is out of range")
        if cache_position != self.sequence_length:
            raise ValueError(
                f"cache_position {cache_position} does not match populated length {self.sequence_length}"
            )
        batch, heads, length, width = key.shape
        end = cache_position + length
        if batch > self.max_batch_size or end > self.max_length:
            raise ValueError("KV cache capacity exceeded")
        if value.shape != key.shape:
            raise ValueError("key and value cache tensors must have the same shape")
        key_buffer = self._keys[layer_index]
        value_buffer = self._values[layer_index]
        expected_shape = (self.max_batch_size, heads, self.max_length, width)
        if key_buffer is None:
            key_buffer = torch.empty(expected_shape, dtype=key.dtype, device=key.device)
            value_buffer = torch.empty(expected_shape, dtype=value.dtype, device=value.device)
            self._keys[layer_index] = key_buffer
            self._values[layer_index] = value_buffer
        elif (
            tuple(key_buffer.shape) != expected_shape
            or key_buffer.device != key.device
            or key_buffer.dtype != key.dtype
            or value_buffer is None
            or value_buffer.dtype != value.dtype
        ):
            raise ValueError("KV cache batch, head, device, or dtype changed during generation")
        key_buffer[:batch, :, cache_position:end].copy_(key)
        assert value_buffer is not None
        value_buffer[:batch, :, cache_position:end].copy_(value)
        return key_buffer[:batch, :, :end], value_buffer[:batch, :, :end]

    def advance(self, new_length: int) -> None:
        """Mark a sequence range as populated after every layer wrote its cache."""
        if not self.sequence_length <= new_length <= self.max_length:
            raise ValueError("invalid cache sequence length")
        self.sequence_length = new_length

    def reset(self) -> None:
        """Reuse allocated buffers for a new sequence."""
        self.sequence_length = 0


def _allowed_attention(
    *,
    batch: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    attention_mask: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    causal: bool,
    local_window: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    if query_positions.ndim == 1:
        query_grid = query_positions.view(1, 1, -1, 1)
    elif query_positions.ndim == 2 and query_positions.shape[0] == batch:
        query_grid = query_positions.view(batch, 1, -1, 1)
    else:
        raise ValueError("query positions must be [sequence] or [batch, sequence]")
    key_grid = key_positions.view(1, 1, 1, -1)
    allowed: torch.Tensor | None = None
    if causal:
        allowed = key_grid <= query_grid
    if local_window is not None:
        if causal:
            local = (query_grid - key_grid) < local_window
            local &= key_grid <= query_grid
        else:
            radius_left = (local_window - 1) // 2
            radius_right = local_window - 1 - radius_left
            delta = key_grid - query_grid
            local = (delta >= -radius_left) & (delta <= radius_right)
        allowed = local if allowed is None else allowed & local
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            if attention_mask.shape != (batch, key_positions.numel()):
                raise ValueError("2D attention_mask must have shape [batch, key_length]")
            key_allowed = attention_mask.to(device=device, dtype=torch.bool).view(batch, 1, 1, -1)
        elif attention_mask.ndim == 3:
            if attention_mask.shape != (batch, query_grid.shape[-2], key_positions.numel()):
                raise ValueError(
                    "3D attention_mask must have shape [batch, query_length, key_length]"
                )
            key_allowed = attention_mask.to(device=device, dtype=torch.bool).unsqueeze(1)
        elif attention_mask.ndim == 4:
            if attention_mask.shape[0] != batch or attention_mask.shape[-2:] != (
                query_grid.shape[-2],
                key_positions.numel(),
            ):
                raise ValueError("4D attention_mask has incompatible batch/query/key dimensions")
            key_allowed = attention_mask.to(device=device, dtype=torch.bool)
        else:
            raise ValueError("attention_mask must be 2D, 3D, or 4D")
        allowed = key_allowed if allowed is None else allowed & key_allowed
    if document_ids is not None:
        key_length = key_positions.numel()
        if document_ids.shape != (batch, key_length):
            raise ValueError("document_ids must have shape [batch, key_length]")
        query_length = query_grid.shape[-2]
        query_documents = document_ids[:, -query_length:].to(device=device)
        key_documents = document_ids.to(device=device)
        same_document = query_documents[:, None, :, None].eq(key_documents[:, None, None, :])
        same_document &= query_documents[:, None, :, None].ge(0)
        same_document &= key_documents[:, None, None, :].ge(0)
        allowed = same_document if allowed is None else allowed & same_document
    if allowed is None:
        return None
    return allowed


class MultiHeadAttention(nn.Module):
    """Self-attention with optional GQA, RoPE, causality, locality, and KV caching."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        theta: float = 10000.0,
        causal: bool = False,
        local_window: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if local_window is not None and local_window <= 0:
            raise ValueError("local_window must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.local_window = local_window
        self.dropout_p = dropout
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.rope = RotaryEmbedding(self.head_dim, theta=theta) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal: bool | None = None,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
        static_cache: StaticKVCache | None = None,
        cache_layer_index: int | None = None,
        cache_position: int | None = None,
    ) -> tuple[torch.Tensor, KeyValue | None]:
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError("x must have shape [batch, sequence, hidden_size]")
        batch, query_length, _ = x.shape
        if static_cache is not None and past_key_value is not None:
            raise ValueError("use either static_cache or past_key_value, not both")
        past_length = (
            static_cache.sequence_length
            if static_cache is not None
            else (0 if past_key_value is None else past_key_value.key.shape[-2])
        )
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + query_length, device=x.device, dtype=torch.long
            ).expand(batch, -1)
        if position_ids.shape != (batch, query_length):
            raise ValueError("position_ids must have shape [batch, query_length]")

        query = (
            self.q_proj(x).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key = (
            self.k_proj(x)
            .view(batch, query_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(x)
            .view(batch, query_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.rope is not None:
            q_cos, q_sin = self.rope(position_ids, dtype=query.dtype)
            query = apply_rotary(query, q_cos, q_sin)
            k_cos, k_sin = self.rope(position_ids, dtype=key.dtype)
            key = apply_rotary(key, k_cos, k_sin)
        if static_cache is not None:
            if cache_layer_index is None:
                raise ValueError("cache_layer_index is required with static_cache")
            key, value = static_cache.update(
                cache_layer_index,
                key,
                value,
                cache_position=past_length if cache_position is None else cache_position,
            )
        elif past_key_value is not None:
            key = torch.cat((past_key_value.key, key), dim=-2)
            value = torch.cat((past_key_value.value, value), dim=-2)
        present = KeyValue(key, value) if use_cache else None

        key_length = key.shape[-2]
        key_positions = torch.arange(key_length, device=x.device)
        query_positions = position_ids
        allowed = _allowed_attention(
            batch=batch,
            query_positions=query_positions,
            key_positions=key_positions,
            attention_mask=attention_mask,
            document_ids=document_ids,
            causal=self.causal if causal is None else causal,
            local_window=self.local_window,
            device=x.device,
        )
        dropout_p = self.dropout_p if self.training else 0.0
        if self.num_heads == self.num_kv_heads:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            try:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=dropout_p,
                    is_causal=False,
                    enable_gqa=True,
                )
            except TypeError as exc:
                if "enable_gqa" not in str(exc):
                    raise
                repeat_factor = self.num_heads // self.num_kv_heads
                attended = F.scaled_dot_product_attention(
                    query,
                    key.repeat_interleave(repeat_factor, dim=1),
                    value.repeat_interleave(repeat_factor, dim=1),
                    attn_mask=allowed,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        attended = attended.transpose(1, 2).contiguous().view(batch, query_length, self.hidden_size)
        return self.o_proj(attended), present
