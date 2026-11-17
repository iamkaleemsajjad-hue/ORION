"""Causal ORION-300M decoder with GQA, RoPE, RMSNorm, and tied embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from orion_300m.config import Orion300MConfig
from orion_core.blocks.attention import KeyValue, MultiHeadAttention, StaticKVCache
from orion_core.blocks.feedforward import SwiGLU
from orion_core.blocks.norms import RMSNorm
from orion_core.errors import InputLimitError


@dataclass(frozen=True)
class DecoderOutput:
    """Causal decoder outputs; logits share the token embedding parameter."""

    last_hidden_state: torch.Tensor
    logits: torch.Tensor
    past_key_values: tuple[KeyValue, ...] | None
    cache: StaticKVCache | None
    loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


class DecoderBlock(nn.Module):
    """Pre-RMSNorm causal GQA and SwiGLU residual block."""

    def __init__(self, config: Orion300MConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            num_kv_heads=config.num_kv_heads,
            theta=config.rope_theta,
            causal=True,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLU(
            config.hidden_size, config.intermediate_size, dropout=config.dropout
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        layer_index: int,
        cache: StaticKVCache | None,
        cache_position: int,
        past_key_value: KeyValue | None,
        use_cache: bool,
        causal_override: bool | None = None,
    ) -> tuple[torch.Tensor, KeyValue | None]:
        attention_out, present = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal=causal_override,
            past_key_value=past_key_value,
            use_cache=use_cache,
            static_cache=cache,
            cache_layer_index=layer_index if cache is not None else None,
            cache_position=cache_position if cache is not None else None,
        )
        hidden = hidden + attention_out
        hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
        return hidden, present


class Orion300MDecoder(nn.Module):
    """ORION-300M model body with exactly 306,763,200 unique parameters by default."""

    def __init__(self, config: Orion300MConfig | None = None) -> None:
        super().__init__()
        self.config = config or Orion300MConfig()
        self.token_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=self.config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [DecoderBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.gradient_checkpointing = False
        self.reset_parameters()

    def gradient_checkpointing_enable(self) -> None:
        """Recompute decoder blocks during training to reduce activation storage."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable decoder block recomputation."""
        self.gradient_checkpointing = False

    def reset_parameters(self) -> None:
        """Use truncated-normal initialization and scale residual output projections."""
        residual_std = 0.02 / (2 * self.config.num_layers) ** 0.5
        layers = cast(list[DecoderBlock], list(self.layers))
        residual_projections = {layer.attention.o_proj for layer in layers} | {
            layer.feed_forward.down_proj for layer in layers
        }
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = residual_std if module in residual_projections else 0.02
                if not module.weight.is_meta:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
                if module.bias is not None and not module.bias.is_meta:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and not module.weight.is_meta:
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            elif isinstance(module, RMSNorm) and not module.weight.is_meta:
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: tuple[KeyValue, ...] | None = None,
        cache: StaticKVCache | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        prefix_lengths: torch.Tensor | None = None,
    ) -> DecoderOutput:
        if input_ids.ndim != 2 or input_ids.shape[-1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
        if cache is not None and past_key_values is not None:
            raise ValueError("use either a static cache or dynamic past_key_values")
        if prefix_lengths is not None and (
            cache is not None or past_key_values is not None or use_cache
        ):
            raise ValueError("prefix-LM pilot mode does not support autoregressive KV caching")
        batch, query_length = input_ids.shape
        past_length = (
            cache.sequence_length
            if cache is not None
            else (0 if past_key_values is None else past_key_values[0].key.shape[-2])
        )
        total_length = past_length + query_length
        if total_length > self.config.context_length:
            raise InputLimitError(
                f"Prompt and cache contain {total_length} tokens; context limit is {self.config.context_length}"
            )
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if checkpointing and (cache is not None or past_key_values is not None or use_cache):
            raise ValueError("gradient checkpointing cannot be combined with a KV cache")
        if use_cache and cache is None and past_key_values is None:
            cache = StaticKVCache(self.config.num_layers, batch, self.config.context_length)
        if attention_mask is None:
            token_attention_mask = torch.ones(
                (batch, total_length), dtype=torch.bool, device=input_ids.device
            )
        elif attention_mask.shape != (batch, total_length):
            raise ValueError("attention_mask must have shape [batch, populated_sequence_length]")
        else:
            token_attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        layer_attention_mask: torch.Tensor = token_attention_mask
        causal_override = None
        if prefix_lengths is not None:
            if past_length or total_length != query_length:
                raise ValueError("prefix-LM pilot mode requires a full uncached sequence")
            if prefix_lengths.shape != (batch,):
                raise ValueError("prefix_lengths must have one value per batch row")
            prefix_lengths = prefix_lengths.to(device=input_ids.device, dtype=torch.long)
            if torch.any(prefix_lengths < 1) or torch.any(prefix_lengths >= query_length):
                raise ValueError("each prefix length must leave at least one target token")
            positions = torch.arange(query_length, device=input_ids.device)
            query_positions = positions.view(1, query_length, 1)
            key_positions = positions.view(1, 1, query_length)
            prefix_positions = prefix_lengths.view(batch, 1, 1)
            prefix_bidirectional = (query_positions < prefix_positions) & (
                key_positions < prefix_positions
            )
            target_causal = key_positions <= query_positions
            layer_attention_mask = (prefix_bidirectional | target_causal) & token_attention_mask[
                :, None, :
            ]
            causal_override = False
        if position_ids is None:
            position_ids = torch.arange(
                past_length, total_length, device=input_ids.device, dtype=torch.long
            ).expand(batch, -1)
        if position_ids.shape != (batch, query_length):
            raise ValueError("position_ids must have shape [batch, current_sequence_length]")

        hidden = self.token_embeddings(input_ids)
        hidden_states: list[torch.Tensor] | None = [hidden] if output_hidden_states else None
        presents = []
        for layer_index, layer in enumerate(cast(list[DecoderBlock], list(self.layers))):
            past = past_key_values[layer_index] if past_key_values is not None else None
            if checkpointing:

                def run_block(
                    value: torch.Tensor,
                    block: DecoderBlock = layer,
                    current_layer_index: int = layer_index,
                ) -> torch.Tensor:
                    output, _ = block(
                        value,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        layer_index=current_layer_index,
                        cache=None,
                        cache_position=past_length,
                        past_key_value=None,
                        use_cache=False,
                        causal_override=causal_override,
                    )
                    return cast(torch.Tensor, output)

                hidden = checkpoint(run_block, hidden, use_reentrant=False, preserve_rng_state=True)
                present = None
            else:
                hidden, present = layer(
                    hidden,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    layer_index=layer_index,
                    cache=cache,
                    cache_position=past_length,
                    past_key_value=past,
                    use_cache=use_cache and cache is None,
                    causal_override=causal_override,
                )
            if present is not None:
                presents.append(present)
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(hidden)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embeddings.weight)
        if cache is not None:
            cache.advance(total_length)
        loss = z_loss = None
        if labels is not None:
            loss, z_loss = causal_lm_loss(
                logits,
                input_ids,
                labels=labels,
                attention_mask=token_attention_mask,
                z_loss_coefficient=self.config.z_loss_coefficient,
            )
        return DecoderOutput(
            last_hidden_state=hidden,
            logits=logits,
            past_key_values=tuple(presents) if presents else None,
            cache=cache,
            loss=loss,
            z_loss=z_loss,
        )


def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    z_loss_coefficient: float = 1e-4,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token CE plus the planned squared-log-partition z-loss."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits and input_ids must be rank three and rank two")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logit sequence shape must match input_ids")
    target = input_ids if labels is None else labels
    if target.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")
    shifted_logits = logits[:, :-1, :].float()
    shifted_target = target[:, 1:].to(device=logits.device, dtype=torch.long)
    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
        shifted_target = shifted_target.masked_fill(~shifted_mask, ignore_index)
    else:
        shifted_mask = shifted_target.ne(ignore_index)
    shifted_mask = shifted_mask & shifted_target.ne(ignore_index)
    if shifted_logits.shape[1] == 0:
        raise ValueError("causal language modeling requires at least two sequence positions")
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_target.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shifted_target)
    valid_count = shifted_target.ne(ignore_index).sum().clamp_min(1)
    cross_entropy = token_loss.sum() / valid_count
    if z_loss_coefficient < 0:
        raise ValueError("z_loss_coefficient must be non-negative")
    log_partition = torch.logsumexp(shifted_logits, dim=-1)
    z_values = log_partition.square() * shifted_mask.to(log_partition.dtype)
    z_loss = z_values.sum() / shifted_mask.sum().clamp_min(1)
    return cross_entropy + z_loss_coefficient * z_loss, z_loss
