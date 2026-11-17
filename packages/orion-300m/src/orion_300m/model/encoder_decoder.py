"""Pilot-scale tied-embedding encoder-decoder alternative for E-300-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from orion_300m.config import Orion300MConfig
from orion_core.blocks.attention import MultiHeadAttention
from orion_core.blocks.feedforward import SwiGLU
from orion_core.blocks.norms import RMSNorm
from orion_core.errors import InputLimitError


@dataclass(frozen=True)
class EncoderDecoderOutput:
    """Sequence-to-sequence outputs from the experimental alternative model."""

    encoder_last_hidden_state: torch.Tensor
    last_hidden_state: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


class CrossAttention(nn.Module):
    """GQA cross-attention from decoder queries to encoded source tokens."""

    def __init__(self, config: Orion300MConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout_p = config.dropout
        self.query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        memory_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("cross-attention inputs must have [batch, sequence, hidden] shapes")
        batch, target_length, hidden = query_states.shape
        if hidden != self.hidden_size or memory_states.shape[0] != batch:
            raise ValueError("cross-attention batch or hidden dimensions do not match")
        source_length = memory_states.shape[1]
        if memory_attention_mask.shape != (batch, source_length):
            raise ValueError("memory_attention_mask must match the source sequence")
        query = (
            self.query(query_states)
            .view(batch, target_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(memory_states)
            .view(batch, source_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(memory_states)
            .view(batch, source_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        allowed = memory_attention_mask.to(device=query.device, dtype=torch.bool)[:, None, None, :]
        dropout_p = self.dropout_p if self.training else 0.0
        if self.num_heads == self.num_kv_heads:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=dropout_p
            )
        else:
            try:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=dropout_p,
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
                )
        attended = attended.transpose(1, 2).contiguous().view(batch, target_length, hidden)
        return cast(torch.Tensor, self.output(attended))


class EncoderBlock(nn.Module):
    """Bidirectional self-attention and SwiGLU in a pre-RMSNorm block."""

    def __init__(self, config: Orion300MConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            num_kv_heads=config.num_kv_heads,
            theta=config.rope_theta,
            causal=False,
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
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden = hidden + attended
        return hidden + cast(torch.Tensor, self.feed_forward(self.ffn_norm(hidden)))


class DecoderBlock(nn.Module):
    """Causal self-attention followed by source cross-attention and SwiGLU."""

    def __init__(self, config: Orion300MConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            num_kv_heads=config.num_kv_heads,
            theta=config.rope_theta,
            causal=True,
            dropout=config.dropout,
        )
        self.cross_query_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_memory_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attention = CrossAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLU(
            config.hidden_size, config.intermediate_size, dropout=config.dropout
        )

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        target_attention_mask: torch.Tensor,
        source_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.self_attention(
            self.self_norm(hidden),
            attention_mask=target_attention_mask,
            position_ids=position_ids,
        )
        hidden = hidden + attended
        hidden = hidden + cast(
            torch.Tensor,
            self.cross_attention(
                self.cross_query_norm(hidden),
                self.cross_memory_norm(memory),
                memory_attention_mask=source_attention_mask,
            ),
        )
        return hidden + cast(torch.Tensor, self.feed_forward(self.ffn_norm(hidden)))


class Orion300MEncoderDecoderPilot(nn.Module):
    """Experimental seq2seq model; it is not the selected 300M release model."""

    def __init__(
        self,
        config: Orion300MConfig,
        *,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()
        if encoder_layers <= 0 or decoder_layers <= 0:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        self.config = config
        self.encoder_layer_count = encoder_layers
        self.decoder_layer_count = decoder_layers
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(encoder_layers)])
        self.encoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(decoder_layers)])
        self.decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.gradient_checkpointing = False
        self.reset_parameters()

    def gradient_checkpointing_enable(self) -> None:
        """Recompute pilot transformer blocks during training."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable pilot block recomputation."""
        self.gradient_checkpointing = False

    def reset_parameters(self) -> None:
        """Initialize the pilot from scratch and scale residual projections."""
        depth = self.encoder_layer_count + self.decoder_layer_count
        residual_std = 0.02 / (2 * depth) ** 0.5
        encoder_layers = cast(list[EncoderBlock], list(self.encoder))
        decoder_layers = cast(list[DecoderBlock], list(self.decoder))
        residual_projections = {layer.attention.o_proj for layer in encoder_layers} | {
            layer.feed_forward.down_proj for layer in encoder_layers
        }
        for layer in decoder_layers:
            residual_projections.update(
                {
                    layer.self_attention.o_proj,
                    layer.cross_attention.output,
                    layer.feed_forward.down_proj,
                }
            )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head:
                    continue
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
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        source_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> EncoderDecoderOutput:
        if (
            source_ids.ndim != 2
            or decoder_input_ids.ndim != 2
            or source_ids.shape[0] != decoder_input_ids.shape[0]
            or source_ids.shape[1] == 0
            or decoder_input_ids.shape[1] == 0
        ):
            raise ValueError("source_ids and decoder_input_ids must be non-empty rank-two batches")
        batch, source_length = source_ids.shape
        target_length = decoder_input_ids.shape[1]
        if max(source_length, target_length) > self.config.context_length:
            raise InputLimitError("source or target exceeds the pilot context length")
        if source_attention_mask is None:
            source_attention_mask = source_ids.ne(self.config.pad_token_id)
        if target_attention_mask is None:
            target_attention_mask = decoder_input_ids.ne(self.config.pad_token_id)
        if source_attention_mask.shape != source_ids.shape:
            raise ValueError("source_attention_mask must match source_ids")
        if target_attention_mask.shape != decoder_input_ids.shape:
            raise ValueError("target_attention_mask must match decoder_input_ids")
        source_attention_mask = source_attention_mask.to(device=source_ids.device, dtype=torch.bool)
        target_attention_mask = target_attention_mask.to(
            device=decoder_input_ids.device, dtype=torch.bool
        )
        if not source_attention_mask.any(dim=-1).all():
            raise ValueError("every encoder source must contain at least one unmasked token")
        if not target_attention_mask.any(dim=-1).all():
            raise ValueError("every decoder input must contain at least one unmasked token")
        source_positions = torch.arange(source_length, device=source_ids.device).expand(batch, -1)
        target_positions = torch.arange(target_length, device=decoder_input_ids.device).expand(
            batch, -1
        )
        memory = self.token_embeddings(source_ids)
        for encoder_layer in cast(list[EncoderBlock], list(self.encoder)):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                def run_encoder_block(
                    value: torch.Tensor, block: EncoderBlock = encoder_layer
                ) -> torch.Tensor:
                    return cast(
                        torch.Tensor,
                        block(
                            value,
                            attention_mask=source_attention_mask,
                            position_ids=source_positions,
                        ),
                    )

                memory = checkpoint(
                    run_encoder_block, memory, use_reentrant=False, preserve_rng_state=True
                )
            else:
                memory = encoder_layer(
                    memory,
                    attention_mask=source_attention_mask,
                    position_ids=source_positions,
                )
        memory = self.encoder_norm(memory)
        hidden = self.token_embeddings(decoder_input_ids)
        for decoder_layer in cast(list[DecoderBlock], list(self.decoder)):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                def run_decoder_block(
                    value: torch.Tensor,
                    encoded: torch.Tensor,
                    block: DecoderBlock = decoder_layer,
                ) -> torch.Tensor:
                    return cast(
                        torch.Tensor,
                        block(
                            value,
                            encoded,
                            target_attention_mask=target_attention_mask,
                            source_attention_mask=source_attention_mask,
                            position_ids=target_positions,
                        ),
                    )

                hidden = checkpoint(
                    run_decoder_block,
                    hidden,
                    memory,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden = decoder_layer(
                    hidden,
                    memory,
                    target_attention_mask=target_attention_mask,
                    source_attention_mask=source_attention_mask,
                    position_ids=target_positions,
                )
        hidden = self.decoder_norm(hidden)
        logits = F.linear(hidden, self.token_embeddings.weight)
        loss = z_loss = None
        if labels is not None:
            if labels.shape != decoder_input_ids.shape:
                raise ValueError("labels must align with decoder_input_ids")
            targets = labels.to(device=logits.device, dtype=torch.long)
            targets = targets.masked_fill(~target_attention_mask, -100)
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(targets)
            valid = targets.ne(-100)
            valid_count = valid.sum().clamp_min(1)
            loss = token_losses.sum() / valid_count
            log_partition = torch.logsumexp(logits.float(), dim=-1)
            z_loss = (log_partition.square() * valid).sum() / valid_count
            loss = loss + self.config.z_loss_coefficient * z_loss
        return EncoderDecoderOutput(
            encoder_last_hidden_state=memory,
            last_hidden_state=hidden,
            logits=logits,
            loss=loss,
            z_loss=z_loss,
        )
