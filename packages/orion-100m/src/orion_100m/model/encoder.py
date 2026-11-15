"""Bidirectional ORION-100M encoder with alternating local/global RoPE attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from orion_100m.config import Orion100MConfig
from orion_core.blocks.attention import MultiHeadAttention
from orion_core.blocks.feedforward import GeGLU
from orion_core.blocks.norms import LayerNormNoBias
from orion_core.errors import InputLimitError


@dataclass(frozen=True)
class EncoderOutput:
    """Encoder representations; ``pooled_output`` is masked mean pooling."""

    last_hidden_state: torch.Tensor
    pooled_output: torch.Tensor
    cls_and_mean: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


class EncoderBlock(nn.Module):
    """Pre-LayerNorm attention and GeGLU residual block."""

    def __init__(self, config: Orion100MConfig, layer_index: int) -> None:
        super().__init__()
        is_global = (layer_index + 1) % config.global_attention_every == 0
        self.attention_norm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            theta=config.rope_theta_global if is_global else config.rope_theta_local,
            local_window=None if is_global else config.local_window,
            dropout=config.dropout,
        )
        self.ffn_norm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = GeGLU(
            config.hidden_size, config.intermediate_size, dropout=config.dropout
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_out, _ = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
        )
        hidden = hidden + attention_out
        hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
        return hidden


class Orion100MEncoder(nn.Module):
    """ORION-100M body. The default body has exactly 100,291,840 trainable parameters."""

    def __init__(self, config: Orion100MConfig | None = None) -> None:
        super().__init__()
        self.config = config or Orion100MConfig()
        self.token_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=self.config.pad_token_id,
        )
        self.embedding_norm = LayerNormNoBias(
            self.config.hidden_size, eps=self.config.layer_norm_eps
        )
        self.layers = nn.ModuleList(
            [EncoderBlock(self.config, index) for index in range(self.config.num_layers)]
        )
        self.final_norm = LayerNormNoBias(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the plan's truncated-normal initialization and residual scaling."""
        scale = 0.02 / (2 * self.config.num_layers) ** 0.5
        layers = cast(list[EncoderBlock], list(self.layers))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = (
                    scale
                    if module
                    in {layer.attention.o_proj for layer in layers}
                    | {layer.feed_forward.down_proj for layer in layers}
                    else 0.02
                )
                if not module.weight.is_meta:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
                if module.bias is not None and not module.bias.is_meta:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and not module.weight.is_meta:
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            elif isinstance(module, LayerNormNoBias) and not module.weight.is_meta:
                nn.init.ones_(module.weight)

    def parameter_count(self) -> int:
        """Count unique trainable parameters in the encoder body."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        detach_embeddings: bool = False,
        output_hidden_states: bool = False,
    ) -> EncoderOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("input_ids must contain at least one token")
        if sequence_length > self.config.max_sequence_length:
            raise InputLimitError(
                f"Sequence has {sequence_length} tokens; limit is {self.config.max_sequence_length}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device).expand(batch, -1)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        if document_ids is not None and document_ids.shape != input_ids.shape:
            raise ValueError("document_ids must have the same shape as input_ids")

        token_embeds = self.token_embeddings(input_ids)
        if detach_embeddings:
            token_embeds = token_embeds.detach()
        hidden = self.embedding_norm(token_embeds)
        hidden_states: list[torch.Tensor] | None = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
            )
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(hidden)
        hidden = self.final_norm(hidden)
        valid = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        mean = (hidden * valid).sum(dim=1) / denominator
        cls_positions = input_ids.eq(self.config.cls_token_id).to(torch.int64).argmax(dim=1)
        cls = hidden[torch.arange(batch, device=input_ids.device), cls_positions]
        return EncoderOutput(
            last_hidden_state=hidden,
            pooled_output=mean,
            cls_and_mean=torch.cat((cls, mean), dim=-1),
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
