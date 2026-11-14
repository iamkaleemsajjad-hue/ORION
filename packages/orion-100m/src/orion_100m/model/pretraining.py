"""ELECTRA-style generator/discriminator pretraining with disentangled embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import Orion100MEncoder
from orion_100m.model.losses import masked_language_model_loss, replaced_token_detection_loss
from orion_core.blocks.attention import MultiHeadAttention
from orion_core.blocks.feedforward import GeGLU
from orion_core.blocks.norms import LayerNormNoBias


def whole_word_mask(
    input_ids: torch.Tensor,
    *,
    word_ids: torch.Tensor | None,
    attention_mask: torch.Tensor,
    mask_token_id: int = 4,
    mask_rate: float = 0.2,
    special_id_limit: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select whole tokenizer word groups, protecting special tokens and padding."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have equal [batch, sequence] shapes")
    if not 0 < mask_rate <= 1:
        raise ValueError("mask_rate must be in (0, 1]")
    if word_ids is not None and word_ids.shape != input_ids.shape:
        raise ValueError("word_ids must have the same shape as input_ids")
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(input_ids.shape[0]):
        eligible = attention_mask[row].bool() & input_ids[row].ge(special_id_limit)
        if word_ids is None:
            groups = [
                [int(position)] for position in eligible.nonzero(as_tuple=False).flatten().tolist()
            ]
        else:
            group_map: dict[int, list[int]] = {}
            for position in eligible.nonzero(as_tuple=False).flatten().tolist():
                word_id = int(word_ids[row, position].item())
                if word_id < 0:
                    groups = [
                        [int(position)]
                        for position in eligible.nonzero(as_tuple=False).flatten().tolist()
                    ]
                    break
                group_map.setdefault(word_id, []).append(int(position))
            else:
                groups = list(group_map.values())
        if not groups:
            continue
        choices = torch.rand(len(groups), device=input_ids.device) < mask_rate
        if not choices.any():
            choices[torch.randint(len(groups), (), device=input_ids.device)] = True
        for group, chosen in zip(groups, choices.tolist(), strict=True):
            if chosen:
                selected[row, group] = True
    masked_ids = input_ids.clone()
    if not selected.any():
        raise ValueError("RTD batch contains no maskable non-special tokens")
    masked_ids[selected] = mask_token_id
    labels = input_ids.masked_fill(~selected, -100)
    return masked_ids, labels


class GeneratorBlock(nn.Module):
    """Small bidirectional generator block used only during ORION-100M RTD training."""

    def __init__(self, hidden_size: int, heads: int, ffn_size: int) -> None:
        super().__init__()
        self.attention_norm = LayerNormNoBias(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, heads)
        self.ffn_norm = LayerNormNoBias(hidden_size)
        self.ffn = GeGLU(hidden_size, ffn_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.attention_norm(x), attention_mask=attention_mask)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


class SmallMLMGenerator(nn.Module):
    """Compact MLM generator whose vocabulary embedding is shared with the discriminator."""

    def __init__(
        self,
        *,
        shared_embedding: nn.Embedding,
        num_layers: int = 8,
        hidden_size: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("generator hidden_size must be divisible by num_heads")
        self.shared_embedding = shared_embedding
        self.input_projection = nn.Linear(shared_embedding.embedding_dim, hidden_size, bias=False)
        self.embedding_norm = LayerNormNoBias(hidden_size)
        self.layers = nn.ModuleList(
            GeneratorBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        )
        self.final_norm = LayerNormNoBias(hidden_size)
        self.output_projection = nn.Linear(hidden_size, shared_embedding.embedding_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and not module.weight.is_meta:
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            elif isinstance(module, LayerNormNoBias) and not module.weight.is_meta:
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:
        x = F.embedding(
            input_ids,
            self.shared_embedding.weight,
            padding_idx=self.shared_embedding.padding_idx,
        )
        x = self.embedding_norm(self.input_projection(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return F.linear(self.output_projection(self.final_norm(x)), self.shared_embedding.weight)


class RTDDiscriminatorHead(nn.Module):
    """ELECTRA discriminator projection, norm, GELU, and replacement logit."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = LayerNormNoBias(hidden_size)
        self.output = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(F.gelu(self.dense(hidden)))).squeeze(-1)


@dataclass(frozen=True)
class RTDOutput:
    loss: torch.Tensor
    generator_loss: torch.Tensor
    discriminator_loss: torch.Tensor
    generator_logits: torch.Tensor
    discriminator_logits: torch.Tensor
    masked_input_ids: torch.Tensor
    corrupted_input_ids: torch.Tensor
    replaced_labels: torch.Tensor


class Orion100MRTDPretrainer(nn.Module):
    """RTD objective with discriminator gradients stopped at the shared embeddings."""

    def __init__(
        self,
        config: Orion100MConfig | None = None,
        *,
        generator_layers: int = 8,
        generator_hidden_size: int = 384,
        generator_heads: int = 6,
        generator_intermediate_size: int = 1024,
        rtd_weight: float = 50.0,
        mask_rate: float = 0.2,
        pad_token_id: int = 0,
        mask_token_id: int = 4,
    ) -> None:
        super().__init__()
        self.config = config or Orion100MConfig()
        if rtd_weight <= 0 or not 0 < mask_rate <= 1:
            raise ValueError("rtd_weight and mask_rate are out of range")
        self.discriminator = Orion100MEncoder(self.config)
        self.generator = SmallMLMGenerator(
            shared_embedding=self.discriminator.token_embeddings,
            num_layers=generator_layers,
            hidden_size=generator_hidden_size,
            num_heads=generator_heads,
            intermediate_size=generator_intermediate_size,
        )
        self.discriminator_head = RTDDiscriminatorHead(self.config.hidden_size)
        self.rtd_weight = rtd_weight
        self.mask_rate = mask_rate
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> RTDOutput:
        masked_ids, mlm_targets = whole_word_mask(
            input_ids,
            word_ids=word_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            mask_rate=self.mask_rate,
        )
        generator_logits = self.generator(masked_ids, attention_mask=attention_mask)
        generator_loss = masked_language_model_loss(generator_logits, mlm_targets)
        sample_logits = generator_logits.detach().float()
        sample_logits[..., :256] = -torch.inf
        sampled = torch.distributions.Categorical(logits=sample_logits).sample()
        selected = mlm_targets.ne(-100)
        corrupted = input_ids.clone()
        corrupted[selected] = sampled[selected]
        replaced = selected & corrupted.ne(input_ids)
        encoded = self.discriminator(
            corrupted,
            attention_mask=attention_mask,
            document_ids=document_ids,
            detach_embeddings=True,
        )
        discriminator_logits = self.discriminator_head(encoded.last_hidden_state)
        discriminator_loss = replaced_token_detection_loss(
            discriminator_logits,
            replaced,
            attention_mask=attention_mask,
        )
        total = generator_loss + self.rtd_weight * discriminator_loss
        return RTDOutput(
            loss=total,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            generator_logits=generator_logits,
            discriminator_logits=discriminator_logits,
            masked_input_ids=masked_ids,
            corrupted_input_ids=corrupted,
            replaced_labels=replaced,
        )
