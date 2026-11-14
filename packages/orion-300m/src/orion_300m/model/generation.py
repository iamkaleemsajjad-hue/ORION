"""Autoregressive greedy/sampling generation over the trained ORION decoder."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_core.blocks.attention import StaticKVCache
from orion_core.errors import InputLimitError
from orion_tokenizer.spec import DECODER_FORBIDDEN_TOKEN_IDS


class GenerationConfig(BaseModel):
    """Sampling and stopping controls with bounded generation length."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=256, gt=0, le=512)
    temperature: float = Field(default=0.2, ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    stop_token_ids: tuple[int, ...] = (6, 19)
    stop_sequences: tuple[tuple[int, ...], ...] = ()
    seed: int | None = None
    forbid_encoder_controls: bool = True

    @model_validator(mode="after")
    def validate_stops(self) -> GenerationConfig:
        if any(not sequence for sequence in self.stop_sequences):
            raise ValueError("stop sequences must not be empty")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop token IDs must be non-negative")
        return self


@torch.inference_mode()
def generate_ids(
    model: Orion300MDecoder,
    prompt_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    generation_config: GenerationConfig | None = None,
) -> list[list[int]]:
    """Generate token IDs from left-aligned equal-length prompts using a static KV cache."""
    config = generation_config or GenerationConfig()
    if config.max_new_tokens > model.config.max_generation_tokens:
        raise InputLimitError(
            f"Requested {config.max_new_tokens} new tokens; model limit is "
            f"{model.config.max_generation_tokens}"
        )
    if prompt_ids.ndim != 2 or prompt_ids.shape[-1] == 0:
        raise ValueError("prompt_ids must be a non-empty [batch, sequence] tensor")
    if prompt_ids.shape[-1] + config.max_new_tokens > model.config.context_length:
        raise InputLimitError("prompt plus requested generation exceeds the decoder context limit")
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    if attention_mask is None:
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool, device=device)
    else:
        if attention_mask.shape != prompt_ids.shape:
            raise ValueError("attention_mask must have the same shape as prompt_ids")
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    if not torch.all(attention_mask[:, -1]):
        raise ValueError("prompts must be left padded so the final input position is valid")

    was_training = model.training
    model.eval()
    try:
        cache = StaticKVCache(
            model.config.num_layers, prompt_ids.shape[0], model.config.context_length
        )
        output = model(prompt_ids, attention_mask=attention_mask, cache=cache, use_cache=True)
        generated: list[list[int]] = [[] for _ in range(prompt_ids.shape[0])]
        finished = [False] * prompt_ids.shape[0]
        history = [row.tolist() for row in prompt_ids]
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
        next_logits = output.logits[:, -1, :]
        for step_index in range(config.max_new_tokens):
            next_token = _select_next(
                next_logits,
                history,
                model.config,
                config,
                generator=generator,
            )
            for index, token in enumerate(next_token.tolist()):
                if finished[index]:
                    continue
                generated[index].append(token)
                history[index].append(token)
                if token in config.stop_token_ids or _ends_with_any(
                    generated[index], config.stop_sequences
                ):
                    finished[index] = True
            if all(finished) or step_index == config.max_new_tokens - 1:
                break
            step_ids = next_token.unsqueeze(1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=device),
                ),
                dim=1,
            )
            position = cache.sequence_length
            step_positions = torch.full(
                (step_ids.shape[0], 1), position, dtype=torch.long, device=device
            )
            output = model(
                step_ids,
                attention_mask=attention_mask,
                position_ids=step_positions,
                cache=cache,
                use_cache=True,
            )
            next_logits = output.logits[:, -1, :]
        return generated
    finally:
        model.train(was_training)


@torch.inference_mode()
def stream_ids(
    model: Orion300MDecoder,
    prompt_ids: Sequence[int],
    *,
    generation_config: GenerationConfig | None = None,
) -> Iterator[int]:
    """Yield generated IDs for one prompt while retaining the model's KV cache."""
    config = generation_config or GenerationConfig()
    if config.max_new_tokens > model.config.max_generation_tokens:
        raise InputLimitError(
            f"Requested {config.max_new_tokens} new tokens; model limit is "
            f"{model.config.max_generation_tokens}"
        )
    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    if prompt.shape[-1] == 0:
        raise ValueError("prompt_ids must not be empty")
    if prompt.shape[-1] + config.max_new_tokens > model.config.context_length:
        raise InputLimitError("prompt plus requested generation exceeds the decoder context limit")
    was_training = model.training
    model.eval()
    try:
        cache = StaticKVCache(model.config.num_layers, 1, model.config.context_length)
        output = model(prompt, cache=cache, use_cache=True)
        history = [prompt[0].tolist()]
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
        generated: list[int] = []
        logits = output.logits[:, -1, :]
        for _ in range(config.max_new_tokens):
            token = int(
                _select_next(logits, history, model.config, config, generator=generator)[0].item()
            )
            generated.append(token)
            history[0].append(token)
            yield token
            if token in config.stop_token_ids or _ends_with_any(generated, config.stop_sequences):
                break
            position = cache.sequence_length
            output = model(
                torch.tensor([[token]], dtype=torch.long, device=device),
                position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
                cache=cache,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
    finally:
        model.train(was_training)


def _select_next(
    logits: torch.Tensor,
    history: list[list[int]],
    model_config: Orion300MConfig,
    generation_config: GenerationConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = logits.float().clone()
    if generation_config.forbid_encoder_controls:
        forbidden = [
            token_id
            for token_id in DECODER_FORBIDDEN_TOKEN_IDS
            if token_id < model_config.vocab_size
        ]
        if forbidden:
            scores[:, forbidden] = -torch.inf
    for row, tokens in enumerate(history):
        if generation_config.repetition_penalty > 1.0:
            unique = set(tokens)
            if unique:
                indices = torch.tensor(sorted(unique), device=scores.device, dtype=torch.long)
                values = scores[row, indices]
                scores[row, indices] = torch.where(
                    values < 0,
                    values * generation_config.repetition_penalty,
                    values / generation_config.repetition_penalty,
                )
    if generation_config.temperature == 0:
        return scores.argmax(dim=-1)
    scores = scores / generation_config.temperature
    if generation_config.top_k:
        count = min(generation_config.top_k, scores.shape[-1])
        threshold = torch.topk(scores, count, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    if generation_config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= generation_config.top_p
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(1, sorted_indices, sorted_probs)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def _ends_with_any(tokens: list[int], sequences: tuple[tuple[int, ...], ...]) -> bool:
    return any(
        len(tokens) >= len(sequence) and tuple(tokens[-len(sequence) :]) == sequence
        for sequence in sequences
    )
