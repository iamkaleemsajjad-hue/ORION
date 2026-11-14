"""Public prompt and generation helpers backed by the canonical decoder engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.model.generation import GenerationConfig, generate_ids
from orion_core.errors import InputLimitError
from orion_tokenizer.spec import SPECIAL_TOKEN_IDS


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated IDs and the actual stopping condition observed by the engine."""

    token_ids: tuple[int, ...]
    stop_reason: str


def build_grounded_prompt(
    tokenizer: Any,
    *,
    system: str,
    question: str,
    documents: list[dict[str, str]],
    structured: bool = False,
) -> list[int]:
    """Build a control-safe prompt with explicit context and source markers."""
    if not question.strip() or not system.strip():
        raise ValueError("system and question text must be non-empty")
    if len(documents) > 16:
        raise ValueError("At most 16 documents can receive citation markers")
    ids = [SPECIAL_TOKEN_IDS["<|bos|>"], SPECIAL_TOKEN_IDS["<|system|>"]]
    ids.extend(tokenizer.encode(system).input_ids)
    ids.append(SPECIAL_TOKEN_IDS["<|end|>"])
    if documents:
        ids.append(SPECIAL_TOKEN_IDS["<|context|>"])
        for index, document in enumerate(documents, start=1):
            label = document.get("label", document.get("title", ""))
            text = document.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"document {index} must contain non-empty text")
            ids.append(SPECIAL_TOKEN_IDS[f"<|doc_{index}|>"])
            ids.extend(tokenizer.encode(label).input_ids)
            ids.append(SPECIAL_TOKEN_IDS["<field_sep>"])
            ids.extend(tokenizer.encode(text).input_ids)
        ids.append(SPECIAL_TOKEN_IDS["<|/context|>"])
    ids.append(SPECIAL_TOKEN_IDS["<|user|>"])
    ids.extend(tokenizer.encode(question).input_ids)
    ids.append(SPECIAL_TOKEN_IDS["<|end|>"])
    ids.append(SPECIAL_TOKEN_IDS["<|assistant|>"])
    if structured:
        ids.append(SPECIAL_TOKEN_IDS["<|json|>"])
    return ids


@torch.inference_mode()
def generate_tokens(
    model: Orion300MDecoder,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 0,
    stop_sequences: tuple[tuple[int, ...], ...] = (),
    seed: int | None = None,
) -> GenerationResult:
    """Run the same GQA/KV-cache generation path used by the public SDK."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("generate_tokens expects a nonempty [1, sequence] prompt")
    if max_new_tokens > model.config.max_generation_tokens:
        raise InputLimitError("Requested output exceeds the model generation limit")
    values = generate_ids(
        model,
        input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stop_token_ids=(
                model.config.eos_token_id,
                model.config.abstain_token_id,
                model.config.end_token_id,
            ),
            seed=seed,
        ),
    )[0]
    if values and values[-1] == model.config.eos_token_id:
        reason = "eos"
    elif values and values[-1] == model.config.abstain_token_id:
        reason = "abstain"
    elif values and values[-1] == model.config.end_token_id:
        reason = "end"
    else:
        reason = "max_new_tokens_or_stop_sequence"
    return GenerationResult(tuple(values), reason)


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "Orion300MConfig",
    "build_grounded_prompt",
    "generate_tokens",
]
