"""Whole-word dynamic masking and ELECTRA-style token replacement."""

from __future__ import annotations

import torch


def whole_word_mask(
    token_ids: torch.Tensor,
    word_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    special_token_ids: set[int] | None = None,
    probability: float = 0.20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select complete word groups and return masked input plus MLM labels."""
    if token_ids.shape != word_ids.shape or token_ids.shape != attention_mask.shape:
        raise ValueError("token_ids, word_ids, and attention_mask must have matching shapes")
    if not 0 < probability < 1:
        raise ValueError("mask probability must be between 0 and 1")
    special_token_ids = special_token_ids or set()
    selected = torch.zeros_like(attention_mask, dtype=torch.bool)
    for batch_index in range(token_ids.shape[0]):
        eligible_groups = torch.unique(
            word_ids[batch_index][(word_ids[batch_index] >= 0) & attention_mask[batch_index].bool()]
        )
        selected_groups = []
        for group in eligible_groups:
            positions = (word_ids[batch_index] == group) & attention_mask[batch_index].bool()
            values = token_ids[batch_index][positions]
            if any(int(value) in special_token_ids for value in values):
                continue
            if torch.rand((), generator=generator, device=token_ids.device).item() < probability:
                selected[batch_index] |= positions
                selected_groups.append(group)
        if eligible_groups.numel() and not selected_groups:
            group_index = torch.randint(
                eligible_groups.numel(), (), generator=generator, device=token_ids.device
            )
            group = eligible_groups[group_index]
            positions = (word_ids[batch_index] == group) & attention_mask[batch_index].bool()
            values = token_ids[batch_index][positions]
            if not any(int(value) in special_token_ids for value in values):
                selected[batch_index] |= positions
    labels = token_ids.clone()
    labels[~selected] = -100
    masked = token_ids.clone()
    masked[selected] = mask_token_id
    return masked, labels


def sample_replacements(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = torch.softmax(logits.float() / temperature, dim=-1)
    samples = torch.multinomial(flat.reshape(-1, flat.shape[-1]), 1, generator=generator).reshape(
        logits.shape[:-1]
    )
    return samples.to(device=logits.device)


def replace_masked_tokens(
    original_ids: torch.Tensor,
    sampled_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if original_ids.shape != sampled_ids.shape or original_ids.shape != masked_positions.shape:
        raise ValueError("token and position shapes must match")
    corrupted = original_ids.clone()
    candidate = masked_positions.bool() & attention_mask.bool()
    corrupted[candidate] = sampled_ids[candidate]
    replaced = (corrupted != original_ids) & attention_mask.bool()
    return corrupted, replaced
