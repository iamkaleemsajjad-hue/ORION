"""Token-budget packing of retrieved snippets while preserving their IDs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .contracts import RetrievedDocument


@dataclass(frozen=True, slots=True)
class PackedContext:
    documents: tuple[RetrievedDocument, ...]
    used_tokens: int
    available_tokens: int
    dropped_ids: tuple[str, ...]


def pack_documents(
    documents: Sequence[RetrievedDocument],
    *,
    count_tokens: Callable[[str], int],
    question: str,
    max_new_tokens: int,
    context_limit: int = 4096,
    template_reserve: int = 192,
    safety_reserve: int = 256,
) -> PackedContext:
    """Keep retrieval order, clip only the last admitted passage, and retain IDs."""
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be in [1,512]")
    fixed = count_tokens(question) + max_new_tokens + template_reserve + safety_reserve
    available = context_limit - fixed
    if available <= 0:
        raise ValueError("Question and answer reserve leave no room for retrieved evidence")
    packed: list[RetrievedDocument] = []
    used = 0
    for document in documents:
        label_cost = count_tokens(document.label)
        text_cost = count_tokens(document.text)
        cost = label_cost + text_cost + 8
        if used + cost <= available:
            packed.append(document)
            used += cost
            continue
        remainder = available - used - label_cost - 8
        if remainder > 0:
            # Token IDs are opaque here; binary search the longest text prefix.
            lo, hi, best = 0, len(document.text), ""
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = document.text[:mid]
                if count_tokens(candidate) <= remainder:
                    best, lo = candidate, mid + 1
                else:
                    hi = mid - 1
            if best.strip():
                packed.append(RetrievedDocument(document.id, document.label, best))
        break
    packed_ids = {document.id for document in packed}
    dropped = [document.id for document in documents if document.id not in packed_ids]
    used = sum(
        count_tokens(document.label) + count_tokens(document.text) + 8 for document in packed
    )
    if used > available:
        raise AssertionError("Packed context exceeded its token budget")
    return PackedContext(tuple(packed), used, available, tuple(dropped))
