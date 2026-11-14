"""Exact SHA-256 and deterministic MinHash/LSH near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from orion_core.hashing import sha256_text

_TOKEN = re.compile(r"\w+", re.UNICODE)


def exact_duplicate_key(text: str) -> str:
    """Return an exact normalized-text key."""
    return sha256_text(text)


def shingles(text: str, *, size: int = 5) -> set[str]:
    """Return normalized word n-grams used by near-duplicate checks."""
    if size <= 0:
        raise ValueError("shingle size must be positive")
    words = [word.casefold() for word in _TOKEN.findall(text)]
    if not words:
        return set()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity with the conventional empty-set value of one."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class DuplicateMatch:
    document_id: str
    similarity: float


class MinHashLSH:
    """Fixed-seed MinHash band index followed by exact Jaccard confirmation."""

    def __init__(self, *, num_perm: int = 64, bands: int = 16, threshold: float = 0.8) -> None:
        if num_perm <= 0 or bands <= 0 or num_perm % bands:
            raise ValueError("num_perm must be a positive multiple of bands")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.num_perm = num_perm
        self.bands = bands
        self.threshold = threshold
        self._band_index: dict[tuple[int, tuple[int, ...]], set[str]] = defaultdict(set)
        self._documents: dict[str, set[str]] = {}

    def _signature(self, values: set[str]) -> tuple[int, ...]:
        if not values:
            return tuple([0] * self.num_perm)
        encoded = [value.encode("utf-8") for value in values]
        signature = []
        for index in range(self.num_perm):
            seed = index.to_bytes(4, "big")
            signature.append(
                min(
                    int.from_bytes(hashlib.blake2b(value, digest_size=8, key=seed).digest(), "big")
                    for value in encoded
                )
            )
        return tuple(signature)

    def query(self, document_id: str, text: str) -> list[DuplicateMatch]:
        """Find previously inserted documents at or above the configured threshold."""
        values = shingles(text)
        signature = self._signature(values)
        rows_per_band = self.num_perm // self.bands
        candidates: set[str] = set()
        for band in range(self.bands):
            start = band * rows_per_band
            candidates.update(self._band_index[(band, signature[start : start + rows_per_band])])
        matches = [
            DuplicateMatch(existing, jaccard(values, self._documents[existing]))
            for existing in candidates
            if jaccard(values, self._documents[existing]) >= self.threshold
        ]
        return sorted(matches, key=lambda item: (-item.similarity, item.document_id))

    def add(self, document_id: str, text: str) -> None:
        """Insert a document after its duplicate query has been handled."""
        if not document_id or document_id in self._documents:
            raise ValueError("document_id must be non-empty and unique")
        values = shingles(text)
        signature = self._signature(values)
        rows_per_band = self.num_perm // self.bands
        self._documents[document_id] = values
        for band in range(self.bands):
            start = band * rows_per_band
            self._band_index[(band, signature[start : start + rows_per_band])].add(document_id)
