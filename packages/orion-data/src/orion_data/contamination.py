"""Exact n-gram contamination checks against explicitly supplied evaluation text."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)


def word_ngrams(text: str, size: int = 13) -> set[tuple[str, ...]]:
    """Lowercase Unicode word n-grams with punctuation removed."""
    if size <= 0:
        raise ValueError("ngram size must be positive")
    words = [word.casefold() for word in _WORD.findall(text)]
    if len(words) < size:
        return set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


@dataclass(frozen=True)
class ContaminationHit:
    evaluation_id: str
    shared_ngrams: int


class ContaminationIndex:
    """Memory-resident index built from sealed evaluation records, not raw test files."""

    def __init__(self, evaluation_texts: dict[str, str], *, ngram_size: int = 13) -> None:
        if not evaluation_texts:
            raise ValueError("evaluation_texts must not be empty")
        self.ngram_size = ngram_size
        self._index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for evaluation_id, text in evaluation_texts.items():
            for gram in word_ngrams(text, ngram_size):
                self._index[gram].add(evaluation_id)

    def check(self, text: str) -> list[ContaminationHit]:
        counts: dict[str, int] = defaultdict(int)
        for gram in word_ngrams(text, self.ngram_size):
            for evaluation_id in self._index.get(gram, ()):
                counts[evaluation_id] += 1
        return [ContaminationHit(key, value) for key, value in sorted(counts.items())]
