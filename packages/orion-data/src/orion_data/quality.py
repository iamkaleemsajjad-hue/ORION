"""Deterministic text-quality checks and transparent language-script heuristics."""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str | None
    character_count: int
    symbol_ratio: float
    repeated_line_ratio: float


def check_text_quality(
    text: str,
    *,
    min_characters: int = 40,
    max_symbol_ratio: float = 0.45,
    max_repeated_line_ratio: float = 0.8,
) -> QualityResult:
    """Reject empty, very short, symbol-heavy, or heavily repeated-line text."""
    if (
        min_characters < 0
        or not 0 <= max_symbol_ratio <= 1
        or not 0 <= max_repeated_line_ratio <= 1
    ):
        raise ValueError("quality thresholds must be non-negative proportions")
    nonspace = [char for char in text if not char.isspace()]
    symbol_count = sum(
        1
        for char in nonspace
        if unicodedata.category(char)[0] in {"S", "P"} and char not in string.punctuation
    )
    symbol_ratio = symbol_count / max(len(nonspace), 1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = len(lines) - len(set(lines))
    repeated_ratio = repeated / max(len(lines), 1)
    reason = None
    if len(text.strip()) < min_characters:
        reason = "too_short"
    elif symbol_ratio > max_symbol_ratio:
        reason = "symbol_heavy"
    elif repeated_ratio > max_repeated_line_ratio:
        reason = "repeated_lines"
    return QualityResult(reason is None, reason, len(text), symbol_ratio, repeated_ratio)


class LanguageIdentifier(Protocol):
    """Interface for an optional licensed language-ID component."""

    def identify(self, text: str) -> str: ...


class ScriptHeuristicLanguageIdentifier:
    """Conservative en/ur script heuristic; it is not a statistical LID model."""

    def identify(self, text: str) -> str:
        latin = arabic = 0
        for char in text:
            name = unicodedata.name(char, "")
            if "LATIN" in name:
                latin += 1
            elif any(part in name for part in ("ARABIC", "PERSIAN")):
                arabic += 1
        if latin and arabic:
            return "mixed-en-ur"
        if arabic:
            return "ur-script"
        if latin:
            return "en-script"
        return "unknown"
