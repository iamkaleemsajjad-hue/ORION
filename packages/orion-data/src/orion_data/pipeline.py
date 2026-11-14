"""Deterministic, local text-cleaning and quality-filtering primitives."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{5}-\d{7}-\d(?!\d)")
_OCR_CONFUSIONS = (("rn", "m"), ("cl", "d"), ("O", "0"), ("l", "1"), ("ﬁ", "fi"))


@dataclass(frozen=True, slots=True)
class QualityResult:
    accepted: bool
    reason: str | None
    char_count: int
    alnum_ratio: float
    repeated_line_ratio: float


def normalize_text(text: str, *, fold_urdu_variants: bool = False) -> str:
    """NFC normalization, retaining ZWNJ and ZWJ which matter in Urdu."""
    text = unicodedata.normalize("NFC", text)
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in {"\u200c", "\u200d"} or char in "\n\r\t":
            cleaned.append(char)
        elif category == "Cc" or category == "Cf":
            continue
        else:
            cleaned.append(char)
    text = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    if fold_urdu_variants:
        text = text.translate(str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی"}))
    return text


def mask_pii(text: str, replacement: str = "[PII]") -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern in (
        ("email", EMAIL_RE),
        ("url", URL_RE),
        ("national_id", NATIONAL_ID_RE),
        ("phone", PHONE_RE),
    ):
        text, count = pattern.subn(replacement, text)
        counts[name] = count
    return text, counts


def quality_check(
    text: str,
    *,
    min_chars: int = 80,
    max_symbol_ratio: float = 0.45,
    max_repeat_ratio: float = 0.35,
) -> QualityResult:
    if len(text) < min_chars:
        return QualityResult(False, "too_short", len(text), 0.0, 0.0)
    visible = [char for char in text if not char.isspace()]
    alnum_ratio = sum(char.isalnum() for char in visible) / max(1, len(visible))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = len(lines) - len(set(lines))
    repeat_ratio = repeated / max(1, len(lines))
    if alnum_ratio < 1.0 - max_symbol_ratio:
        return QualityResult(False, "symbol_heavy", len(text), alnum_ratio, repeat_ratio)
    if repeat_ratio > max_repeat_ratio:
        return QualityResult(False, "repeated_lines", len(text), alnum_ratio, repeat_ratio)
    if len(set(text)) <= 3:
        return QualityResult(False, "low_character_diversity", len(text), alnum_ratio, repeat_ratio)
    return QualityResult(True, None, len(text), alnum_ratio, repeat_ratio)


def exact_dedup_key(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_split(
    document_id: str,
    *,
    seed: int = 0,
    proportions: tuple[float, float, float] = (0.998, 0.001, 0.001),
) -> str:
    if len(proportions) != 3 or any(p < 0 for p in proportions) or abs(sum(proportions) - 1) > 1e-9:
        raise ValueError("proportions must be three nonnegative values summing to one")
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < proportions[0]:
        return "train"
    if value < proportions[0] + proportions[1]:
        return "validation"
    return "test"


def synthesize_ocr_noise(text: str, *, cer: float, seed: int) -> str:
    """Apply seeded character confusions/deletions, for augmentation only."""
    if not 0 <= cer <= 1:
        raise ValueError("cer must lie in [0, 1]")
    import random

    rng = random.Random(seed)
    output: list[str] = []
    for char in text:
        if rng.random() < cer:
            choice = rng.randrange(4)
            if choice == 0:
                output.append("")
            elif choice == 1:
                output.append(char + "\n")
            else:
                replacement = next((b for a, b in _OCR_CONFUSIONS if char in a), None)
                output.append(replacement if replacement else char)
        else:
            output.append(char)
    return "".join(output)


def process_record(
    record: dict, *, fold_urdu_variants: bool = False, mask_personal_data: bool = True
) -> tuple[dict, dict]:
    if not isinstance(record.get("text"), str):
        raise ValueError("record.text must be a string")
    text = normalize_text(record["text"], fold_urdu_variants=fold_urdu_variants)
    pii_counts = {}
    if mask_personal_data:
        text, pii_counts = mask_pii(text)
    result = dict(record)
    result["text"] = text
    result["text_sha256"] = exact_dedup_key(text)
    result["language"] = record.get("language") or detect_language_hint(text)
    quality = quality_check(text)
    result["quality"] = quality.accepted
    return result, {"quality": quality, "pii_counts": pii_counts}


def detect_language_hint(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unknown"
    arabic = sum("ARABIC" in unicodedata.name(char, "") for char in letters)
    latin = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    if arabic and latin:
        return "mixed-en-ur"
    return "ur" if arabic else "en" if latin else "other"


def deduplicate_records(records: Iterable[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    output = []
    duplicates = 0
    for record in records:
        key = exact_dedup_key(record["text"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append({**record, "text_sha256": key})
    return output, duplicates
