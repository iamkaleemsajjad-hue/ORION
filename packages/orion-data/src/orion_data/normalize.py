"""Conservative Unicode normalization and PII-aware text sanitation."""

from __future__ import annotations

import re
import unicodedata

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d() .-]{7,}\d)(?!\w)")
_PAKISTAN_ID = re.compile(r"(?<!\d)\d{5}-\d{7}-\d(?!\d)")
_URL_QUERY = re.compile(r"\bhttps?://[^\s?#]+\?[^\s#]+", re.IGNORECASE)


def normalize_text(text: str, *, fold_urdu_variants: bool = False) -> str:
    """Apply NFC and remove controls while preserving Urdu ZWNJ/ZWJ and line breaks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text)
    cleaned = []
    for char in normalized:
        category = unicodedata.category(char)
        if category == "Cc" and char not in {"\n", "\r", "\t"}:
            continue
        if category == "Cf" and char not in {"\u200c", "\u200d"}:
            continue
        cleaned.append(char)
    result = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    if fold_urdu_variants:
        result = result.translate(str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی"}))
    return result


def mask_pii(text: str, *, replacement: str = "[REDACTED]") -> tuple[str, dict[str, int]]:
    """Mask common email, phone, Pakistan ID, and credential-bearing URLs."""
    counts: dict[str, int] = {}
    result = text
    for name, pattern in (
        ("email", _EMAIL),
        ("phone", _PHONE),
        ("national_id", _PAKISTAN_ID),
        ("url_query", _URL_QUERY),
    ):
        result, count = pattern.subn(replacement, result)
        counts[name] = count
    return result, counts
