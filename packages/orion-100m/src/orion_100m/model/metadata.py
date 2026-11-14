"""Deterministic metadata candidate extraction and BIO span decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetadataSpan:
    name: str
    value: str
    start: int
    end: int
    source: str
    confidence: float


_COURSE_CODE = re.compile(r"\b[A-Z]{2,8}[- ]?\d{2,4}[A-Z]?\b")
_SEMESTER = re.compile(
    r"\b(?:spring|summer|fall|autumn|winter)\s+(?:semester\s+)?(?:19|20)\d{2}\b|"
    r"\b(?:19|20)\d{2}\s*[-/]\s*(?:spring|summer|fall|autumn|winter)\b",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)?\d{2}|(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\b"
)
_INSTRUCTOR = re.compile(r"(?im)^\s*(?:instructor|lecturer|professor|teacher)\s*[:\-]\s*(.+?)\s*$")
_TITLE_LINE = re.compile(r"(?m)^\s*(?:title|course title)\s*[:\-]\s*(.+?)\s*$")


def regex_metadata(text: str) -> list[MetadataSpan]:
    """Extract high-precision literal candidates; regex values are not model predictions."""
    spans: list[MetadataSpan] = []
    patterns = (
        ("course_code", _COURSE_CODE, 0.9),
        ("semester", _SEMESTER, 0.85),
        ("date", _DATE, 0.8),
        ("instructor", _INSTRUCTOR, 0.8),
        ("course_title", _TITLE_LINE, 0.8),
    )
    for name, pattern, confidence in patterns:
        match = pattern.search(text)
        if match is not None:
            value_group = 1 if name in {"instructor", "course_title"} else 0
            start, end = match.span(value_group)
            value = match.group(value_group).strip()
            leading = len(match.group(value_group)) - len(match.group(value_group).lstrip())
            start += leading
            spans.append(MetadataSpan(name, value, start, start + len(value), "regex", confidence))
    return spans


def decode_bio_spans(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    tag_ids: list[int],
    tag_names: list[str],
    *,
    probabilities: list[float] | None = None,
) -> list[MetadataSpan]:
    """Convert token-level BIO IDs and tokenizer offsets into exact source spans."""
    if len(offsets) != len(tag_ids):
        raise ValueError("offset and BIO tag counts differ")
    if probabilities is not None and len(probabilities) != len(tag_ids):
        raise ValueError("probability and BIO tag counts differ")
    spans: list[MetadataSpan] = []
    active_name: str | None = None
    active_start = active_end = 0
    active_scores: list[float] = []

    def flush() -> None:
        nonlocal active_name, active_start, active_end, active_scores
        if active_name is not None and 0 <= active_start < active_end <= len(text):
            value = text[active_start:active_end].strip()
            if value:
                spans.append(
                    MetadataSpan(
                        active_name,
                        value,
                        active_start,
                        active_end,
                        "model",
                        sum(active_scores) / max(len(active_scores), 1),
                    )
                )
        active_name = None
        active_scores = []

    for index, (tag_id, (start, end)) in enumerate(zip(tag_ids, offsets, strict=True)):
        if not 0 <= tag_id < len(tag_names):
            raise ValueError(f"BIO tag ID {tag_id} is out of range")
        tag = tag_names[tag_id]
        if tag == "O" or start == end:
            flush()
            continue
        if "-" not in tag or tag[0] not in {"B", "I"}:
            raise ValueError(f"Invalid BIO tag name {tag!r}")
        prefix, name = tag.split("-", maxsplit=1)
        if prefix == "B" or active_name != name:
            flush()
            active_name, active_start, active_end = name, start, end
        else:
            active_end = end
        active_scores.append(1.0 if probabilities is None else probabilities[index])
    flush()
    return spans


def prefer_deterministic_spans(
    model_spans: list[MetadataSpan], regex_spans: list[MetadataSpan]
) -> list[MetadataSpan]:
    """Keep one highest-priority candidate per field, preferring precise regex matches."""
    chosen: dict[str, MetadataSpan] = {}
    for span in regex_spans + model_spans:
        current = chosen.get(span.name)
        if current is None or (span.source == "regex" and current.source != "regex"):
            chosen[span.name] = span
    return [chosen[name] for name in sorted(chosen)]
