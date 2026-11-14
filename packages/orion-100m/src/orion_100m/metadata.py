"""Deterministic metadata candidate extraction and BIO span decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass

COURSE_CODE_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z]{2,6}[- ]?\d{2,4}[A-Z]?)(?![A-Z0-9])", re.IGNORECASE
)
DATE_RE = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
SEMESTER_RE = re.compile(r"(?i)\b(?:fall|spring|summer|winter)\s+(?:semester\s+)?(?:19|20)\d{2}\b")
INSTRUCTOR_RE = re.compile(r"(?im)^\s*(?:instructor|lecturer|teacher)\s*[:\-]\s*(.+?)\s*$")
TITLE_RE = re.compile(r"(?im)^\s*title\s*[:\-]\s*(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class MetadataField:
    name: str
    value: str
    start: int
    end: int
    source: str
    confidence: float


def regex_candidates(text: str) -> list[MetadataField]:
    result = []
    patterns = (("course_code", COURSE_CODE_RE), ("date", DATE_RE), ("semester", SEMESTER_RE))
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            result.append(
                MetadataField(
                    name,
                    match.group(1) if match.lastindex else match.group(0),
                    match.start(1) if match.lastindex else match.start(),
                    match.end(1) if match.lastindex else match.end(),
                    "regex",
                    0.9,
                )
            )
    for name, pattern in (("instructor", INSTRUCTOR_RE), ("title", TITLE_RE)):
        for match in pattern.finditer(text):
            start, end = match.span(1)
            result.append(MetadataField(name, match.group(1), start, end, "regex", 0.85))
    return result


def decode_bio_spans(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    tag_ids: list[int],
    tags: tuple[str, ...],
    confidences: list[float] | None = None,
) -> list[MetadataField]:
    if len(offsets) != len(tag_ids):
        raise ValueError("offset count must match BIO tag count")
    if confidences is not None and len(confidences) != len(tag_ids):
        raise ValueError("confidence count must match BIO tag count")
    fields: list[MetadataField] = []
    active_type = None
    start = end = 0
    scores: list[float] = []

    def flush() -> None:
        nonlocal active_type, start, end, scores
        if active_type is not None and end > start:
            value = text[start:end]
            if value.strip():
                fields.append(
                    MetadataField(
                        active_type,
                        value,
                        start,
                        end,
                        "model",
                        sum(scores) / len(scores) if scores else 0.0,
                    )
                )
        active_type, scores = None, []

    for index, (tag_id, span) in enumerate(zip(tag_ids, offsets, strict=True)):
        if tag_id < 0 or tag_id >= len(tags):
            flush()
            continue
        tag = tags[tag_id]
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, name = tag.split("-", 1)
        token_start, token_end = span
        if token_end <= token_start:
            continue
        if prefix == "B" or active_type != name:
            flush()
            active_type, start, end = name, token_start, token_end
            scores = [confidences[index]] if confidences else []
        elif prefix == "I":
            end = token_end
            if confidences:
                scores.append(confidences[index])
        else:
            flush()
    flush()
    return fields


def merge_metadata_candidates(
    regex_fields: list[MetadataField],
    model_fields: list[MetadataField],
    *,
    model_threshold: float = 0.65,
) -> list[MetadataField]:
    merged = list(regex_fields)
    for candidate in model_fields:
        if candidate.confidence < model_threshold:
            continue
        duplicate = any(
            item.name == candidate.name
            and not (item.end <= candidate.start or item.start >= candidate.end)
            for item in merged
        )
        if not duplicate:
            merged.append(candidate)
    return sorted(merged, key=lambda item: (item.start, item.end, item.name))
