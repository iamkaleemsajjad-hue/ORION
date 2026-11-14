"""Structure-preserving paragraph and sentence segmentation for local text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    index: int
    kind: str
    text: str


def segment_document(text: str, *, max_characters: int = 4000) -> list[TextSegment]:
    """Split on paragraphs and sentence boundaries without crossing document IDs."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments: list[TextSegment] = []
    for paragraph in paragraphs:
        kind = "heading" if len(paragraph) <= 160 and "\n" not in paragraph else "text"
        if len(paragraph) <= max_characters:
            segments.append(TextSegment(len(segments), kind, paragraph))
            continue
        sentences = re.split(r"(?<=[.!?۔])\s+", paragraph)
        buffer = ""
        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) + 1 > max_characters:
                segments.append(TextSegment(len(segments), kind, buffer))
                buffer = ""
            if len(sentence) > max_characters:
                for offset in range(0, len(sentence), max_characters):
                    piece = sentence[offset : offset + max_characters]
                    if piece:
                        segments.append(TextSegment(len(segments), kind, piece))
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            segments.append(TextSegment(len(segments), kind, buffer))
    return segments
