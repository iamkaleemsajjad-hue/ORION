"""Citation marker mapping and conservative lexical support checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .contracts import RetrievedDocument

CITATION_RE = re.compile(r"<\|cite_(\d+)\|>|\[(\d+)\]")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?(?:\n|$)")
WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "what",
    "when",
    "which",
}


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    marker: int
    document_id: str
    valid: bool
    support_ratio: float


@dataclass(frozen=True, slots=True)
class CitationResult:
    text: str
    citations: tuple[VerifiedCitation, ...]
    warnings: tuple[str, ...]
    unsupported_sentences: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "citations": [asdict(item) for item in self.citations],
            "warnings": list(self.warnings),
            "unsupported_sentences": list(self.unsupported_sentences),
        }


def _content_words(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in WORD_RE.finditer(text)
        if match.group(0).casefold() not in STOPWORDS and len(match.group(0)) > 2
    }


def verify_citations(
    text: str,
    documents: list[RetrievedDocument],
    *,
    min_overlap: float = 0.2,
) -> CitationResult:
    """Map citations by prompt index, remove invalid IDs, and flag weak support.

    The lexical ratio is a deterministic screening heuristic, not an entailment
    guarantee. Host applications must show sources from their own retrieved set.
    """
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must be in [0,1]")
    by_index = {index: document for index, document in enumerate(documents, 1)}
    citations: list[VerifiedCitation] = []
    warnings: list[str] = []
    unsupported: list[str] = []
    cleaned = text
    for match in list(CITATION_RE.finditer(text)):
        index = int(match.group(1) or match.group(2))
        document = by_index.get(index)
        if document is None:
            cleaned = cleaned.replace(match.group(0), "")
            citations.append(VerifiedCitation(index, "", False, 0.0))
            warnings.append(f"removed_unmapped_citation:{index}")
            continue
        containing_sentence = next(
            (
                s.group(0)
                for s in SENTENCE_RE.finditer(text)
                if match.start() >= s.start() and match.end() <= s.end()
            ),
            text,
        )
        answer_words = _content_words(CITATION_RE.sub("", containing_sentence))
        source_words = _content_words(document.text)
        ratio = len(answer_words & source_words) / max(1, len(answer_words))
        valid = ratio >= min_overlap if answer_words else False
        citations.append(VerifiedCitation(index, document.id, valid, ratio))
        if not valid:
            unsupported.append(containing_sentence.strip())
            warnings.append(f"weak_lexical_support:{index}")
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return CitationResult(
        cleaned, tuple(citations), tuple(warnings), tuple(dict.fromkeys(unsupported))
    )
