"""Proposal-only organizer mapping; performs no file operations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

from .contracts import OrganizerRequest


class Classifier(Protocol):
    def classify(
        self,
        text: str,
        *,
        fields: dict[str, object] | None = None,
        registry: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> Any: ...


def to_classifier_inputs(
    request: OrganizerRequest,
) -> tuple[str, dict[str, object], dict[str, object]]:
    request.validate()
    document = request.document
    fields = {
        "filename": document.filename,
        "file_type": document.file_type,
        "title": document.title,
        "headings": list(document.headings),
        "ocr_confidence": document.ocr_confidence,
        "language": document.language_hint,
    }
    text = "\n".join([*document.headings, *document.excerpts])
    registry = {"courses": [asdict(course) for course in request.courses]}
    return text, fields, registry


def create_organization_proposal(
    request: OrganizerRequest,
    classifier: Classifier,
    *,
    review_threshold: float = 0.65,
) -> dict[str, Any]:
    """Return a reviewable proposal. The caller alone decides whether to move a file."""
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be in [0,1]")
    text, fields, registry = to_classifier_inputs(request)
    result = classifier.classify(
        text,
        fields=fields,
        registry=registry,
        options={"top_k": 3, "tasks": ["subject", "doc_type", "topic", "course_match", "unknown"]},
    )
    if isinstance(result, dict):
        payload = result
    elif callable(getattr(result, "model_dump", None)):
        payload = result.model_dump(mode="json")
    else:
        payload = result.to_dict()
    unknown_data = payload.get("unknown") or {}
    unknown = unknown_data.get("is_unknown", True)
    confidence_values = [
        item.get("confidence", 0.0)
        for key in ("subject", "doc_type", "document_type", "topic")
        for item in payload.get(key, [])[:1]
    ]
    confidence = min(confidence_values, default=0.0)
    return {
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "proposal": payload,
        "review": {
            "required": bool(unknown or confidence < review_threshold),
            "reason": "unknown_or_low_confidence"
            if unknown or confidence < review_threshold
            else None,
        },
        "file_operation": None,
    }
