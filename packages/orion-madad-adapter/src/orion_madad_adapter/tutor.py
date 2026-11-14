"""Retrieval-gated tutor calls and citation filtering for host integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Protocol

from orion_core.errors import NoContextError

from .citations import verify_citations
from .contracts import TutorRequest
from .packing import pack_documents


class Tutor(Protocol):
    def answer(
        self,
        question: str,
        *,
        documents: list[dict[str, object]],
        max_new_tokens: int = 256,
        options: dict[str, object] | None = None,
    ) -> Any: ...


def answer_from_retrieval(
    request: TutorRequest,
    tutor: Tutor,
    *,
    count_tokens: Callable[[str], int],
    context_limit: int = 4096,
    template_reserve: int = 192,
    support_threshold: float = 0.2,
) -> dict[str, Any]:
    """Do not call the generator without retrieval evidence."""
    request.validate()
    if not request.documents:
        raise NoContextError("No retrieved material was supplied; generation was not called")
    packed = pack_documents(
        request.documents,
        count_tokens=count_tokens,
        question=request.question,
        max_new_tokens=request.max_new_tokens,
        context_limit=context_limit,
        template_reserve=template_reserve,
    )
    if not packed.documents:
        raise NoContextError("No retrieved material fits the context budget")
    model_docs = [asdict(document) for document in packed.documents]
    raw = tutor.answer(
        request.question,
        documents=model_docs,
        max_new_tokens=request.max_new_tokens,
        options={"task": request.task, "require_grounding": True},
    )
    payload = raw if isinstance(raw, dict) else raw.to_dict()
    output = payload.get("output", payload)
    text = str(output.get("text", ""))
    abstained = bool(output.get("abstained", False) or output.get("status") == "abstained")
    citation_result = verify_citations(text, list(packed.documents), min_overlap=support_threshold)
    return {
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "status": "abstained" if abstained else "answered",
        "text": citation_result.text,
        "abstained": abstained,
        "citations": [asdict(item) for item in citation_result.citations],
        "warnings": list(citation_result.warnings),
        "unsupported_sentences": list(citation_result.unsupported_sentences),
        "sources": [{"id": document.id, "label": document.label} for document in packed.documents],
        "dropped_document_ids": list(packed.dropped_ids),
    }
