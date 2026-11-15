from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from orion_100m.sdk import ClassifyResult, LabelScore
from orion_core.errors import NoContextError
from orion_core.schemas import CourseProfile, DocumentPayload
from orion_madad_adapter import (
    OrganizerRequest,
    Orion100MClassifierAdapter,
    Orion300MTutorAdapter,
    RetrievedDocument,
    TutorRequest,
    answer_from_retrieval,
    create_organization_proposal,
)


class _ClassifierStub:
    def __init__(self) -> None:
        self.payload: DocumentPayload | None = None
        self.options: dict[str, object] | None = None

    def classify(
        self,
        document: DocumentPayload,
        *,
        registry: Sequence[CourseProfile] | None = None,
        top_k: int = 3,
        tasks: Sequence[str] = (),
    ) -> ClassifyResult:
        self.payload = document
        self.options = {"registry": registry, "top_k": top_k, "tasks": tasks}
        return ClassifyResult(
            model={"name": "orion-100m", "trained": True},
            subject=[LabelScore(label="math", confidence=0.91)],
            document_type=[LabelScore(label="notes", confidence=0.84)],
            topic=[LabelScore(label="algebra", confidence=0.8)],
            tags=[],
            unknown={"is_unknown": False, "score": 0.09},
            confidence=0.91,
            confidence_method="calibrated",
            calibrated=True,
        )


class _TutorStub:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response: dict[str, object] = dict(response)
        self.calls = 0
        self.documents: list[dict[str, object]] | None = None
        self.options: dict[str, object] | None = None

    def generate(
        self,
        question: str,
        *,
        documents: list[dict[str, object]],
        task: str,
        max_new_tokens: int,
    ) -> dict[str, object]:
        self.calls += 1
        self.documents = documents
        self.options = {"task": task, "max_new_tokens": max_new_tokens}
        return self.response


def test_organizer_contract_maps_synthetic_payload_without_file_operations() -> None:
    request = OrganizerRequest.from_dict(
        {
            "schema_version": "1.0",
            "request_id": "fixture-1",
            "document": {
                "filename": "lecture.pdf",
                "file_type": "pdf",
                "title": "Algebra",
                "headings": ["Linear equations"],
                "excerpts": ["A synthetic teaching example."],
                "ocr_confidence": 0.98,
                "language_hint": "en",
            },
            "registry": {
                "courses": [{"course_id": "MATH-101", "name": "Algebra", "aliases": ["Math"]}]
            },
        }
    )
    classifier = _ClassifierStub()
    proposal = create_organization_proposal(
        request, Orion100MClassifierAdapter(classifier), review_threshold=0.7
    )
    assert proposal["request_id"] == "fixture-1"
    assert proposal["review"]["required"] is False
    assert proposal["file_operation"] is None
    assert classifier.payload is not None
    assert classifier.payload.text == "Linear equations\nA synthetic teaching example."
    assert classifier.options is not None
    registry = classifier.options["registry"]
    tasks = classifier.options["tasks"]
    assert isinstance(registry, list) and registry
    assert isinstance(registry[0], CourseProfile)
    assert registry[0].course_id == "MATH-101"
    assert isinstance(tasks, list)
    assert "document_type" in tasks


def test_tutor_adapter_requires_retrieval_and_maps_only_supplied_sources() -> None:
    request = TutorRequest(
        question="How does reinforcement learning train an agent?",
        documents=[
            RetrievedDocument(
                id="chunk-1",
                label="Lecture 2, page 4",
                text="Reinforcement learning trains an agent through rewards.",
            )
        ],
        task="answer",
        max_new_tokens=48,
        request_id="tutor-1",
    )
    response: dict[str, object] = {
        "text": "Reinforcement learning trains an agent through rewards [1].",
        "status": "generated",
    }
    sdk = _TutorStub(response)
    tutor = Orion300MTutorAdapter(sdk)
    result = answer_from_retrieval(
        request,
        tutor,
        count_tokens=lambda text: max(1, len(text.split())),
        context_limit=1024,
        template_reserve=16,
        support_threshold=0.2,
    )
    assert sdk.calls == 1
    assert sdk.options == {"task": "answer", "max_new_tokens": 48}
    assert result["sources"] == [{"id": "chunk-1", "label": "Lecture 2, page 4"}]
    assert result["citations"][0]["valid"] is True


def test_tutor_adapter_abstains_before_call_when_retrieval_is_missing() -> None:
    sdk = _TutorStub({"text": "", "status": "generated"})
    request = TutorRequest(question="question", documents=[])
    with pytest.raises(NoContextError):
        answer_from_retrieval(
            request,
            Orion300MTutorAdapter(sdk),
            count_tokens=lambda text: len(text.split()),
            context_limit=1024,
        )
    assert sdk.calls == 0
