"""Typed, dependency-free representations of the frozen host adapter payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orion_core.version import SCHEMA_VERSION


@dataclass(slots=True)
class CourseProfile:
    course_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    profile_text: str = ""


@dataclass(slots=True)
class OrganizerDocument:
    filename: str
    file_type: str
    title: str = ""
    headings: list[str] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    language_hint: str | None = None

    def validate(self) -> None:
        if not self.filename and not self.title and not self.excerpts:
            raise ValueError("document requires filename, title, or excerpts")
        if self.ocr_confidence is not None and not 0 <= self.ocr_confidence <= 1:
            raise ValueError("ocr_confidence must be in [0,1]")
        if len(self.headings) > 128 or len(self.excerpts) > 256:
            raise ValueError("document has too many headings or excerpts")


@dataclass(slots=True)
class OrganizerRequest:
    document: OrganizerDocument
    courses: list[CourseProfile] = field(default_factory=list)
    request_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version.split(".", 1)[0] != "1":
            raise ValueError(f"Unsupported schema version {self.schema_version!r}")
        self.document.validate()
        if len(self.courses) > 512:
            raise ValueError("registry contains more than 512 courses")
        ids = [course.course_id for course in self.courses]
        if len(ids) != len(set(ids)):
            raise ValueError("registry course_id values must be unique")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OrganizerRequest:
        doc = OrganizerDocument(**value.get("document", {}))
        registry = value.get("registry", {})
        courses = [CourseProfile(**course) for course in registry.get("courses", [])]
        request = cls(
            document=doc,
            courses=courses,
            request_id=value.get("request_id"),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )
        request.validate()
        return request


@dataclass(slots=True)
class RetrievedDocument:
    id: str
    label: str
    text: str

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("retrieved document id must be nonempty")
        if not self.text.strip():
            raise ValueError(f"retrieved document {self.id!r} has empty text")


@dataclass(slots=True)
class TutorRequest:
    question: str
    documents: list[RetrievedDocument]
    task: str = "answer"
    max_new_tokens: int = 256
    request_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version.split(".", 1)[0] != "1":
            raise ValueError(f"Unsupported schema version {self.schema_version!r}")
        if self.task not in {
            "answer",
            "summarize",
            "explain",
            "compare",
            "questions",
            "flashcards",
            "mcq",
        }:
            raise ValueError(f"Unsupported tutor task {self.task!r}")
        if not self.question.strip():
            raise ValueError("question must be nonempty")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be in [1,512]")
        if len(self.documents) > 16:
            raise ValueError("at most 16 retrieved documents are allowed")
        ids = [document.id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved document IDs must be unique")
        for document in self.documents:
            document.validate()
