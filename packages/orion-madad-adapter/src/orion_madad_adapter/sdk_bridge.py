"""Optional thin bridges from frozen adapter payloads to standalone ORION SDKs."""

from __future__ import annotations

from typing import Any, cast

from orion_core.schemas import CourseProfile as OrionCourseProfile
from orion_core.schemas import DocumentPayload


class Orion100MClassifierAdapter:
    """Translate organizer host fields into the ORION-100M SDK contract."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def classify(
        self,
        text: str,
        *,
        fields: dict[str, object] | None = None,
        registry: dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> dict[str, object]:
        fields = fields or {}
        options = options or {}
        raw_headings = fields.get("headings", [])
        if not isinstance(raw_headings, list) or any(
            not isinstance(value, str) for value in raw_headings
        ):
            raise ValueError("fields.headings must be a list of strings")
        raw_confidence = fields.get("ocr_confidence")
        if raw_confidence is not None and (
            isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float))
        ):
            raise ValueError("fields.ocr_confidence must be a number or null")
        raw_top_k = options.get("top_k", 3)
        if type(raw_top_k) is not int or raw_top_k < 1:
            raise ValueError("options.top_k must be a positive integer")
        document = DocumentPayload(
            filename=str(fields.get("filename") or "") or None,
            file_type=str(fields.get("file_type") or "") or None,
            title=str(fields.get("title") or "") or None,
            headings=list(raw_headings),
            text=text,
            ocr_confidence=float(raw_confidence) if raw_confidence is not None else None,
            language_hint=str(fields.get("language") or "") or None,
        )
        course_values = (registry or {}).get("courses", [])
        if not isinstance(course_values, list):
            raise ValueError("registry.courses must be a list")
        courses = [
            OrionCourseProfile.model_validate(item)
            for item in course_values
            if isinstance(item, dict)
        ]
        requested = options.get("tasks")
        task_aliases = {"doc_type": "document_type"}
        tasks = (
            [task_aliases.get(str(item), str(item)) for item in requested]
            if isinstance(requested, list)
            else ["subject", "document_type", "topic", "course_match", "unknown"]
        )
        result = self.model.classify(
            document,
            registry=courses,
            top_k=raw_top_k,
            tasks=tasks,
        )
        payload = result.model_dump(mode="json")
        if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
            raise TypeError("ORION classify result must serialize to a string-keyed object")
        return cast(dict[str, object], payload)


class Orion300MTutorAdapter:
    """Translate retrieval-approved tutor requests into the ORION-300M SDK contract."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def answer(
        self,
        question: str,
        *,
        documents: list[dict[str, object]],
        max_new_tokens: int = 256,
        options: dict[str, object] | None = None,
    ) -> Any:
        options = options or {}
        task = str(options.get("task", "answer"))
        if task == "questions":
            task = "quiz"
        return self.model.generate(
            question,
            documents=documents,
            task=task,
            max_new_tokens=max_new_tokens,
        )
