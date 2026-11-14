"""Shared typed request and response contracts for SDKs, adapters, and REST."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base schema that rejects unrecognized fields to avoid silent contract drift."""

    model_config = ConfigDict(extra="forbid")


class CourseProfile(StrictModel):
    course_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=512)
    aliases: list[str] = Field(default_factory=list, max_length=64)
    profile_text: str = Field(default="", max_length=20000)


class DocumentPayload(StrictModel):
    """Host-extracted document text; ORION SDKs never open or parse local files."""

    document_id: str | None = Field(default=None, max_length=256)
    filename: str | None = Field(default=None, max_length=1024)
    file_type: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, max_length=2048)
    headings: list[str] = Field(default_factory=list, max_length=256)
    excerpts: list[str] = Field(default_factory=list, max_length=64)
    text: str = Field(default="", max_length=2_000_000)
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    language_hint: str | None = Field(default=None, max_length=64)

    @field_validator("text")
    @classmethod
    def reject_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("document text must not contain NUL bytes")
        return value


class ContextDocument(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=2048)
    text: str = Field(min_length=1, max_length=200000)
    page: int | None = Field(default=None, ge=1)
    slide: int | None = Field(default=None, ge=1)


class ModelVersionInfo(StrictModel):
    name: str
    version: str
    tokenizer_version: str
    weights_sha256: str | None = None
    artifact_status: str


class CitationReference(StrictModel):
    marker: int = Field(ge=1, le=16)
    document_id: str
    label: str
    mapped: bool
    support_verified: bool = False


class TaskResult(StrictModel):
    status: str
    text: str = ""
    citations: list[CitationReference] = Field(default_factory=list)
    abstained: bool = False
    citation_verification: str = "unverified"
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class BatchItemError(StrictModel):
    code: str
    type: str
    message: str


class APIErrorBody(StrictModel):
    code: str
    type: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class APIErrorResponse(StrictModel):
    error: APIErrorBody
