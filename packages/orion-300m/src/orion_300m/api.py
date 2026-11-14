"""Thin offline FastAPI wrapper around the canonical ORION-300M SDK."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from orion_300m.sdk import Orion300M
from orion_core.errors import OrionError
from orion_core.schemas import ContextDocument


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=20000)
    documents: list[ContextDocument] = Field(default_factory=list, max_length=16)
    task: Literal[
        "generate",
        "answer",
        "explain",
        "summarize",
        "compare",
        "short_answer",
        "mcq",
        "quiz",
        "flashcards",
    ] = "answer"
    max_new_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=1000)
    seed: int | None = None
    json_mode: bool = False
    request_id: str | None = Field(default=None, max_length=256)


class GenerationResponse(BaseModel):
    """Validated result schema shared by generated and no-context responses."""

    schema_version: str
    request_id: str | None
    model: dict[str, Any]
    task: str
    status: str
    text: str
    citations: list[dict[str, Any]]
    source_documents: list[dict[str, Any]]
    warnings: list[str]
    structured: Any | None = None


class BatchGenerateRequest(BaseModel):
    """Bounded ordered generation batch using the single-request schema."""

    model_config = ConfigDict(extra="forbid")
    requests: list[GenerateRequest] = Field(min_length=1, max_length=32)


class BatchGenerationResponse(BaseModel):
    results: list[GenerationResponse]


class RequestBodySizeLimit:
    """ASGI receive wrapper that bounds both declared and chunked request bodies."""

    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        queued = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    body = (
                        b'{"error":{"code":"ORION-E1001","type":"InputLimitError",'
                        b'"message":"Request body exceeds configured size limit","details":{}}}'
                    )
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii")),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
                queued.append(message)
                if not message.get("more_body", False):
                    break
            else:
                queued.append(message)
                if message["type"] == "http.disconnect":
                    break
        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(queued):
                message = queued[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)


def create_app(sdk: Orion300M | None = None) -> FastAPI:
    """Create an API app using an injected SDK or an offline lazy local loader."""
    app = FastAPI(title="ORION-300M API", version="1.0", docs_url="/docs")
    loaded = sdk

    def get_sdk() -> Orion300M:
        nonlocal loaded
        if loaded is None:
            loaded = Orion300M.load()
        return loaded

    @app.middleware("http")
    async def optional_api_key(request: Request, call_next):
        expected = os.environ.get("ORION_API_TOKEN")
        if (
            expected
            and request.url.path not in {"/health", "/v1/health"}
            and request.headers.get("x-api-key") != expected
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "ORION-E3001",
                        "type": "AuthenticationError",
                        "message": "A valid x-api-key is required",
                        "details": {},
                    }
                },
            )
        return await call_next(request)

    @app.exception_handler(OrionError)
    async def orion_error_handler(_request: Request, exc: OrionError):
        status = (
            404
            if exc.code == "ORION-E2001"
            else 400
            if exc.code.startswith(("ORION-E1", "ORION-E4", "ORION-E0"))
            else 500
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": exc.code,
                    "type": type(exc).__name__,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        errors = [
            {
                "location": list(error.get("loc", ())),
                "type": error.get("type", "validation_error"),
                "message": error.get("msg", "Invalid request field"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "ORION-E1000",
                    "type": "RequestValidationError",
                    "message": "Request fields did not satisfy the API schema",
                    "details": {"errors": errors},
                }
            },
        )

    def health_value() -> dict[str, Any]:
        try:
            return get_sdk().health()
        except OrionError as exc:
            return {
                "status": "unavailable",
                "loaded": False,
                "error": {"code": exc.code, "message": exc.message},
            }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_value()

    @app.get("/v1/health")
    def versioned_health() -> dict[str, Any]:
        return health_value()

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        return get_sdk().model_info()

    @app.post("/v1/generate", response_model=GenerationResponse)
    def generate(payload: GenerateRequest) -> dict[str, Any]:
        if not payload.documents and payload.task != "generate":
            return {
                "schema_version": "1.0",
                "request_id": payload.request_id,
                "model": {
                    "name": "orion-300m",
                    "trained": False,
                    "status": "weights_required_for_generation",
                },
                "task": payload.task,
                "status": "no_context",
                "text": "",
                "citations": [],
                "source_documents": [],
                "warnings": ["No context documents were supplied; the model was not called."],
            }
        return (
            get_sdk()
            .generate(
                payload.prompt,
                documents=[item.model_dump() for item in payload.documents],
                task=payload.task,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                seed=payload.seed,
                json_mode=payload.json_mode,
                request_id=payload.request_id,
            )
            .to_dict()
        )

    @app.post("/v1/generate:batch", response_model=BatchGenerationResponse)
    def generate_batch(payload: BatchGenerateRequest) -> dict[str, Any]:
        requests = payload.requests
        if all(not item.documents and item.task != "generate" for item in requests):
            return {
                "results": [
                    {
                        "schema_version": "1.0",
                        "request_id": item.request_id,
                        "model": {
                            "name": "orion-300m",
                            "trained": False,
                            "status": "weights_required_for_generation",
                        },
                        "task": item.task,
                        "status": "no_context",
                        "text": "",
                        "citations": [],
                        "source_documents": [],
                        "warnings": [
                            "No context documents were supplied; the model was not called."
                        ],
                    }
                    for item in requests
                ]
            }
        return {
            "results": [
                result.to_dict()
                for result in get_sdk().generate_batch([item.model_dump() for item in requests])
            ]
        }

    @app.post("/v1/generate/stream")
    def generate_stream(payload: GenerateRequest):
        if not payload.documents and payload.task != "generate":
            events = iter(
                (
                    {
                        "event": "done",
                        "schema_version": "1.0",
                        "request_id": payload.request_id,
                        "model": {
                            "name": "orion-300m",
                            "trained": False,
                            "status": "weights_required_for_generation",
                        },
                        "task": payload.task,
                        "status": "no_context",
                        "text": "",
                        "citations": [],
                        "source_documents": [],
                        "warnings": [
                            "No context documents were supplied; the model was not called."
                        ],
                    },
                )
            )
        else:
            sdk = get_sdk()
            events = sdk.stream_generate(
                payload.prompt,
                documents=[item.model_dump() for item in payload.documents],
                task=payload.task,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                seed=payload.seed,
                json_mode=payload.json_mode,
                request_id=payload.request_id,
            )

        def encode_events():
            for item in events:
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            encode_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.add_middleware(RequestBodySizeLimit, max_bytes=3_500_000)
    return app


__all__ = ["create_app"]
