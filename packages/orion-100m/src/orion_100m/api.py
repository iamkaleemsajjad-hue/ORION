"""Thin offline FastAPI wrapper around the canonical ORION-100M SDK."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from orion_100m.sdk import (
    BatchPredictionItem,
    ClassifyResult,
    ExtractResult,
    Orion100M,
)
from orion_core.errors import OrionError
from orion_core.schemas import CourseProfile, DocumentPayload


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str | DocumentPayload
    registry: list[CourseProfile] = Field(default_factory=list, max_length=10000)
    top_k: int = Field(default=3, ge=1, le=20)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[str | DocumentPayload] = Field(min_length=1, max_length=64)
    batch_size: int = Field(default=32, ge=1, le=64)
    top_k: int = Field(default=3, ge=1, le=20)


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str | DocumentPayload
    fields: list[str] | None = None


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=64)
    dimension: int = Field(default=640, ge=1)


class EmbeddingResponse(BaseModel):
    """Validated batch embedding response."""

    model: dict[str, Any]
    dimension: int
    embeddings: list[list[float]]


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


def create_app(sdk: Orion100M | None = None) -> FastAPI:
    """Create an API app; load a local trained artifact lazily when no SDK is injected."""
    app = FastAPI(title="ORION-100M API", version="1.0", docs_url="/docs")
    loaded = sdk

    def get_sdk() -> Orion100M:
        nonlocal loaded
        if loaded is None:
            loaded = Orion100M.load()
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
            if exc.code in {"ORION-E1001", "ORION-E1003", "ORION-E4001", "ORION-E0002"}
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

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            return get_sdk().health()
        except OrionError as exc:
            return {
                "status": "unavailable",
                "loaded": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            }

    @app.get("/v1/health")
    def versioned_health() -> dict[str, Any]:
        return health()

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        return get_sdk().model_info()

    @app.post("/v1/classify", response_model=ClassifyResult)
    def classify(payload: ClassifyRequest) -> dict[str, Any]:
        return (
            get_sdk()
            .classify(payload.document, registry=payload.registry, top_k=payload.top_k)
            .model_dump(mode="json")
        )

    @app.post("/v1/predict:batch", response_model=dict[str, list[BatchPredictionItem]])
    def predict_batch(payload: BatchRequest) -> dict[str, Any]:
        values = get_sdk().predict_batch(
            payload.documents, batch_size=payload.batch_size, top_k=payload.top_k
        )
        return {"items": [value.model_dump(mode="json") for value in values]}

    @app.post("/v1/extract", response_model=ExtractResult)
    def extract(payload: ExtractRequest) -> dict[str, Any]:
        return get_sdk().extract(payload.document, fields=payload.fields).model_dump(mode="json")

    @app.post("/v1/embed", response_model=EmbeddingResponse)
    def embed(payload: EmbedRequest) -> dict[str, Any]:
        vectors = get_sdk().get_embeddings(payload.texts, dim=payload.dimension)
        return {
            "model": get_sdk().model_info(),
            "dimension": payload.dimension,
            "embeddings": vectors.tolist(),
        }

    app.add_middleware(RequestBodySizeLimit, max_bytes=2_500_000)
    return app


__all__ = ["create_app"]
