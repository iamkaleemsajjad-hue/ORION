"""Standalone ORION-100M document classification, extraction, and embedding SDK."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError

from orion_100m.config import Orion100MConfig
from orion_100m.model.metadata import (
    decode_bio_spans,
    prefer_deterministic_spans,
    regex_metadata,
)
from orion_100m.model.tasks import Orion100MForTasks
from orion_core.device import select_device
from orion_core.errors import (
    ArtifactIntegrityError,
    EmptyInputError,
    InputLimitError,
    ModelWeightsNotFound,
    OrionError,
)
from orion_core.hashing import sha256_file
from orion_core.schemas import CourseProfile, DocumentPayload
from orion_core.serialization import load_model_state
from orion_data.normalize import normalize_text
from orion_tokenizer import OrionTokenizer


class LabelScore(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class CourseMatchResult(BaseModel):
    course_id: str
    name: str
    similarity: float
    confidence: float = Field(ge=0, le=1)
    is_unknown: bool


class ClassifyResult(BaseModel):
    model: dict[str, Any]
    subject: list[LabelScore]
    document_type: list[LabelScore]
    topic: list[LabelScore]
    tags: list[LabelScore]
    course_match: CourseMatchResult | None = None
    unknown: dict[str, float | bool] | None = None
    confidence: float
    confidence_method: str
    calibrated: bool
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    name: str
    value: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    source: str
    confidence: float = Field(ge=0, le=1)


class ExtractResult(BaseModel):
    model: dict[str, Any]
    fields: list[ExtractedField]
    warnings: list[str] = Field(default_factory=list)


class BatchPredictionItem(BaseModel):
    result: ClassifyResult | None = None
    error: dict[str, str] | None = None


@dataclass(frozen=True)
class _EncodedDocument:
    input_ids: tuple[int, ...]
    text: str


def _validate_calibration_metadata(
    thresholds: object,
    temperatures: object,
    labels: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require tuned thresholds and temperatures before accepting a trained release."""
    if not isinstance(thresholds, dict) or not isinstance(temperatures, dict):
        raise ArtifactIntegrityError("trained releases require threshold and temperature objects")
    for name in ("subject", "document_type", "topic", "confidence"):
        value = temperatures.get(name)
        if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            raise ArtifactIntegrityError(f"missing or invalid {name} calibration temperature")
    unknown_threshold = thresholds.get("unknown_max_known_probability")
    course_threshold = thresholds.get("course_min_similarity")
    if (
        not isinstance(unknown_threshold, int | float)
        or not math.isfinite(unknown_threshold)
        or not 0 <= unknown_threshold <= 1
    ):
        raise ArtifactIntegrityError(
            "unknown detection requires a calibrated probability threshold"
        )
    if not isinstance(course_threshold, int | float) or not math.isfinite(course_threshold):
        raise ArtifactIntegrityError("course matching requires a calibrated similarity threshold")
    tag_thresholds = thresholds.get("tags")
    if not isinstance(tag_thresholds, dict) or any(
        label not in tag_thresholds
        or not isinstance(tag_thresholds[label], int | float)
        or not math.isfinite(tag_thresholds[label])
        or not 0 <= tag_thresholds[label] <= 1
        for label in labels["tags"]
    ):
        raise ArtifactIntegrityError("each configured tag needs a calibrated threshold")
    return thresholds, temperatures


def _validated_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ArtifactIntegrityError(f"A trained ORION-100M release has an invalid {field}")
    return value.lower()


class Orion100M:
    """Loaded trained local encoder with explicit label maps and calibration metadata."""

    def __init__(
        self,
        model: Orion100MForTasks,
        tokenizer: OrionTokenizer,
        *,
        device: torch.device,
        artifact_config: dict[str, Any],
        weights_sha256: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.artifact_config = artifact_config
        self.weights_sha256 = weights_sha256
        self.labels = artifact_config["label_maps"]
        self.thresholds = artifact_config.get("thresholds", {})
        self.temperatures = artifact_config.get("temperatures", {})

    @classmethod
    def load(
        cls,
        *,
        device: str = "auto",
        weights_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
    ) -> Orion100M:
        """Load a verified trained local release without network access or random fallback."""
        configured = weights_dir or os.environ.get("ORION_100M_WEIGHTS")
        root = Path(configured) if configured else Path(__file__).resolve().parents[2] / "weights"
        weights_path = root / "model.safetensors"
        if not weights_path.is_file():
            raise ModelWeightsNotFound(
                f"ORION-100M trained weights were not found at {weights_path}. "
                "Architecture code is available, but a trained model release has not been supplied."
            )
        config_path = root / "config.json"
        if not config_path.is_file():
            raise ModelWeightsNotFound(f"Trained model config was not found at {config_path}")
        try:
            artifact = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"Could not read model config at {config_path}") from exc
        if not isinstance(artifact, dict):
            raise ArtifactIntegrityError("trained model config must be a JSON object")
        if (
            artifact.get("artifact_status") != "trained"
            or artifact.get("training_completed") is not True
        ):
            raise ModelWeightsNotFound(
                "The supplied artifact is not marked as a completed trained model"
            )
        for field in (
            "training_data_manifest_sha256",
            "training_report",
            "evaluation_report",
        ):
            if not artifact.get(field):
                raise ArtifactIntegrityError(f"A trained ORION-100M release is missing {field}")
        _validated_sha256(
            artifact.get("training_data_manifest_sha256"), "training_data_manifest_sha256"
        )
        weights_digest = _validated_sha256(artifact.get("weights_sha256"), "weights_sha256")
        tokenizer_digest = _validated_sha256(artifact.get("tokenizer_sha256"), "tokenizer_sha256")
        if sha256_file(weights_path) != weights_digest:
            raise ArtifactIntegrityError("model.safetensors SHA-256 does not match config.json")
        config_data = artifact.get("model_config", {})
        if not isinstance(config_data, dict):
            raise ArtifactIntegrityError("model_config must be a JSON object")
        try:
            config = Orion100MConfig.model_validate(config_data)
        except ValidationError as exc:
            raise ArtifactIntegrityError(f"Invalid ORION-100M model config: {exc}") from exc
        if config.model_name != "orion-100m":
            raise ArtifactIntegrityError("Model config does not identify ORION-100M")
        label_maps = artifact.get("label_maps")
        if not isinstance(label_maps, dict):
            raise ArtifactIntegrityError("trained model config must include label_maps")
        required = ("subject", "document_type", "topic", "tags", "metadata_tags")
        if any(
            not isinstance(label_maps.get(key), list) or not label_maps[key] for key in required
        ):
            raise ArtifactIntegrityError(f"label_maps must include non-empty arrays: {required}")
        _validate_calibration_metadata(
            artifact.get("thresholds"), artifact.get("temperatures"), label_maps
        )
        tokenizer_root = Path(tokenizer_dir) if tokenizer_dir is not None else root
        tokenizer = OrionTokenizer.load(
            tokenizer_root,
            expected_version=config.tokenizer_version,
            expected_sha256=tokenizer_digest,
        )
        if tokenizer.vocab_size != config.vocab_size:
            raise ArtifactIntegrityError("Tokenizer vocabulary differs from encoder config")
        selected_device = select_device(device)
        model = Orion100MForTasks(
            config,
            subject_labels=len(label_maps["subject"]),
            document_type_labels=len(label_maps["document_type"]),
            topic_labels=len(label_maps["topic"]),
            tag_labels=len(label_maps["tags"]),
            metadata_tags=len(label_maps["metadata_tags"]),
        )
        load_model_state(model, weights_path, strict=True, device="cpu")
        model.to(selected_device)
        model.eval()
        return cls(
            model,
            tokenizer,
            device=selected_device,
            artifact_config=artifact,
            weights_sha256=sha256_file(weights_path),
        )

    @torch.inference_mode()
    def classify(
        self,
        document: str | DocumentPayload | dict[str, Any],
        *,
        registry: Sequence[CourseProfile | dict[str, Any]] | None = None,
        top_k: int = 3,
        tasks: Sequence[str] = ("subject", "document_type", "topic", "course_match", "unknown"),
    ) -> ClassifyResult:
        """Classify a document through its actual encoder and configured task heads."""
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        allowed_tasks = {"subject", "document_type", "topic", "tags", "course_match", "unknown"}
        if not set(tasks).issubset(allowed_tasks):
            raise ValueError(f"tasks may contain only {sorted(allowed_tasks)}")
        encoded = self._encode_document(document)
        if not encoded:
            raise EmptyInputError("document contains no classifiable text")
        rows = self._run_windows([item.input_ids for item in encoded])
        output = self.model(rows["input_ids"], attention_mask=rows["attention_mask"])

        def mean(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.mean(dim=0, keepdim=True)

        subject_logits = mean(output.subject_logits)
        document_logits = mean(output.document_type_logits)
        topic_logits = mean(output.topic_logits)
        tag_logits = mean(output.multilabel_logits)
        confidence_temperature = float(self.temperatures.get("confidence", 1.0))
        if confidence_temperature <= 0:
            raise ArtifactIntegrityError("confidence calibration temperature must be positive")
        confidence = float(
            torch.sigmoid(mean(output.confidence_logits) / confidence_temperature).item()
        )
        warnings: list[str] = []
        subject = (
            self._top_scores(subject_logits, self.labels["subject"], top_k, "subject")
            if "subject" in tasks
            else []
        )
        document_type = (
            self._top_scores(document_logits, self.labels["document_type"], top_k, "document_type")
            if "document_type" in tasks
            else []
        )
        topic = (
            self._top_scores(topic_logits, self.labels["topic"], top_k, "topic")
            if "topic" in tasks
            else []
        )
        tag_thresholds = self.thresholds.get("tags", {})
        tag_probabilities = torch.sigmoid(tag_logits[0])
        tags = []
        if "tags" in tasks:
            for index, label in enumerate(self.labels["tags"]):
                threshold = float(tag_thresholds.get(label, 0.5))
                probability = float(tag_probabilities[index].item())
                if probability >= threshold:
                    tags.append(LabelScore(label=label, confidence=probability))
            tags.sort(key=lambda item: item.confidence, reverse=True)
        course_match = None
        unknown_result: dict[str, float | bool] | None = None
        if "course_match" in tasks and registry:
            courses = [
                item if isinstance(item, CourseProfile) else CourseProfile.model_validate(item)
                for item in registry
            ]
            if not courses:
                warnings.append("The supplied course registry is empty.")
            else:
                profile_texts = [
                    " ".join([course.name, *course.aliases, course.profile_text]).strip()
                    for course in courses
                ]
                profile_vectors = self.get_embeddings(
                    profile_texts, dim=self.model.config.hidden_size
                )
                with torch.inference_mode():
                    course_logits = self.model.course_matcher(
                        output.encoder_output.pooled_output.mean(dim=0, keepdim=True),
                        torch.as_tensor(profile_vectors, dtype=torch.float32, device=self.device),
                    )[0]
                similarities, best = torch.max(course_logits, dim=-1)
                raw_similarity = float(similarities.item())
                min_similarity = float(self.thresholds.get("course_min_similarity", 0.0))
                is_unknown = raw_similarity < min_similarity
                probabilities = torch.softmax(course_logits, dim=-1)
                course_confidence = float(probabilities[best].item())
                course = courses[int(best.item())]
                course_match = CourseMatchResult(
                    course_id=course.course_id,
                    name=course.name,
                    similarity=raw_similarity,
                    confidence=course_confidence,
                    is_unknown=is_unknown,
                )
                unknown_result = {"is_unknown": is_unknown, "score": raw_similarity}
        elif "unknown" in tasks:
            class_threshold = float(self.thresholds.get("unknown_max_known_probability", 0.0))
            known = [
                (index, label)
                for index, label in enumerate(self.labels["subject"])
                if label.casefold() not in {"unknown", "other"}
            ]
            probabilities = torch.softmax(subject_logits[0], dim=-1)
            max_known = max((float(probabilities[index].item()) for index, _ in known), default=0.0)
            unknown_result = {"is_unknown": max_known < class_threshold, "score": 1.0 - max_known}
        return ClassifyResult(
            model=self.model_info(),
            subject=subject,
            document_type=document_type,
            topic=topic,
            tags=tags,
            course_match=course_match,
            unknown=unknown_result,
            confidence=confidence,
            confidence_method="auxiliary_head+temperature_scaling"
            if self.temperatures
            else "auxiliary_head",
            calibrated=bool(self.temperatures),
            evidence=[],
            warnings=warnings,
        )

    def predict(
        self, document: str | DocumentPayload | dict[str, Any], **options: Any
    ) -> ClassifyResult:
        """Single-document alias for :meth:`classify`."""
        return self.classify(document, **options)

    def predict_batch(
        self,
        documents: Sequence[str | DocumentPayload | dict[str, Any]],
        *,
        batch_size: int = 32,
        top_k: int = 3,
    ) -> list[BatchPredictionItem]:
        """Return one result/error per input in original order."""
        if len(documents) > 64:
            raise InputLimitError("predict_batch accepts at most 64 documents")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        results = []
        for document in documents:
            try:
                results.append(
                    BatchPredictionItem(result=self.classify(document, top_k=top_k), error=None)
                )
            except (OrionError, ValueError, TypeError) as exc:
                results.append(
                    BatchPredictionItem(
                        result=None,
                        error={
                            "code": str(getattr(exc, "code", "ORION-E1002")),
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                )
        return results

    @torch.inference_mode()
    def extract(
        self,
        document: str | DocumentPayload | dict[str, Any],
        *,
        fields: Sequence[str] | None = None,
    ) -> ExtractResult:
        """Extract metadata spans with model BIO tags and deterministic regex candidates."""
        source_text = _source_text(document)
        if not source_text.strip():
            raise EmptyInputError("document contains no text for metadata extraction")
        normalized = normalize_text(source_text)
        maximum = self.model.config.max_sequence_length
        encoded = self.tokenizer.encode(normalized, max_length=maximum, truncation="head")
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([encoded.attention_mask], dtype=torch.bool, device=self.device)
        output = self.model(input_ids, attention_mask=attention)
        tag_probabilities, tag_ids = torch.softmax(output.metadata_logits[0].float(), dim=-1).max(
            dim=-1
        )
        tag_names = self.labels["metadata_tags"]
        spans = decode_bio_spans(
            normalized,
            encoded.offset_mapping,
            tag_ids.tolist(),
            tag_names,
            probabilities=tag_probabilities.tolist(),
        )
        deterministic = regex_metadata(normalized)
        selected = prefer_deterministic_spans(spans, deterministic)
        allowed = set(fields) if fields is not None else None
        extracted = [
            ExtractedField(
                name=item.name,
                value=item.value,
                start=item.start,
                end=item.end,
                source=item.source,
                confidence=item.confidence,
            )
            for item in selected
            if allowed is None or item.name in allowed
        ]
        return ExtractResult(model=self.model_info(), fields=extracted)

    @torch.inference_mode()
    def get_embeddings(self, texts: Sequence[str], *, dim: int = 640) -> NDArray[np.float32]:
        """Return normalized masked-mean embeddings or a trained Matryoshka prefix."""
        if dim not in {640, 256, 128} or dim > self.model.config.hidden_size:
            raise ValueError("dim must be one of the supported prefix dimensions 640, 256, or 128")
        if not texts:
            return np.empty((0, dim), dtype=np.float32)
        if len(texts) > 64:
            raise InputLimitError("get_embeddings accepts at most 64 texts per call")
        encoded = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmptyInputError("each embedding input must be a non-empty string")
            item = self.tokenizer.encode(
                text, max_length=self.model.config.max_sequence_length, truncation="head"
            )
            encoded.append(item)
        rows = self._run_windows([item.input_ids for item in encoded])
        output = self.model.encoder(rows["input_ids"], attention_mask=rows["attention_mask"])
        vectors = output.pooled_output[:, :dim]
        vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        return cast(NDArray[np.float32], vectors.cpu().numpy().astype(np.float32, copy=False))

    def model_info(self) -> dict[str, Any]:
        """Return local trained artifact metadata; taxonomy names come from its config."""
        return {
            "name": self.model.config.model_name,
            "version": self.model.config.model_version,
            "tokenizer_version": self.tokenizer.version,
            "weights_sha256": self.weights_sha256,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "max_sequence_length": self.model.config.max_sequence_length,
            "embedding_dimensions": [
                dimension
                for dimension in (640, 256, 128)
                if dimension <= self.model.config.hidden_size
            ],
            "tasks": [
                "subject",
                "document_type",
                "topic",
                "multilabel",
                "course_match",
                "metadata",
                "unknown",
            ],
            "training_data_manifest_sha256": self.artifact_config.get(
                "training_data_manifest_sha256"
            ),
            "artifact_status": self.artifact_config.get("artifact_status", "untrained-instance"),
            "pretrained_weights_imported": False,
        }

    def health(self) -> dict[str, Any]:
        """Report local model and tokenizer state without network calls."""
        return {
            "status": "ready"
            if self.artifact_config.get("artifact_status") == "trained"
            and self.artifact_config.get("training_completed") is True
            else "untrained-instance",
            "device": str(self.device),
            "loaded": True,
            "tokenizer_ok": self.tokenizer.sha256 == self.artifact_config.get("tokenizer_sha256"),
            "weights_ok": self.artifact_config.get("artifact_status") == "trained"
            and self.artifact_config.get("training_completed") is True,
            "latency_probe_ms": None,
        }

    def _encode_document(
        self, document: str | DocumentPayload | dict[str, Any]
    ) -> list[_EncodedDocument]:
        if isinstance(document, str):
            payload = DocumentPayload(text=document)
        elif isinstance(document, DocumentPayload):
            payload = document
        else:
            payload = DocumentPayload.model_validate(document)
        fields: list[tuple[str, bool]] = [("<|cls|>", True)]
        for marker, value in (
            ("<meta:filename>", payload.filename),
            ("<meta:ext>", payload.file_type),
            ("<meta:title>", payload.title),
        ):
            if value:
                fields.extend(((marker, True), (value, False), ("<field_sep>", True)))
        for heading in payload.headings:
            fields.extend((("<meta:heading>", True), (heading, False), ("<field_sep>", True)))
        body = payload.text or "\n".join(payload.excerpts)
        if body:
            fields.extend((("<meta:text>", True), (body, False)))
        fields.append(("<|sep|>", True))
        normalized_parts = [
            (text if trusted else normalize_text(text), trusted) for text, trusted in fields
        ]
        nonempty = any(text.strip() and not trusted for text, trusted in normalized_parts)
        if not nonempty:
            return []
        full = self.tokenizer.encode_parts(tuple(normalized_parts))
        max_length = self.model.config.max_sequence_length
        if len(full.input_ids) <= max_length:
            return [_EncodedDocument(full.input_ids, body)]
        if max_length < 4:
            raise InputLimitError("model max_sequence_length must be at least four tokens")
        content = full.input_ids[1:-1]
        width = max_length - 2
        window_count = min(32, max(1, (len(content) + width - 1) // width))
        starts = sorted(
            {
                round(index * max(0, len(content) - width) / max(1, window_count - 1))
                for index in range(window_count)
            }
        )
        cls_id, sep_id = (
            self.tokenizer.special_token_id("<|cls|>"),
            self.tokenizer.special_token_id("<|sep|>"),
        )
        return [
            _EncodedDocument((cls_id, *content[start : start + width], sep_id), body)
            for start in starts[:32]
        ]

    def _run_windows(self, windows: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
        if not windows:
            raise EmptyInputError("no token windows were supplied")
        if len(windows) > 2048:
            raise InputLimitError("a call may contain at most 2,048 encoder windows")
        width = max(len(item) for item in windows)
        if width > self.model.config.max_sequence_length:
            raise InputLimitError("a token window exceeds max_sequence_length")
        pad_id = self.model.config.pad_token_id
        ids = torch.full((len(windows), width), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(windows), width), dtype=torch.bool, device=self.device)
        for index, values in enumerate(windows):
            ids[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=self.device)
            mask[index, : len(values)] = True
        return {"input_ids": ids, "attention_mask": mask}

    def _top_scores(
        self,
        logits: torch.Tensor,
        labels: list[str],
        top_k: int,
        task: str,
    ) -> list[LabelScore]:
        if len(labels) != logits.shape[-1]:
            raise ArtifactIntegrityError(f"{task} labels do not match the loaded head size")
        temperature = float(self.temperatures.get(task, 1.0))
        if temperature <= 0:
            raise ArtifactIntegrityError(f"{task} calibration temperature must be positive")
        probabilities = torch.softmax(logits[0].float() / temperature, dim=-1)
        count = min(top_k, len(labels))
        scores, indices = torch.topk(probabilities, count)
        return [
            LabelScore(label=labels[int(index)], confidence=float(score))
            for score, index in zip(scores, indices, strict=True)
        ]


def _source_text(document: str | DocumentPayload | dict[str, Any]) -> str:
    if isinstance(document, str):
        return document
    payload = (
        document
        if isinstance(document, DocumentPayload)
        else DocumentPayload.model_validate(document)
    )
    if payload.text:
        return payload.text
    return "\n".join(item for item in [payload.title, *payload.headings, *payload.excerpts] if item)
