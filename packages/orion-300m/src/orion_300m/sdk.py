"""Offline ORION-300M SDK for context-grounded educational generation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.model.generation import GenerationConfig, generate_ids, stream_ids
from orion_core.device import select_device
from orion_core.errors import (
    ArtifactIntegrityError,
    EmptyInputError,
    InputLimitError,
    ModelWeightsNotFound,
    NoContextError,
    TokenizerMismatchError,
)
from orion_core.hashing import sha256_file
from orion_core.serialization import load_model_state
from orion_tokenizer import OrionTokenizer
from orion_tokenizer.spec import SPECIAL_TOKEN_IDS

_CITATION_IDS = {SPECIAL_TOKEN_IDS[f"<|cite_{index}|>"]: index for index in range(1, 17)}
_TASKS = {
    "generate",
    "answer",
    "explain",
    "summarize",
    "compare",
    "short_answer",
    "mcq",
    "quiz",
    "flashcards",
}


@dataclass(frozen=True)
class GenerationResult:
    """Generation result with citations mapped only to supplied context documents."""

    schema_version: str
    request_id: str | None
    model: dict[str, Any]
    task: str
    status: str
    text: str
    citations: tuple[dict[str, Any], ...]
    source_documents: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    structured: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Invalid model artifact file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"{path.name} must contain a JSON object")
    return value


class Orion300M:
    """Standalone generation SDK. It never downloads weights or tokenizer assets."""

    def __init__(
        self,
        model: Orion300MDecoder,
        tokenizer: OrionTokenizer,
        *,
        device: torch.device,
        manifest: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.manifest = manifest

    @classmethod
    def load(
        cls,
        model_dir: str | Path | None = None,
        *,
        device: str = "auto",
    ) -> Orion300M:
        """Load a locally installed, hash-verified trained model release."""
        root = Path(
            model_dir
            or os.environ.get("ORION_300M_HOME", Path.home() / ".cache/orion-ai/orion-300m")
        )
        if not root.is_dir():
            raise ModelWeightsNotFound(f"No local ORION-300M release is installed at {root}")
        artifact_path = root / "artifact.json"
        if not artifact_path.is_file():
            raise ModelWeightsNotFound(f"No trained ORION-300M artifact manifest exists at {root}")
        manifest = _read_json(artifact_path)
        if manifest.get("model_name") != "orion-300m":
            raise ArtifactIntegrityError("Artifact is not an ORION-300M release")
        if manifest.get("trained") is not True or manifest.get("status") not in {
            "accepted",
            "preview",
        }:
            raise ModelWeightsNotFound(
                "Artifact is not marked as a trained accepted/preview release"
            )
        for field in ("training_report", "evaluation_report", "data_manifest_sha256"):
            if not manifest.get(field):
                raise ArtifactIntegrityError(f"A trained release is missing {field}")
        for name in (
            "weights_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "data_manifest_sha256",
        ):
            if not isinstance(manifest.get(name), str) or len(manifest[name]) != 64:
                raise ArtifactIntegrityError(f"A trained release has an invalid {name}")
            if any(character not in "0123456789abcdef" for character in manifest[name].lower()):
                raise ArtifactIntegrityError(f"A trained release has an invalid {name}")

        weights_path, config_path = root / "model.safetensors", root / "config.json"
        for path, digest_key in (
            (weights_path, "weights_sha256"),
            (config_path, "config_sha256"),
        ):
            if not path.is_file():
                raise ModelWeightsNotFound(f"Required release file is missing: {path.name}")
            if sha256_file(path) != manifest[digest_key]:
                raise ArtifactIntegrityError(f"{path.name} SHA-256 does not match artifact.json")
        try:
            config = Orion300MConfig.model_validate(_read_json(config_path))
        except ValidationError as exc:
            raise ArtifactIntegrityError(f"Invalid ORION-300M config: {exc}") from exc
        if config.model_name != "orion-300m":
            raise ArtifactIntegrityError("Model config does not identify ORION-300M")
        if (
            manifest.get("tokenizer_version") != config.tokenizer_version
            or manifest.get("model_version", config.model_version) != config.model_version
        ):
            raise ArtifactIntegrityError(
                "Artifact manifest and model configuration versions differ"
            )
        if manifest.get("tokenizer_version") != config.tokenizer_version:
            raise ArtifactIntegrityError(
                "Artifact manifest and model config tokenizer versions differ"
            )
        tokenizer_dir = root / "tokenizer"
        if not tokenizer_dir.is_dir():
            tokenizer_dir = root
        tokenizer = OrionTokenizer.load(
            tokenizer_dir,
            expected_version=config.tokenizer_version,
            expected_sha256=manifest["tokenizer_sha256"],
        )
        if tokenizer.vocab_size != config.vocab_size:
            raise TokenizerMismatchError("Tokenizer vocabulary differs from decoder config")
        selected = select_device(device)
        model = Orion300MDecoder(config)
        load_model_state(model, weights_path, strict=True, device="cpu")
        model.to(selected).eval()
        return cls(model, tokenizer, device=selected, manifest=manifest)

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path | None = None, *, device: str = "auto"
    ) -> Orion300M:
        """Alias for local-only ``load``; no repository or network lookup is performed."""
        return cls.load(model_dir, device=device)

    def _document(self, value: Any, index: int) -> tuple[str, str, str, int | None, int | None]:
        if isinstance(value, dict):
            document_id = str(value.get("id", index))
            title = str(value.get("title", value.get("label", "")))
            text = value.get("text", value.get("content", ""))
            page = value.get("page")
            slide = value.get("slide")
        else:
            document_id, title, text = str(index), "", value
            page = slide = None
        if not isinstance(text, str) or not text.strip():
            raise EmptyInputError(f"Context document {index} must have nonempty text")
        return document_id, title, text, page, slide

    def _prompt(
        self,
        *,
        task: str,
        question: str,
        documents: list[Any],
        max_new_tokens: int,
        json_mode: bool,
    ) -> tuple[list[int], list[tuple[str, str, int | None, int | None]]]:
        if task not in _TASKS:
            raise ValueError(f"task must be one of: {', '.join(sorted(_TASKS))}")
        if not isinstance(question, str) or not question.strip():
            raise EmptyInputError("question or task instruction must contain text")
        if not documents and task != "generate":
            raise NoContextError("No context documents were supplied; generation was not called")
        if len(documents) > 16:
            raise InputLimitError("At most 16 context documents can be cited in one request")
        if max_new_tokens <= 0 or max_new_tokens > self.model.config.max_generation_tokens:
            raise InputLimitError(
                f"max_new_tokens must be in [1,{self.model.config.max_generation_tokens}]"
            )

        instruction = (
            "Continue the requested text naturally. Do not claim to have consulted external sources."
            if task == "generate"
            else (
                f"Perform the educational task '{task}'. Use only the supplied context. "
                "If the context does not support an answer, emit the abstain control. "
                "When a claim uses a context item, cite it with its matching citation marker. "
                "Do not invent sources."
            )
        )
        ids = [SPECIAL_TOKEN_IDS["<|bos|>"], SPECIAL_TOKEN_IDS["<|system|>"]]
        ids.extend(self.tokenizer.encode(instruction).input_ids)
        ids.append(SPECIAL_TOKEN_IDS["<|user|>"])
        ids.extend(self.tokenizer.encode(question).input_ids)
        if documents:
            ids.append(SPECIAL_TOKEN_IDS["<|context|>"])
        context_budget = self.model.config.context_length - max_new_tokens - 3
        if context_budget <= len(ids):
            raise InputLimitError("Question and generation reserve leave no room for context")
        selected: list[tuple[str, str, int | None, int | None]] = []
        for index, raw in enumerate(documents, start=1):
            document_id, title, text, page, slide = self._document(raw, index)
            title_ids = self.tokenizer.encode(title).input_ids if title.strip() else ()
            text_ids = self.tokenizer.encode(text).input_ids
            entry_prefix = [
                SPECIAL_TOKEN_IDS[f"<|doc_{index}|>"],
                *title_ids,
                SPECIAL_TOKEN_IDS["<field_sep>"],
            ]
            remaining = context_budget - len(ids) - len(entry_prefix) - 1
            if remaining <= 0:
                break
            clipped = text_ids[:remaining]
            if not clipped:
                continue
            ids.extend(entry_prefix)
            ids.extend(clipped)
            selected.append((document_id, title, page, slide))
        if documents and not selected:
            raise InputLimitError("No context text fits within the model context budget")
        if documents:
            ids.append(SPECIAL_TOKEN_IDS["<|/context|>"])
        ids.append(SPECIAL_TOKEN_IDS["<|assistant|>"])
        if json_mode:
            ids.append(SPECIAL_TOKEN_IDS["<|json|>"])
        if len(ids) + max_new_tokens > self.model.config.context_length:
            raise InputLimitError("Prompt and requested output exceed the model context limit")
        return ids, selected

    def generate(
        self,
        question: str,
        *,
        documents: list[Any] | None = None,
        task: str = "generate",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 0,
        seed: int | None = None,
        json_mode: bool = False,
        request_id: str | None = None,
    ) -> GenerationResult:
        """Generate from supplied context and map citation tokens to those documents."""
        documents = documents or []
        if not documents and task != "generate":
            return GenerationResult(
                schema_version="1.0",
                request_id=request_id,
                model=self.model_info(),
                task=task,
                status="no_context",
                text="",
                citations=(),
                source_documents=(),
                warnings=("No context documents were supplied; the model was not called.",),
            )
        prompt, selected_docs = self._prompt(
            task=task,
            question=question,
            documents=documents,
            max_new_tokens=max_new_tokens,
            json_mode=json_mode,
        )
        sampling = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_token_ids=(
                self.model.config.eos_token_id,
                self.model.config.abstain_token_id,
                self.model.config.end_token_id,
            ),
        )
        result_ids = generate_ids(
            self.model,
            torch.tensor([prompt], dtype=torch.long, device=self.device),
            generation_config=sampling,
        )[0]
        text_parts: list[str] = []
        ordinary_ids: list[int] = []
        citations: list[dict[str, Any]] = []
        warnings: list[str] = []
        abstained = False

        def flush_text() -> None:
            if ordinary_ids:
                text_parts.append(self.tokenizer.decode(ordinary_ids, skip_special_tokens=True))
                ordinary_ids.clear()

        for token_id in result_ids:
            if token_id in {self.model.config.eos_token_id, self.model.config.end_token_id}:
                break
            if token_id == self.model.config.abstain_token_id:
                flush_text()
                abstained = True
                break
            citation_number = _CITATION_IDS.get(token_id)
            if citation_number is not None:
                flush_text()
                if citation_number <= len(selected_docs):
                    source_id, title, page, slide = selected_docs[citation_number - 1]
                    text_parts.append(f"[{citation_number}]")
                    citations.append(
                        {
                            "marker": citation_number,
                            "document_id": source_id,
                            "title": title,
                            "page": page,
                            "slide": slide,
                            "support_verified": False,
                        }
                    )
                else:
                    warnings.append(f"removed_unmapped_citation:{citation_number}")
            else:
                ordinary_ids.append(token_id)
        flush_text()
        text = re.sub(r"[ \t]+\n", "\n", "".join(text_parts)).strip()
        if abstained:
            text = ""
            citations.clear()
        structured = None
        status = "abstained" if abstained else "generated"
        if json_mode and not abstained:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                status = "invalid_structured_output"
                warnings.append("generated_text_is_not_valid_json")
        return GenerationResult(
            schema_version="1.0",
            request_id=request_id,
            model=self.model_info(),
            task=task,
            status=status,
            text=text,
            citations=tuple(citations),
            source_documents=tuple(
                {"id": document_id, "title": title, "page": page, "slide": slide}
                for document_id, title, page, slide in selected_docs
            ),
            warnings=tuple(warnings),
            structured=structured,
        )

    def generate_batch(self, requests: Sequence[dict[str, Any]]) -> list[GenerationResult]:
        """Generate one independently validated result per request, preserving order."""
        if not requests:
            raise ValueError("requests must contain at least one generation request")
        if len(requests) > 32:
            raise InputLimitError("generate_batch accepts at most 32 requests")
        allowed = {
            "prompt",
            "documents",
            "task",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "json_mode",
            "request_id",
        }
        results = []
        for index, request in enumerate(requests):
            if not isinstance(request, dict) or set(request) - allowed:
                raise ValueError(f"generation request {index} has unsupported fields")
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"generation request {index} requires a prompt string")
            results.append(
                self.generate(
                    prompt, **{key: value for key, value in request.items() if key != "prompt"}
                )
            )
        return results

    def stream_generate(
        self,
        question: str,
        *,
        documents: list[Any] | None = None,
        task: str = "generate",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 0,
        seed: int | None = None,
        json_mode: bool = False,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield actual decoded model tokens and a final citation/status event."""
        documents = documents or []
        if not documents and task != "generate":

            def no_context_events() -> Iterator[dict[str, Any]]:
                yield {
                    "event": "done",
                    "schema_version": "1.0",
                    "request_id": request_id,
                    "model": self.model_info(),
                    "task": task,
                    "status": "no_context",
                    "text": "",
                    "citations": [],
                    "source_documents": [],
                    "warnings": ["No context documents were supplied; the model was not called."],
                }

            return no_context_events()

        prompt, selected_docs = self._prompt(
            task=task,
            question=question,
            documents=documents,
            max_new_tokens=max_new_tokens,
            json_mode=json_mode,
        )
        sampling = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_token_ids=(
                self.model.config.eos_token_id,
                self.model.config.abstain_token_id,
                self.model.config.end_token_id,
            ),
        )

        def events() -> Iterator[dict[str, Any]]:
            ordinary_ids: list[int] = []
            emitted_text = ""
            output_parts: list[str] = []
            citations: list[dict[str, Any]] = []
            warnings: list[str] = []
            abstained = False

            for token_id in stream_ids(self.model, prompt, generation_config=sampling):
                if token_id in {
                    self.model.config.eos_token_id,
                    self.model.config.end_token_id,
                }:
                    break
                if token_id == self.model.config.abstain_token_id:
                    abstained = True
                    output_parts = []
                    citations.clear()
                    yield {"event": "replace", "text": ""}
                    break
                citation_number = _CITATION_IDS.get(token_id)
                if citation_number is not None:
                    decoded = self.tokenizer.decode(ordinary_ids, skip_special_tokens=True)
                    if decoded.startswith(emitted_text):
                        delta = decoded[len(emitted_text) :]
                        if delta:
                            output_parts.append(delta)
                            yield {"event": "delta", "text": delta}
                    elif decoded != emitted_text:
                        output_parts = [decoded]
                        yield {"event": "replace", "text": decoded}
                    emitted_text = decoded
                    ordinary_ids.clear()
                    if citation_number <= len(selected_docs):
                        source_id, title, page, slide = selected_docs[citation_number - 1]
                        marker = f"[{citation_number}]"
                        output_parts.append(marker)
                        citations.append(
                            {
                                "marker": citation_number,
                                "document_id": source_id,
                                "title": title,
                                "page": page,
                                "slide": slide,
                                "support_verified": False,
                            }
                        )
                        yield {"event": "delta", "text": marker}
                    else:
                        warnings.append(f"removed_unmapped_citation:{citation_number}")
                    emitted_text = ""
                    continue

                ordinary_ids.append(token_id)
                decoded = self.tokenizer.decode(ordinary_ids, skip_special_tokens=True)
                if decoded.startswith(emitted_text):
                    delta = decoded[len(emitted_text) :]
                    if delta:
                        output_parts.append(delta)
                        yield {"event": "delta", "text": delta}
                    emitted_text = decoded
                elif decoded != emitted_text:
                    output_parts = [decoded]
                    yield {"event": "replace", "text": decoded}
                    emitted_text = decoded

            if abstained:
                text = ""
            else:
                final_segment = self.tokenizer.decode(ordinary_ids, skip_special_tokens=True)
                if final_segment.startswith(emitted_text):
                    delta = final_segment[len(emitted_text) :]
                    if delta:
                        output_parts.append(delta)
                        yield {"event": "delta", "text": delta}
                elif final_segment != emitted_text:
                    output_parts = [final_segment]
                    yield {"event": "replace", "text": final_segment}
                text = re.sub(r"[ \t]+\n", "\n", "".join(output_parts)).strip()
            structured = None
            status = "abstained" if abstained else "generated"
            if json_mode and not abstained:
                try:
                    structured = json.loads(text)
                except json.JSONDecodeError:
                    status = "invalid_structured_output"
                    warnings.append("generated_text_is_not_valid_json")
            yield {
                "event": "done",
                "schema_version": "1.0",
                "request_id": request_id,
                "model": self.model_info(),
                "task": task,
                "status": status,
                "text": text,
                "structured": structured,
                "citations": citations,
                "source_documents": [
                    {"id": document_id, "title": title, "page": page, "slide": slide}
                    for document_id, title, page, slide in selected_docs
                ],
                "warnings": warnings,
            }

        return events()

    def answer(self, question: str, *, documents: list[Any], **kwargs: Any) -> GenerationResult:
        return self.generate(question, documents=documents, task="answer", **kwargs)

    def explain(self, concept: str, *, documents: list[Any], **kwargs: Any) -> GenerationResult:
        return self.generate(concept, documents=documents, task="explain", **kwargs)

    def summarize(self, *, documents: list[Any], **kwargs: Any) -> GenerationResult:
        return self.generate(
            "Summarize the supplied educational material.",
            documents=documents,
            task="summarize",
            **kwargs,
        )

    def generate_questions(
        self, *, documents: list[Any], count: int = 5, **kwargs: Any
    ) -> GenerationResult:
        if not 1 <= count <= 25:
            raise ValueError("question count must be in [1,25]")
        return self.generate(
            f"Write {count} educational questions supported by the context.",
            documents=documents,
            task="quiz",
            json_mode=True,
            **kwargs,
        )

    def compare(
        self, concepts: list[str] | tuple[str, ...], *, documents: list[Any], **kwargs: Any
    ) -> GenerationResult:
        if len(concepts) < 2 or any(not item.strip() for item in concepts):
            raise ValueError("compare requires at least two non-empty concepts")
        return self.generate(
            "Compare these concepts: " + "; ".join(concepts),
            documents=documents,
            task="compare",
            **kwargs,
        )

    def generate_mcq(self, topic: str, *, documents: list[Any], **kwargs: Any) -> GenerationResult:
        return self.generate(
            f"Create one multiple-choice question about: {topic}",
            documents=documents,
            task="mcq",
            json_mode=True,
            **kwargs,
        )

    def generate_quiz(
        self, topic: str, *, documents: list[Any], count: int = 5, **kwargs: Any
    ) -> GenerationResult:
        if not 1 <= count <= 25:
            raise ValueError("quiz count must be in [1,25]")
        return self.generate(
            f"Create a quiz with {count} questions about: {topic}",
            documents=documents,
            task="quiz",
            json_mode=True,
            **kwargs,
        )

    def generate_flashcards(
        self, topic: str, *, documents: list[Any], count: int = 8, **kwargs: Any
    ) -> GenerationResult:
        if not 1 <= count <= 50:
            raise ValueError("flashcard count must be in [1,50]")
        return self.generate(
            f"Create {count} study flashcards about: {topic}",
            documents=documents,
            task="flashcards",
            json_mode=True,
            **kwargs,
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "name": "orion-300m",
            "version": self.manifest.get("model_version", self.model.config.model_version),
            "tokenizer_version": self.tokenizer.version,
            "weights_sha256": self.manifest.get("weights_sha256"),
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "context_length": self.model.config.context_length,
            "max_generation_tokens": self.model.config.max_generation_tokens,
            "trained": self.manifest.get("trained") is True,
            "status": self.manifest.get("status"),
        }

    def health(self) -> dict[str, Any]:
        """Report local readiness without making a network call."""
        return {
            "status": "ready" if self.manifest.get("trained") is True else "untrained-instance",
            "loaded": True,
            "device": str(self.device),
            "tokenizer_ok": self.tokenizer.sha256 == self.manifest.get("tokenizer_sha256"),
            "weights_ok": self.manifest.get("trained") is True,
            "latency_probe_ms": None,
        }


__all__ = ["GenerationResult", "Orion300M"]
