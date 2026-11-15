"""Reproducible local manifest-to-training-shard preparation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion_core.hashing import sha256_file, sha256_text
from orion_data.contamination import ContaminationIndex
from orion_data.dedup import MinHashLSH
from orion_data.manifest import DatasetManifest, DatasetSource
from orion_data.normalize import mask_pii, normalize_text
from orion_data.packing import pack_documents, write_memmap_shard
from orion_data.quality import check_text_quality
from orion_data.segment import segment_document
from orion_data.shards import read_jsonl, write_jsonl
from orion_data.tokenize import Encoder, write_aligned_token_shards

Split = Literal["train", "validation", "test"]
_SPLIT_PRIORITY: dict[str, int] = {"test": 0, "validation": 1, "train": 2}
_SAFE_METADATA_FIELDS = (
    "title",
    "course_title",
    "document_type",
    "semester",
    "date",
    "page",
    "slide",
)


class PreparationConfig(BaseModel):
    """Validated controls for deterministic local data preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 1729
    split_proportions: tuple[float, float, float] = (0.98, 0.01, 0.01)
    sequence_length: int = Field(default=2048, ge=2, le=4096)
    max_segment_characters: int = Field(default=4000, ge=1)
    min_characters: int = Field(default=80, ge=1)
    max_symbol_ratio: float = Field(default=0.45, ge=0, le=1)
    max_repeated_line_ratio: float = Field(default=0.35, ge=0, le=1)
    near_duplicate_threshold: float = Field(default=0.85, gt=0, le=1)
    near_duplicate_permutations: int = Field(default=64, ge=16)
    near_duplicate_bands: int = Field(default=16, ge=1)
    mask_personal_data: bool = True
    fold_urdu_variants: bool = False

    @model_validator(mode="after")
    def validate_splits_and_minhash(self) -> PreparationConfig:
        if len(self.split_proportions) != 3 or any(value < 0 for value in self.split_proportions):
            raise ValueError("split_proportions must contain three non-negative values")
        if abs(sum(self.split_proportions) - 1.0) > 1e-9:
            raise ValueError("split_proportions must sum to one")
        if self.near_duplicate_permutations % self.near_duplicate_bands:
            raise ValueError("near_duplicate_permutations must be divisible by bands")
        return self


@dataclass(frozen=True, slots=True)
class InputDocument:
    """One source record with its manifest-derived provenance."""

    document_id: str
    text: str
    source_id: str
    source_revision: str
    source_sha256: str
    license_spdx_id: str
    attribution: str
    course_id: str | None = None
    explicit_split: Split | None = None
    language: str | None = None
    metadata: Mapping[str, object] | None = None


def prepare_records(
    documents: Iterable[InputDocument], config: PreparationConfig
) -> tuple[dict[Split, list[dict[str, object]]], dict[str, int]]:
    """Normalize, filter, split, segment and deduplicate in deterministic order.

    Explicit splits are preserved and must be consistent for every course group.
    Otherwise, courses are assigned as a unit and documents without a course ID
    are split by document ID. Evaluation duplicates are kept ahead of copies in
    training so held-out text cannot leak into the training shard.
    """
    items = list(documents)
    if not items:
        raise ValueError("dataset contains no source records")
    ids = [item.document_id for item in items]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("document_id values must be non-empty and globally unique")

    group_splits: dict[str, tuple[bool, Split]] = {}
    assigned: dict[str, Split] = {}
    proportions = config.split_proportions
    for item in items:
        group_id = item.course_id.strip() if item.course_id else item.document_id
        if not group_id:
            raise ValueError(f"course_id must not be blank for {item.document_id}")
        explicit = item.explicit_split is not None
        split = item.explicit_split or _deterministic_split(group_id, config.seed, proportions)
        previous = group_splits.get(group_id)
        if previous is not None and previous[1] != split:
            raise ValueError(f"course/document group {group_id!r} crosses data splits")
        if previous is not None and previous[0] != explicit:
            raise ValueError(
                f"course/document group {group_id!r} mixes explicit and generated splits"
            )
        group_splits[group_id] = (explicit, split)
        assigned[item.document_id] = split

    counters: Counter[str] = Counter(input_documents=len(items))
    segments: list[dict[str, object]] = []
    for item in items:
        normalized = normalize_text(item.text, fold_urdu_variants=config.fold_urdu_variants)
        if config.mask_personal_data:
            normalized, pii_counts = mask_pii(normalized)
        else:
            pii_counts = {}
        quality = check_text_quality(
            normalized,
            min_characters=config.min_characters,
            max_symbol_ratio=config.max_symbol_ratio,
            max_repeated_line_ratio=config.max_repeated_line_ratio,
        )
        if not quality.accepted:
            counters[f"rejected_{quality.reason}"] += 1
            continue
        counters["accepted_documents"] += 1
        for segment in segment_document(normalized, max_characters=config.max_segment_characters):
            segment_quality = check_text_quality(
                segment.text,
                min_characters=config.min_characters,
                max_symbol_ratio=config.max_symbol_ratio,
                max_repeated_line_ratio=config.max_repeated_line_ratio,
            )
            if not segment_quality.accepted:
                counters[f"rejected_segment_{segment_quality.reason}"] += 1
                continue
            metadata = {
                key: item.metadata[key]
                for key in _SAFE_METADATA_FIELDS
                if item.metadata is not None and key in item.metadata
            }
            segments.append(
                {
                    "document_id": item.document_id,
                    "segment_id": f"{item.document_id}#segment-{segment.index:05d}",
                    "segment_index": segment.index,
                    "segment_kind": segment.kind,
                    "course_id": item.course_id,
                    "split": assigned[item.document_id],
                    "source_id": item.source_id,
                    "source_revision": item.source_revision,
                    "source_sha256": item.source_sha256,
                    "license_spdx_id": item.license_spdx_id,
                    "attribution": item.attribution,
                    "language": item.language or _language_hint(normalized),
                    "text": segment.text,
                    "text_sha256": _text_hash(segment.text),
                    "pii_counts": pii_counts,
                    "metadata": metadata,
                }
            )
            counters["segments_before_deduplication"] += 1

    # Held-out splits are indexed first. Any high-similarity copy in a lower
    # priority split is removed, which prevents known copies entering training.
    segments.sort(
        key=lambda record: (
            _SPLIT_PRIORITY[str(record["split"])],
            str(record["text_sha256"]),
            str(record["segment_id"]),
        )
    )
    heldout_texts = {
        str(record["segment_id"]): str(record["text"])
        for record in segments
        if record["split"] in {"validation", "test"}
    }
    contamination_index = ContaminationIndex(heldout_texts) if heldout_texts else None
    index = MinHashLSH(
        num_perm=config.near_duplicate_permutations,
        bands=config.near_duplicate_bands,
        threshold=config.near_duplicate_threshold,
    )
    seen_exact: dict[str, Split] = {}
    split_by_segment: dict[str, str] = {}
    prepared: dict[Split, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in segments:
        segment_id = str(record["segment_id"])
        split_name = str(record["split"])
        if split_name not in _SPLIT_PRIORITY:
            raise RuntimeError(f"internal split assignment is invalid: {split_name}")
        record_split = cast(Split, split_name)
        digest = str(record["text_sha256"])
        if digest in seen_exact:
            counters["exact_duplicates_removed"] += 1
            if seen_exact[digest] != record_split:
                counters["cross_split_duplicates_removed"] += 1
            continue
        if record_split == "train" and contamination_index is not None:
            hits = contamination_index.check(str(record["text"]))
            if hits:
                counters["contamination_matches_removed"] += 1
                continue
        matches = index.query(segment_id, str(record["text"]))
        if matches:
            counters["near_duplicates_removed"] += 1
            if any(split_by_segment[item.document_id] != record_split for item in matches):
                counters["cross_split_duplicates_removed"] += 1
            continue
        seen_exact[digest] = record_split
        index.add(segment_id, str(record["text"]))
        split_by_segment[segment_id] = record_split
        prepared[record_split].append(record)

    counters["segments_kept"] = sum(len(values) for values in prepared.values())
    for split, values in prepared.items():
        counters[f"{split}_segments"] = len(values)
    return prepared, dict(sorted(counters.items()))


def prepare_dataset(
    manifest_path: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    config: PreparationConfig | None = None,
    license_mode: str = "research",
) -> dict[str, object]:
    """Build a local dataset from checksum-pinned JSONL sources and a trained tokenizer.

    This function never downloads. It refuses synthetic manifests and requires
    each local source to have a verified manifest checksum and license record.
    """
    source_manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = DatasetManifest.load(source_manifest_path)
    if manifest.synthetic_fixture or any(source.synthetic for source in manifest.sources):
        raise ValueError("synthetic fixture manifests cannot create training datasets")
    manifest.validate_license_policy(license_mode, purpose="training")
    documents, source_records = _load_source_documents(source_manifest_path, manifest)
    preparation_config = config or PreparationConfig()
    split_records, counters = prepare_records(documents, preparation_config)
    if not split_records["train"]:
        raise ValueError("prepared dataset has no training segments")

    try:
        from orion_tokenizer.tokenizer import OrionTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Dataset preparation requires the tokenizer extra: pip install orion-data[prepare]"
        ) from exc
    tokenizer = OrionTokenizer.load(tokenizer_dir)
    return _write_prepared_outputs(
        split_records,
        tokenizer,
        output_dir,
        config=preparation_config,
        dataset_id=manifest.dataset_id,
        source_manifest_sha256=sha256_file(source_manifest_path),
        sources=source_records,
        counters=counters,
        synthetic_fixture=False,
    )


def verify_prepared_dataset(dataset_dir: str | Path) -> dict[str, object]:
    """Verify prepared file hashes, token alignment, and split leakage invariants."""
    root = Path(dataset_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("prepared dataset path must be a directory")
    manifest_path = root / "prepared_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prepared dataset manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported prepared dataset manifest")
    status = manifest.get("status")
    synthetic = manifest.get("synthetic_fixture")
    if status not in {"prepared", "synthetic-test-fixture"} or type(synthetic) is not bool:
        raise ValueError("prepared dataset status metadata is invalid")
    if synthetic != (status == "synthetic-test-fixture"):
        raise ValueError("prepared dataset synthetic status is inconsistent")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("prepared dataset has no hashed file manifest")
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("prepared dataset file manifest entries must be strings")
        relative_path = Path(relative)
        candidate = root / relative_path
        current = root
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"prepared dataset manifest contains a symlink: {relative}")
        target = candidate.resolve()
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not target.is_relative_to(root)
            or not target.is_file()
        ):
            raise ValueError(f"prepared dataset manifest contains an unsafe path: {relative}")
        if sha256_file(target) != digest:
            raise ValueError(f"SHA-256 mismatch for prepared file {relative}")

    document_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    text_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        records_path = root / split / "records.jsonl"
        if not records_path.is_file():
            raise ValueError(f"prepared dataset is missing {split}/records.jsonl")
        for row in read_jsonl(records_path):
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError(f"record split label disagrees with {split}/records.jsonl")
            document_id = row.get("document_id")
            if not isinstance(document_id, str):
                raise ValueError("prepared record has no string document_id")
            prior_document_split = document_splits.setdefault(document_id, split)
            if prior_document_split != split:
                raise ValueError(f"document {document_id} crosses prepared splits")
            course_id = row.get("course_id")
            group_id = course_id if isinstance(course_id, str) and course_id else document_id
            prior_group_split = group_splits.setdefault(group_id, split)
            if prior_group_split != split:
                raise ValueError(f"course/document group {group_id} crosses prepared splits")
            text_hash = row.get("text_sha256")
            text = row.get("text")
            if not isinstance(text_hash, str) or not isinstance(text, str):
                raise ValueError("prepared record has no text SHA-256")
            if sha256_text(text) != text_hash:
                raise ValueError(f"text SHA-256 mismatch for document {document_id}")
            prior_text_split = text_splits.setdefault(text_hash, split)
            if prior_text_split != split:
                raise ValueError("identical text appears in more than one prepared split")

    for split in ("train", "validation", "test"):
        token_manifest_path = root / split / "tokens" / "manifest.json"
        if not token_manifest_path.exists():
            continue
        token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(token_manifest, dict):
            raise ValueError(f"invalid token shard manifest for split {split}")
        token_record = token_manifest.get("token_shard")
        word_record = token_manifest.get("word_id_shard")
        if (
            not isinstance(token_record, dict)
            or not isinstance(word_record, dict)
            or token_record.get("token_count") != word_record.get("value_count")
        ):
            raise ValueError(f"token and word-ID shards are misaligned for split {split}")
    return {
        "verified": True,
        "dataset_id": manifest.get("dataset_id"),
        "status": status,
        "file_count": len(files),
        "record_count": len(document_splits),
        "synthetic_fixture": synthetic,
    }


def _load_source_documents(
    manifest_path: Path, manifest: DatasetManifest
) -> tuple[list[InputDocument], list[dict[str, str]]]:
    documents: list[InputDocument] = []
    source_records: list[dict[str, str]] = []
    source_ids: set[str] = set()
    for source in manifest.sources:
        if source.kind != "local-jsonl":
            raise ValueError(
                f"Local preparation supports only local-jsonl sources; got {source.kind!r}"
            )
        if source.source_id in source_ids:
            raise ValueError(f"duplicate source_id in manifest: {source.source_id}")
        source_ids.add(source.source_id)
        if not source.local_path or not source.sha256:
            raise ValueError(
                f"local source {source.source_id} requires local_path and sha256 in the manifest"
            )
        source_path = Path(source.local_path)
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve(strict=True)
        if not source_path.is_file() or source_path.suffix.lower() != ".jsonl":
            raise ValueError(f"source {source.source_id} must be a local JSONL file")
        digest = sha256_file(source_path)
        if digest != source.sha256:
            raise ValueError(f"SHA-256 mismatch for local source {source.source_id}")
        if not source.license.verified:
            raise ValueError(f"license for source {source.source_id} is not verified")
        source_records.append(
            {
                "source_id": source.source_id,
                "revision": source.revision,
                "sha256": digest,
                "license_spdx_id": source.license.spdx_id,
                "attribution": source.license.attribution,
            }
        )
        for line_number, raw in enumerate(read_jsonl(source_path), start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a JSON object at {source_path}:{line_number}")
            documents.append(_parse_document(raw, source, digest, source_path, line_number))
    return documents, source_records


def _parse_document(
    raw: dict[str, Any],
    source: DatasetSource,
    source_sha256: str,
    source_path: Path,
    line_number: int,
) -> InputDocument:
    document_id = raw.get("document_id")
    text = raw.get("text")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(f"document_id must be a non-empty string at {source_path}:{line_number}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text must be a non-empty string at {source_path}:{line_number}")
    row_source_id = raw.get("source_id")
    if row_source_id is not None and row_source_id != source.source_id:
        raise ValueError(f"source_id disagrees with manifest at {source_path}:{line_number}")
    course_id = raw.get("course_id")
    if course_id is not None and (not isinstance(course_id, str) or not course_id.strip()):
        raise ValueError(f"course_id must be a non-empty string at {source_path}:{line_number}")
    supplied_split = raw.get("split")
    if supplied_split is not None and (
        not isinstance(supplied_split, str) or supplied_split not in {"train", "validation", "test"}
    ):
        raise ValueError(f"split must be train, validation, or test at {source_path}:{line_number}")
    language = raw.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError(f"language must be a string at {source_path}:{line_number}")
    metadata = {
        key: raw[key]
        for key in _SAFE_METADATA_FIELDS
        if key in raw and isinstance(raw[key], str | int | float | bool | type(None))
    }
    return InputDocument(
        document_id=document_id.strip(),
        text=text,
        source_id=source.source_id,
        source_revision=source.revision,
        source_sha256=source_sha256,
        license_spdx_id=source.license.spdx_id,
        attribution=source.license.attribution,
        course_id=course_id.strip() if isinstance(course_id, str) else None,
        explicit_split=cast(Split | None, supplied_split),
        language=language,
        metadata=metadata,
    )


def _write_prepared_outputs(
    records: dict[Split, list[dict[str, object]]],
    tokenizer: Encoder,
    output_dir: str | Path,
    *,
    config: PreparationConfig,
    dataset_id: str,
    source_manifest_sha256: str,
    sources: list[dict[str, str]],
    counters: dict[str, int],
    synthetic_fixture: bool,
) -> dict[str, object]:
    destination = Path(output_dir).absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("output_dir must be a regular directory path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for split, items in records.items():
            split_root = staging / split
            split_root.mkdir(parents=True)
            write_jsonl(split_root / "records.jsonl", items)
            if not items:
                continue
            texts = [str(item["text"]) for item in items]
            shard_dir = split_root / "tokens"
            write_aligned_token_shards(
                texts, tokenizer, shard_dir, add_eos=True, synthetic_fixture=synthetic_fixture
            )
            packed_documents = [
                (
                    str(item["segment_id"]),
                    tokenizer.encode(str(item["text"]), add_eos=True).input_ids,
                )
                for item in items
            ]
            pad_id = tokenizer.special_token_id("<|pad|>")
            packed = pack_documents(
                packed_documents,
                sequence_length=config.sequence_length,
                pad_token_id=pad_id,
            )
            packed_dir = split_root / "packed"
            packed_dir.mkdir()
            write_memmap_shard(packed, packed_dir)

        summary: dict[str, object] = {
            "dataset_id": dataset_id,
            "status": "synthetic-test-fixture" if synthetic_fixture else "prepared",
            "synthetic_fixture": synthetic_fixture,
            "source_manifest_sha256": source_manifest_sha256,
            "tokenizer_version": tokenizer.version,
            "tokenizer_sha256": tokenizer.sha256,
            "sequence_length": config.sequence_length,
            "seed": config.seed,
            "sources": sources,
            "counters": counters,
            "split_segments": {split: len(items) for split, items in records.items()},
        }
        (staging / "report.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        dataset_manifest = {
            **summary,
            "schema_version": "1.0",
            "preparation_config": config.model_dump(mode="json"),
            "files": files,
        }
        (staging / "prepared_manifest.json").write_text(
            json.dumps(dataset_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return {**dataset_manifest, "output_dir": str(destination)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _deterministic_split(
    group_id: str, seed: int, proportions: tuple[float, float, float]
) -> Split:
    import hashlib

    value = (
        int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big") / 2**64
    )
    if value < proportions[0]:
        return "train"
    if value < proportions[0] + proportions[1]:
        return "validation"
    return "test"


def _text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _language_hint(text: str) -> str:
    import unicodedata

    latin = arabic = 0
    for char in text:
        name = unicodedata.name(char, "")
        latin += int("LATIN" in name)
        arabic += int("ARABIC" in name or "PERSIAN" in name)
    if latin and arabic:
        return "mixed-en-ur"
    if arabic:
        return "ur-script"
    if latin:
        return "en-script"
    return "unknown"


__all__ = [
    "InputDocument",
    "PreparationConfig",
    "prepare_dataset",
    "prepare_records",
    "verify_prepared_dataset",
]
