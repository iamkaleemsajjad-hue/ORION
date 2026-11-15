from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import pytest

from orion_core.hashing import sha256_file
from orion_data.cli import build_parser
from orion_data.manifest import DatasetManifest, DatasetSource, LicenseRecord
from orion_data.prepare import (
    InputDocument,
    PreparationConfig,
    _load_source_documents,
    _write_prepared_outputs,
    prepare_dataset,
    prepare_records,
    verify_prepared_dataset,
)
from tests.test_tokenizer import synthetic_tokenizer


def _document(
    document_id: str,
    text: str,
    *,
    course_id: str,
    split: Literal["train", "validation", "test"] | None = None,
) -> InputDocument:
    return InputDocument(
        document_id=document_id,
        text=text,
        source_id="local-fixture",
        source_revision="fixture-1",
        source_sha256="a" * 64,
        license_spdx_id="CC0-1.0",
        attribution="Synthetic test fixture only",
        course_id=course_id,
        explicit_split=split,
    )


def test_preparation_config_validates_split_and_dedup_dimensions() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        PreparationConfig(split_proportions=(0.8, 0.1, 0.2))
    with pytest.raises(ValueError, match="divisible"):
        PreparationConfig(near_duplicate_permutations=60, near_duplicate_bands=16)


def test_manifest_license_policy_uses_the_reviewed_allowlist() -> None:
    cc0 = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(
            DatasetSource(
                source_id="cc0-source",
                kind="local-jsonl",
                revision="fixture-v1",
                license=LicenseRecord(spdx_id="cc0-1.0", verified=True),
            ),
        ),
    )
    cc0.validate_license_policy("commercial-clean", purpose="training")

    noncommercial = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(
            DatasetSource(
                source_id="evaluation-only",
                kind="local-jsonl",
                revision="fixture-v1",
                license=LicenseRecord(
                    spdx_id="CC-BY-NC-3.0",
                    attribution="Synthetic evaluation fixture",
                    verified=True,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="restricted to evaluation use"):
        noncommercial.validate_license_policy("research", purpose="training")
    noncommercial.validate_license_policy("research", purpose="evaluation")


def test_manifest_requires_attribution_when_license_policy_requires_it() -> None:
    missing_attribution = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(
            DatasetSource(
                source_id="mit-source",
                kind="local-jsonl",
                revision="fixture-v1",
                license=LicenseRecord(spdx_id="MIT", verified=True),
            ),
        ),
    )
    with pytest.raises(ValueError, match="requires an attribution record"):
        missing_attribution.validate_license_policy("research", purpose="training")

    attributed = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(
            DatasetSource(
                source_id="mit-source",
                kind="local-jsonl",
                revision="fixture-v1",
                license=LicenseRecord(
                    spdx_id="MIT",
                    attribution="Copyright Example Org; MIT; https://example.invalid/source",
                    verified=True,
                ),
            ),
        ),
    )
    attributed.validate_license_policy("commercial-clean", purpose="training")


def test_local_source_loader_checks_manifest_hash_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "records.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "document_id": "local-doc-1",
                "course_id": "BIO-101",
                "text": "A local biology document with enough words for a preparation fixture.",
                "title": "Cell biology",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = DatasetSource(
        source_id="local-fixture",
        kind="local-jsonl",
        revision="fixture-v1",
        local_path=source_path.name,
        sha256=sha256_file(source_path),
        license=LicenseRecord(spdx_id="CC0-1.0", attribution="Synthetic fixture", verified=True),
    )
    manifest = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(source,),
    )

    documents, source_records = _load_source_documents(tmp_path / "manifest.json", manifest)

    assert documents[0].document_id == "local-doc-1"
    assert documents[0].course_id == "BIO-101"
    assert documents[0].source_sha256 == sha256_file(source_path)
    assert documents[0].metadata == {"title": "Cell biology"}
    assert source_records[0]["revision"] == "fixture-v1"
    source_path.write_text(source_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_source_documents(tmp_path / "manifest.json", manifest)


def test_local_preparation_rejects_non_local_source_kinds(tmp_path: Path) -> None:
    source = DatasetSource(
        source_id="remote-source",
        kind="huggingface-dataset",
        revision="a" * 40,
        local_path="records.jsonl",
        sha256="a" * 64,
        license=LicenseRecord(spdx_id="CC0-1.0", verified=True),
    )
    manifest = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-11-14",
        sources=(source,),
    )
    with pytest.raises(ValueError, match="only local-jsonl sources"):
        _load_source_documents(tmp_path / "manifest.json", manifest)


def test_prepare_records_masks_pii_and_assigns_whole_courses() -> None:
    config = PreparationConfig(
        split_proportions=(1.0, 0.0, 0.0),
        min_characters=20,
        max_segment_characters=500,
        near_duplicate_threshold=1.0,
    )
    documents = [
        _document(
            "doc-a",
            "A linear algebra lesson explains matrices, vectors, rank, determinants, and useful examples. "
            * 2,
            course_id="MATH-101",
        ),
        _document(
            "doc-b",
            "A second lesson for the same course studies eigenvalues, diagonalization, basis changes, "
            "and numerical examples. " * 3,
            course_id="MATH-101",
        ),
        _document(
            "doc-c",
            "A separate chemistry document explains molecular structures, reaction balance, and "
            "laboratory safety. Contact teacher@example.edu for the office schedule.",
            course_id="CHEM-202",
        ),
    ]

    records, counters = prepare_records(documents, config)

    assert counters["input_documents"] == 3
    assert counters["segments_kept"] == 3
    assert all(item["split"] == "train" for item in records["train"])
    chemistry = next(item for item in records["train"] if item["document_id"] == "doc-c")
    assert "teacher@example.edu" not in cast(str, chemistry["text"])
    pii_counts = cast(dict[str, int], chemistry["pii_counts"])
    assert pii_counts["email"] == 1


def test_prepare_records_removes_heldout_duplicates_and_rejects_group_leakage() -> None:
    text = "A detailed course handout explains probability distributions and sampling theory. " * 3
    config = PreparationConfig(min_characters=20, near_duplicate_threshold=1.0)
    records, counters = prepare_records(
        [
            _document("copy-train", text, course_id="COURSE-1", split="train"),
            _document("copy-test", text, course_id="COURSE-2", split="test"),
        ],
        config,
    )
    assert not records["train"]
    assert [item["document_id"] for item in records["test"]] == ["copy-test"]
    assert counters["cross_split_duplicates_removed"] == 1

    with pytest.raises(ValueError, match="crosses data splits"):
        prepare_records(
            [
                _document("one", text, course_id="SHARED", split="train"),
                _document("two", text + " Additional material.", course_id="SHARED", split="test"),
            ],
            config,
        )


def test_prepare_cli_writes_verifiable_synthetic_fixture_shards(tmp_path: Path) -> None:
    records, counters = prepare_records(
        [
            _document(
                "doc-a",
                "A synthetic test document describes tokenization and deterministic shard packing. "
                * 3,
                course_id="FIXTURE-1",
            )
        ],
        PreparationConfig(
            split_proportions=(1.0, 0.0, 0.0),
            min_characters=20,
            sequence_length=16,
        ),
    )
    result = _write_prepared_outputs(
        records,
        synthetic_tokenizer(),
        tmp_path / "prepared",
        config=PreparationConfig(
            split_proportions=(1.0, 0.0, 0.0),
            min_characters=20,
            sequence_length=16,
        ),
        dataset_id="test-only-fixture",
        source_manifest_sha256="b" * 64,
        sources=[{"source_id": "local-fixture", "sha256": "a" * 64}],
        counters=counters,
        synthetic_fixture=True,
    )

    verified = verify_prepared_dataset(tmp_path / "prepared")
    assert verified["verified"] is True
    assert verified["synthetic_fixture"] is True
    assert result["status"] == "synthetic-test-fixture"
    assert (tmp_path / "prepared" / "train" / "packed" / "index.json").is_file()

    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "manifest.json",
            "--tokenizer",
            "tokenizer",
            "--output",
            "out",
        ]
    )
    assert args.command == "prepare"


def test_production_prepare_refuses_synthetic_manifest_before_artifacts(tmp_path: Path) -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1] / "datasets" / "manifests" / "synthetic-example.json"
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="synthetic fixture manifests"):
        prepare_dataset(manifest_path, tmp_path / "no-tokenizer", output)
    assert not output.exists()


def test_prepared_verifier_detects_file_tampering(tmp_path: Path) -> None:
    records, counters = prepare_records(
        [
            _document(
                "doc-a",
                "A synthetic fixture paragraph is long enough to exercise the prepared data writer. "
                * 3,
                course_id="FIXTURE-2",
            )
        ],
        PreparationConfig(split_proportions=(1.0, 0.0, 0.0), min_characters=20),
    )
    root = tmp_path / "prepared"
    _write_prepared_outputs(
        records,
        synthetic_tokenizer(),
        root,
        config=PreparationConfig(split_proportions=(1.0, 0.0, 0.0), min_characters=20),
        dataset_id="fixture",
        source_manifest_sha256="c" * 64,
        sources=[{"source_id": "local-fixture", "sha256": "a" * 64}],
        counters=counters,
        synthetic_fixture=True,
    )
    target = root / "train" / "records.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_prepared_dataset(root)


def test_prepared_verifier_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "prepared_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "synthetic-test-fixture",
                "synthetic_fixture": True,
                "files": {"../outside.json": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe path"):
        verify_prepared_dataset(root)


def test_prepared_report_is_json_and_keeps_synthetic_label(tmp_path: Path) -> None:
    records, counters = prepare_records(
        [
            _document(
                "doc",
                "Long enough synthetic educational text for a local test. " * 3,
                course_id="X",
            )
        ],
        PreparationConfig(split_proportions=(1.0, 0.0, 0.0), min_characters=20),
    )
    root = tmp_path / "prepared"
    _write_prepared_outputs(
        records,
        synthetic_tokenizer(),
        root,
        config=PreparationConfig(split_proportions=(1.0, 0.0, 0.0), min_characters=20),
        dataset_id="fixture",
        source_manifest_sha256="d" * 64,
        sources=[{"source_id": "fixture", "sha256": "a" * 64}],
        counters=counters,
        synthetic_fixture=True,
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "synthetic-test-fixture"
    assert report["synthetic_fixture"] is True
