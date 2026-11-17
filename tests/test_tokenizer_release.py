"""Tokenizer recipe and release safety checks using explicit local fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion_core.errors import TokenizerMismatchError
from orion_tokenizer.approval import require_tokenizer_owner_approval
from orion_tokenizer.cli import main
from orion_tokenizer.release import export_tokenizer_release, verify_tokenizer_release
from orion_tokenizer.spec import special_token_map_sha256
from orion_tokenizer.training import (
    TokenizerReleaseConfig,
    TokenizerTrainingConfig,
    train_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = PROJECT_ROOT / "configs" / "tokenizer" / "byte-bpe-v1.json"


def test_release_recipe_is_valid_and_approved_but_does_not_claim_training() -> None:
    recipe = TokenizerReleaseConfig.model_validate(
        json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    )

    assert recipe.vocab_size == 32768
    assert recipe.status == "pending"
    assert recipe.production_tokenizer_trained is False
    assert recipe.special_token_id_table_reviewed is True
    with pytest.raises(ValueError, match="complete owner review"):
        recipe.training_config(approve_token_map=False)


def test_tokenizer_training_recipe_requires_two_review_assertions() -> None:
    recipe = TokenizerReleaseConfig.model_validate(
        {
            **json.loads(RECIPE_PATH.read_text(encoding="utf-8")),
            "special_token_id_table_reviewed": True,
        }
    )
    with pytest.raises(ValueError, match="complete owner review"):
        recipe.training_config(approve_token_map=False)
    config = recipe.training_config(approve_token_map=True)
    assert config == TokenizerTrainingConfig(
        vocab_size=32768,
        tokenizer_version="1.0",
        min_frequency=2,
        seed=1729,
        status="trained",
        token_map_approved=True,
        approved_token_map_sha256=special_token_map_sha256(),
    )
    with pytest.raises(ValueError, match="reviewed token-map digest"):
        TokenizerTrainingConfig(
            vocab_size=32768,
            token_map_approved=True,
            approved_token_map_sha256="0" * 64,
        ).validate()


def test_tokenizer_cli_validates_untrained_recipe_without_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(["validate-config", "--config", str(RECIPE_PATH)])
    result = json.loads(capsys.readouterr().out)

    assert status == 0
    assert result["valid"] is True
    assert result["owner_reviewed"] is True
    assert result["production_tokenizer_trained"] is False
    assert result["training_started"] is False


def test_tokenizer_training_requires_matching_owner_document_and_machine_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "approved-project"
    status_path = root / "packages/orion-tokenizer/releases/v1.0/STATUS.json"
    status_path.parent.mkdir(parents=True)
    status = json.loads(
        (PROJECT_ROOT / "packages/orion-tokenizer/releases/v1.0/STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    status["owner_approval"] = "approved"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    approval_path = root / "docs/tokenizer/tokenizer_approval.md"
    approval_path.parent.mkdir(parents=True)
    digest = special_token_map_sha256()
    approval_path.write_text(
        "| Field | Value |\n|---|---|\n"
        "| Decision | APPROVED |\n"
        "| Owner name or authorized role | Project owner |\n"
        "| Decision date (ISO 8601) | 2026-11-14 |\n"
        f"| Approved special-token map SHA-256 | {digest} |\n"
        "| Approved byte mapping and normalization | reviewed |\n"
        "| Approved generation-emittable controls | reviewed |\n"
        "| Notes or required changes | No changes |\n",
        encoding="utf-8",
    )

    verified_status = require_tokenizer_owner_approval(root, digest)
    assert verified_status["owner_approval"] == "approved"
    with pytest.raises(ValueError, match="differs from the current v1 registry"):
        require_tokenizer_owner_approval(root, "0" * 64)


def test_pending_owner_approval_blocks_tokenizer_cli_before_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe = TokenizerReleaseConfig.model_validate(
        {
            **json.loads(RECIPE_PATH.read_text(encoding="utf-8")),
            "special_token_id_table_reviewed": True,
        }
    )
    config_path = tmp_path / "reviewed-recipe.json"
    config_path.write_text(recipe.model_dump_json(), encoding="utf-8")
    output = tmp_path / "must-not-train"
    project_root = tmp_path / "pending-project"
    status_path = project_root / "packages/orion-tokenizer/releases/v1.0/STATUS.json"
    status_path.parent.mkdir(parents=True)
    status = json.loads(
        (PROJECT_ROOT / "packages/orion-tokenizer/releases/v1.0/STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    status["owner_approval"] = "pending"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    status = main(
        [
            "train",
            "--corpus",
            str(tmp_path / "not-read.txt"),
            "--output",
            str(output),
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
            "--confirm-training",
            "--approve-token-map",
        ]
    )
    error = json.loads(capsys.readouterr().err)

    assert status == 1
    assert "owner approval is pending" in error["error"]
    assert error["training_started"] is False
    assert not output.exists()


def test_tiny_synthetic_tokenizer_training_writes_atomically_and_labels_fixture(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "synthetic.txt"
    corpus.write_text("synthetic byte fixture\n", encoding="utf-8")
    output = tmp_path / "tokenizer-fixture"

    result = train_tokenizer(
        [corpus],
        output,
        config=TokenizerTrainingConfig(vocab_size=512, status="synthetic-test-fixture"),
    )

    manifest = json.loads((result / "vocab_manifest.json").read_text(encoding="utf-8"))
    assert result == output
    assert manifest["status"] == "synthetic-test-fixture"
    assert manifest["vocab_size"] == 512
    assert set(path.name for path in output.iterdir()) == {
        "tokenizer.json",
        "vocab_manifest.json",
        "tokenizer_config.json",
        "special_tokens.json",
        "golden_encodings.jsonl",
    }
    assert not list(tmp_path.glob(".tokenizer-fixture.*"))
    with pytest.raises(TokenizerMismatchError, match="does not identify a trained"):
        verify_tokenizer_release(output)
    export_path = tmp_path / "should-not-export"
    with pytest.raises(TokenizerMismatchError, match="does not identify a trained"):
        export_tokenizer_release(output, export_path)
    assert not export_path.exists()


def test_failed_tokenizer_training_leaves_no_partial_release(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text("{not-json}\n", encoding="utf-8")
    output = tmp_path / "failed-release"

    with pytest.raises(ValueError, match="Invalid JSONL"):
        train_tokenizer(
            [corpus],
            output,
            config=TokenizerTrainingConfig(vocab_size=512, status="synthetic-test-fixture"),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-release.*"))
