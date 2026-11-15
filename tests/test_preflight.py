"""Behavior tests for the source-checkout preflight and CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orion_core.cli import main
from orion_core.preflight import run_preflight

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_preflight_validates_configs_without_persisting_sys_path() -> None:
    path_before = sys.path.copy()

    result = run_preflight(PROJECT_ROOT)

    assert sys.path == path_before
    assert result["training_started"] is False
    assert result["training_ready"] is False
    assert any(check["name"] == "cuda-capability" and check["passed"] for check in result["checks"])
    assert all(
        check["passed"] for check in result["checks"] if str(check["name"]).startswith("config:")
    )
    assert all("tokenizer-owner-approval" not in blocker for blocker in result["blockers"])


def test_training_preflight_reports_pending_prerequisites_without_starting() -> None:
    result = run_preflight(PROJECT_ROOT, model="100m")

    checks = {str(check["name"]): check for check in result["checks"]}
    assert result["training_started"] is False
    assert result["training_ready"] is False
    assert checks["tokenizer-owner-approval"]["passed"] is False
    assert checks["prepared-dataset"]["passed"] is False
    assert checks["trained-tokenizer"]["passed"] is False
    assert "tokenizer-owner-approval" in " ".join(result["blockers"])


def test_preflight_cli_emits_machine_readable_result(capsys: object) -> None:
    status = main(["preflight", "--root", str(PROJECT_ROOT)])

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert status == (0 if payload["status"] == "ready" else 1)
    assert payload["training_started"] is False
    assert payload["mode"] == "source"


def test_preflight_cli_reports_invalid_root_as_json_error(capsys: object, tmp_path: Path) -> None:
    status = main(["preflight", "--root", str(tmp_path / "missing")])

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err)
    assert status == 2
    assert payload["status"] == "error"
    assert payload["training_started"] is False
