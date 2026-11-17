"""Behavior tests for the source-checkout preflight and CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orion_core.cli import main
from orion_core.doctor import run_doctor
from orion_core.preflight import run_preflight
from orion_core.resource_estimates import estimate_adamw_memory_floor

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
    result = run_preflight(PROJECT_ROOT, model="100m", device="cpu")

    checks = {str(check["name"]): check for check in result["checks"]}
    assert result["training_started"] is False
    assert result["training_ready"] is False
    assert checks["tokenizer-owner-approval"]["passed"] is True
    assert checks["prepared-dataset"]["passed"] is False
    assert checks["trained-tokenizer"]["passed"] is False
    assert checks["tokenizer-owner-approval"]["status"] == "PASS"
    assert checks["training-device"]["status"] == "EXTERNAL HARDWARE REQUIRED"
    assert checks["prepared-dataset"]["status"] == "OWNER APPROVAL REQUIRED"
    assert checks["target-accelerator"]["status"] == "WARNING"
    assert checks["training-memory-estimate"]["status"] == "WARNING"
    assert checks["training-memory-estimate"]["passed"] is True
    assert "not a peak-memory or fit guarantee" in checks["training-memory-estimate"]["detail"]
    assert "prepared-dataset" in " ".join(result["blockers"])


def test_adamw_memory_floor_counts_fp32_parameters_gradients_and_two_moments() -> None:
    encoder = estimate_adamw_memory_floor(100_291_840)
    decoder = estimate_adamw_memory_floor(306_763_200)

    assert encoder.parameter_bytes == 401_167_360
    assert encoder.gradient_bytes == 401_167_360
    assert encoder.optimizer_state_bytes == 802_334_720
    assert encoder.total_bytes == 1_604_669_440
    assert decoder.total_bytes == 4_908_211_200
    assert encoder.gib < decoder.gib


def test_adamw_memory_floor_rejects_invalid_counts_and_element_sizes() -> None:
    import pytest

    with pytest.raises(ValueError, match="parameter_count"):
        estimate_adamw_memory_floor(0)
    with pytest.raises(ValueError, match="element sizes"):
        estimate_adamw_memory_floor(10, gradient_bytes_per_element=0)


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


def test_doctor_is_read_only_and_never_marks_unverified_checks_as_pass(
    tmp_path: Path,
) -> None:
    result = run_doctor(tmp_path)

    assert result["status"] == "blocked"
    assert result["training_started"] is False
    assert result["network_access_attempted"] is False
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["workspace-root"]["status"] == "FAIL"
    assert checks["package-index-connectivity"]["status"] == "NOT TESTED"
    assert checks["gloo-backend"]["status"] in {"NOT TESTED", "FAIL"}
    assert checks["nccl-backend"]["status"] in {"NOT TESTED", "EXTERNAL HARDWARE REQUIRED"}


def test_doctor_cli_renders_human_readable_status(capsys: object, tmp_path: Path) -> None:
    status = main(["doctor", "--root", str(tmp_path), "--format", "text"])

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert status == 1
    assert captured.out.startswith("ORION doctor: blocked\n")
    assert "[FAIL] workspace-root:" in captured.out
    assert "No training started; no network request was made." in captured.out
