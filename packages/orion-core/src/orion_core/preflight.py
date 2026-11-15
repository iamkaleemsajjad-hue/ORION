"""Read-only checks for the ORION software environment and training inputs."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

_PACKAGES: tuple[tuple[str, str], ...] = (
    ("orion-core", "orion_core"),
    ("orion-tokenizer", "orion_tokenizer"),
    ("orion-data", "orion_data"),
    ("orion-train", "orion_train"),
    ("orion-eval", "orion_eval"),
    ("orion-100m", "orion_100m"),
    ("orion-300m", "orion_300m"),
    ("orion-madad-adapter", "orion_madad_adapter"),
)
_CONFIGS: tuple[tuple[str, str, str, str], ...] = (
    (
        "100m-rtd",
        "configs/pretraining/orion-100m-rtd.json",
        "orion_100m.training",
        "RTDTrainingConfig",
    ),
    (
        "300m-causal",
        "configs/pretraining/orion-300m-causal.json",
        "orion_300m.training",
        "CausalTrainingConfig",
    ),
    (
        "100m-finetuning",
        "configs/finetuning/orion-100m-multitask.json",
        "orion_100m.finetuning",
        "FineTuningConfig",
    ),
    (
        "100m-contrastive",
        "configs/finetuning/orion-100m-contrastive.json",
        "orion_100m.contrastive",
        "ContrastiveTrainingConfig",
    ),
    (
        "100m-course-matching",
        "configs/finetuning/orion-100m-course-matching.json",
        "orion_100m.course_matching",
        "CourseMatchingTrainingConfig",
    ),
    ("300m-sft", "configs/finetuning/orion-300m-sft.json", "orion_300m.sft", "SFTTrainingConfig"),
    (
        "300m-architecture-pilot",
        "configs/evaluation/e-300-1-architecture-pilot.json",
        "orion_300m.architecture_pilot",
        "ArchitecturePilotConfig",
    ),
    (
        "data-preparation",
        "configs/data/orion-data-preparation.json",
        "orion_data.prepare",
        "PreparationConfig",
    ),
    (
        "tokenizer-training",
        "configs/tokenizer/byte-bpe-v1.json",
        "orion_tokenizer.training",
        "TokenizerReleaseConfig",
    ),
)
_TRAINING_ONLY_CHECKS = {
    "tokenizer-owner-approval",
    "prepared-dataset",
    "prepared-tokenizer-match",
    "training-token-count",
    "whole-word-alignment",
    "trained-tokenizer",
    "training-device",
    "training-precision-device",
}


def run_preflight(
    project_root: str | Path,
    *,
    mode: Literal["source", "installed"] = "source",
    model: Literal["100m", "300m"] | None = None,
    dataset_dir: str | Path | None = None,
    tokenizer_dir: str | Path | None = None,
    device: str = "auto",
    precision: Literal["fp32", "bf16", "fp16"] | None = None,
) -> dict[str, Any]:
    """Check imports, dependencies, project configs and optional local data.

    This function is read-only and never starts training, downloads data, or
    attempts to resolve artifacts over the network.
    """
    root = Path(project_root).resolve(strict=True)
    checks: list[dict[str, object]] = []
    blockers: list[str] = []
    notes: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            blockers.append(f"{name}: {detail}")

    record("project-root", root.is_dir() and (root / "pyproject.toml").is_file(), str(root))
    record("python-version", sys.version_info >= (3, 10), sys.version.split()[0])

    source_dirs: list[Path] = []
    if mode == "source":
        source_dirs = sorted((root / "packages").glob("*/src"))
        missing = [name for name, _ in _PACKAGES if not (root / "packages" / name / "src").is_dir()]
        if missing:
            record(
                "workspace-source",
                False,
                "missing package source directories: " + ", ".join(missing),
            )
        else:
            record("workspace-source", True, "checking the eight package source trees")

    if mode == "installed":
        for distribution, module in _PACKAGES:
            try:
                version = importlib.metadata.version(distribution)
                importlib.import_module(module)
            except importlib.metadata.PackageNotFoundError:
                record(f"distribution:{distribution}", False, "not installed")
            except Exception as exc:
                record(
                    f"distribution:{distribution}",
                    False,
                    f"import failed: {type(exc).__name__}: {exc}",
                )
            else:
                record(f"distribution:{distribution}", version == "0.1.0", f"version={version}")
    else:
        with _source_import_path(source_dirs):
            for distribution, module in _PACKAGES:
                try:
                    imported = importlib.import_module(module)
                except Exception as exc:
                    record(f"import:{module}", False, f"{type(exc).__name__}: {exc}")
                else:
                    module_file = getattr(imported, "__file__", None)
                    package_source = (root / "packages" / distribution / "src").resolve()
                    from_checkout = False
                    if isinstance(module_file, str):
                        from_checkout = Path(module_file).resolve().is_relative_to(package_source)
                    record(
                        f"import:{module}",
                        from_checkout,
                        "imported from the local source checkout"
                        if from_checkout
                        else f"resolved outside expected source tree {package_source}",
                    )

    try:
        import tokenizers
        import torch
    except ImportError as exc:
        record("runtime-dependencies", False, f"required import failed: {exc}")
    else:
        torch_version = _version_tuple(torch.__version__)
        tokenizer_version = _version_tuple(tokenizers.__version__)
        record(
            "pytorch-version",
            torch_version is not None and (2, 1) <= torch_version < (3, 0),
            torch.__version__,
        )
        tokenizers_ok = tokenizer_version is not None and (0, 23, 3) <= tokenizer_version < (
            0,
            24,
            0,
        )
        record("tokenizers-version", tokenizers_ok, tokenizers.__version__)
        cuda = torch.cuda.is_available()
        record(
            "cuda-capability",
            True,
            "CUDA available" if cuda else "CUDA unavailable; CPU software checks remain valid",
        )
        if not cuda:
            notes.append(
                "CUDA is unavailable; this preflight can only verify CPU-side software behavior"
            )

    with _source_import_path(source_dirs if mode == "source" else []):
        _validate_configs(root, record)
    status_path = root / "packages/orion-tokenizer/releases/v1.0/STATUS.json"
    try:
        release = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record("tokenizer-release-status", False, f"{type(exc).__name__}: {exc}")
        release = {}
        tokenizer_approved = False
    else:
        tokenizer_approved = release.get("owner_approval") == "approved"
        trained = _release_is_trained(release)
        record(
            "tokenizer-owner-approval",
            tokenizer_approved,
            "approved"
            if tokenizer_approved
            else "pending; review docs/tokenizer/tokenizer_approval.md",
        )
        if not trained:
            notes.append("production tokenizer v1 is not marked trained and available")

    training_checks_start = len(checks)
    if model is not None:
        try:
            from orion_core.device import autocast_dtype, select_device

            selected = select_device(device)
            if precision is not None:
                autocast_dtype(precision, selected)
        except (RuntimeError, ValueError) as exc:
            record("training-device", False, str(exc))
        else:
            record("training-device", True, str(selected))
            if precision is not None:
                record(
                    "training-precision-device", True, f"precision={precision}, device={selected}"
                )

        prepared: dict[str, object] | None = None
        if dataset_dir is None:
            record(
                "prepared-dataset",
                False,
                "pass --dataset-dir with a verified local prepared dataset",
            )
        else:
            try:
                from orion_data.prepare import verify_prepared_dataset

                candidate = verify_prepared_dataset(dataset_dir)
            except (OSError, ValueError) as exc:
                record("prepared-dataset", False, str(exc))
            else:
                usable = (
                    candidate.get("verified") is True
                    and candidate.get("synthetic_fixture") is False
                )
                record(
                    "prepared-dataset",
                    usable,
                    "verified production dataset" if usable else "synthetic or invalid dataset",
                )
                if usable:
                    prepared = candidate

        if tokenizer_dir is None:
            record(
                "trained-tokenizer",
                False,
                "pass --tokenizer-dir with an owner-approved trained tokenizer",
            )
        else:
            try:
                from orion_tokenizer.tokenizer import OrionTokenizer

                tokenizer = OrionTokenizer.load(tokenizer_dir)
            except Exception as exc:
                record("trained-tokenizer", False, f"{type(exc).__name__}: {exc}")
            else:
                tokenizer_ok = (
                    tokenizer.vocab_size == 32768
                    and tokenizer_approved
                    and _release_is_trained(release)
                )
                record(
                    "trained-tokenizer",
                    tokenizer_ok,
                    f"version={tokenizer.version}, vocab_size={tokenizer.vocab_size}",
                )
                if prepared is not None:
                    matches = prepared.get("tokenizer_sha256") == tokenizer.sha256
                    record(
                        "prepared-tokenizer-match",
                        matches,
                        "prepared data uses the supplied tokenizer"
                        if matches
                        else "tokenizer hash differs from prepared data",
                    )

        if prepared is not None and dataset_dir is not None:
            token_path = Path(dataset_dir) / "train/tokens/tokens.u16"
            word_path = Path(dataset_dir) / "train/tokens/word_ids.i32"
            token_count = token_path.stat().st_size // 2 if token_path.is_file() else 0
            required = 2048 if model == "100m" else 4096
            record(
                "training-token-count",
                token_count >= required,
                f"{token_count} tokens; one {required}-token block required",
            )
            if model == "100m":
                aligned = word_path.is_file() and word_path.stat().st_size == token_count * 4
                record(
                    "whole-word-alignment",
                    aligned,
                    "aligned int32 word IDs" if aligned else "missing or misaligned word_ids.i32",
                )

    # Preserve only software checks in the software readiness result. Missing
    # owner-approved training assets do not indicate a broken installation.
    training_checks = checks[training_checks_start:]
    software_checks = [check for check in checks if check["name"] not in _TRAINING_ONLY_CHECKS]
    software_ready = all(check["passed"] for check in software_checks)
    training_ready = (
        model is not None
        and software_ready
        and all(check["passed"] for check in training_checks)
        and tokenizer_approved
    )
    if model is None:
        notes.append(
            "software readiness does not imply that tokenizer, data, or model training prerequisites are ready"
        )
    elif not training_ready:
        notes.append(f"ORION-{model} training is blocked by one or more preflight checks")
    blockers = [
        f"{check['name']}: {check['detail']}"
        for check in checks
        if not check["passed"] and (model is not None or check["name"] not in _TRAINING_ONLY_CHECKS)
    ]
    return {
        "status": "ready" if software_ready and (model is None or training_ready) else "blocked",
        "mode": mode,
        "model": model,
        "software_ready": software_ready,
        "training_ready": training_ready,
        "training_started": False,
        "checks": checks,
        "blockers": blockers,
        "notes": notes,
    }


def _version_tuple(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def _release_is_trained(release: dict[str, Any]) -> bool:
    return release.get("status") == "trained" and release.get("trained_tokenizer_present") is True


@contextmanager
def _source_import_path(paths: list[Path]) -> Iterator[None]:
    original = sys.path.copy()
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path[:] = original


def _validate_configs(root: Path, record: Any) -> None:
    for name, relative, module_name, class_name in _CONFIGS:
        try:
            raw = json.loads((root / relative).read_text(encoding="utf-8"))
            module = importlib.import_module(module_name)
            schema = getattr(module, class_name)
            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                raise TypeError(f"{module_name}.{class_name} is not a Pydantic model")
            schema.model_validate(raw)
        except (
            OSError,
            json.JSONDecodeError,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            record(f"config:{name}", False, f"{type(exc).__name__}: {exc}")
        else:
            record(f"config:{name}", True, "validated")

    for name, relative in (
        ("tokenizer", "configs/tokenizer/byte-bpe-v1.json"),
        ("evaluation", "configs/evaluation/default.json"),
        ("deployment", "configs/deployment/local-api.json"),
    ):
        try:
            raw = json.loads((root / relative).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be a JSON object")
            if name == "tokenizer":
                expected = {
                    "vocab_size": 32768,
                    "byte_alphabet_size": 256,
                    "special_token_count": 256,
                    "learned_merge_count": 32256,
                }
                if any(raw.get(key) != value for key, value in expected.items()):
                    raise ValueError("tokenizer vocabulary allocation is inconsistent")
            elif name == "evaluation":
                if (
                    not isinstance(raw.get("bootstrap_resamples"), int)
                    or raw["bootstrap_resamples"] <= 0
                ):
                    raise ValueError("bootstrap_resamples must be a positive integer")
                if not isinstance(raw.get("report_metrics"), list) or not raw["report_metrics"]:
                    raise ValueError("report_metrics must be a non-empty list")
            elif raw.get("host") != "127.0.0.1" or raw.get("telemetry") is not False:
                raise ValueError("local deployment must bind loopback with telemetry disabled")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            record(f"config:{name}", False, f"{type(exc).__name__}: {exc}")
        else:
            record(f"config:{name}", True, "validated")


__all__ = ["run_preflight"]
