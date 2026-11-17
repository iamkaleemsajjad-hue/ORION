"""Read-only diagnostics for an installed ORION environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from orion_core.errors import ConfigurationError

Status = Literal[
    "PASS",
    "FAIL",
    "WARNING",
    "NOT TESTED",
    "OWNER APPROVAL REQUIRED",
    "EXTERNAL HARDWARE REQUIRED",
]

_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("orion-core", "orion_core"),
    ("orion-tokenizer", "orion_tokenizer"),
    ("orion-data", "orion_data"),
    ("orion-train", "orion_train"),
    ("orion-eval", "orion_eval"),
    ("orion-100m", "orion_100m"),
    ("orion-300m", "orion_300m"),
    ("orion-madad-adapter", "orion_madad_adapter"),
)


def _version_tuple(value: str) -> tuple[int, ...] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    return tuple(int(part) for part in match.groups(default="0")) if match else None


def _available_memory_bytes() -> int | None:
    """Return currently available physical memory using platform APIs only."""
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        # ``windll`` is exposed only by Windows' ctypes runtime.
        kernel32 = getattr(ctypes, "windll")  # noqa: B009
        kernel32 = kernel32.kernel32
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)

    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return None
    try:
        pages = sysconf("SC_AVPHYS_PAGES")
        page_size = sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return int(pages * page_size) if pages > 0 and page_size > 0 else None


def run_doctor(project_root: str | Path | None = None) -> dict[str, Any]:
    """Inspect the local environment without downloading or changing anything.

    A ``PASS`` confirms only the stated local check. Runtime backend availability
    is kept distinct from successful distributed or target-hardware execution.
    """
    root = Path.cwd() if project_root is None else Path(project_root)
    root = root.expanduser().resolve()
    checks: list[dict[str, str]] = []

    def record(name: str, status: Status, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    root_valid = root.is_dir() and (root / "pyproject.toml").is_file()
    record(
        "workspace-root",
        "PASS" if root_valid else "FAIL",
        str(root) if root_valid else "not an ORION workspace root",
    )
    record(
        "python-version",
        "PASS" if sys.version_info >= (3, 10) else "FAIL",
        f"{platform.python_version()} ({platform.python_implementation()})",
    )
    record("platform", "PASS", f"{platform.system()} {platform.machine()}")
    record("cpu-count", "PASS", f"{os.cpu_count() or 0} logical CPUs detected")

    available_memory = _available_memory_bytes()
    record(
        "host-memory",
        "PASS" if available_memory is not None else "NOT TESTED",
        (
            f"{available_memory / (1024**3):.2f} GiB currently available; this is not a "
            "model training memory estimate"
            if available_memory is not None
            else "platform API did not expose available physical memory"
        ),
    )
    try:
        disk_free = shutil.disk_usage(root if root.exists() else Path.cwd()).free
    except OSError as exc:
        record("workspace-disk", "NOT TESTED", f"{type(exc).__name__}: {exc}")
    else:
        record("workspace-disk", "PASS", f"{disk_free / (1024**3):.2f} GiB free")

    torch_module: Any | None = None
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        record("pytorch", "FAIL", f"import failed: {exc}")
    else:
        version = str(torch_module.__version__)
        parsed = _version_tuple(version)
        compatible = parsed is not None and (2, 1) <= parsed < (3, 0)
        record("pytorch", "PASS" if compatible else "FAIL", version)
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            devices = int(torch_module.cuda.device_count())
            try:
                free, total = torch_module.cuda.mem_get_info()
            except (RuntimeError, AttributeError) as exc:
                record("cuda-device", "NOT TESTED", f"detected; memory query failed: {exc}")
            else:
                record(
                    "cuda-device",
                    "PASS",
                    f"{devices} device(s); {free / (1024**3):.2f} GiB free of "
                    f"{total / (1024**3):.2f} GiB",
                )
        else:
            record(
                "cuda-device",
                "EXTERNAL HARDWARE REQUIRED",
                "CUDA is unavailable; CPU smoke checks do not establish full-model training capacity",
            )
        distributed = getattr(torch_module, "distributed", None)
        gloo_available = bool(
            distributed and distributed.is_available() and distributed.is_gloo_available()
        )
        record(
            "gloo-backend",
            "NOT TESTED" if gloo_available else "FAIL",
            "backend is compiled in; multiprocess communication was not executed"
            if gloo_available
            else "PyTorch Gloo backend is unavailable",
        )
        nccl_available = bool(
            distributed and distributed.is_available() and distributed.is_nccl_available()
        )
        record(
            "nccl-backend",
            "NOT TESTED" if nccl_available else "EXTERNAL HARDWARE REQUIRED",
            "backend is compiled in; CUDA/NCCL communication was not executed"
            if nccl_available
            else "CUDA/NCCL support is unavailable in this runtime",
        )

    try:
        tokenizer_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        record("tokenizers", "FAIL", "distribution is not installed")
    else:
        parsed = _version_tuple(tokenizer_version)
        compatible = parsed is not None and (0, 23, 3) <= parsed < (0, 24, 0)
        record(
            "tokenizers",
            "PASS" if compatible else "FAIL",
            f"{tokenizer_version}; required >=0.23.3,<0.24",
        )

    status_path = root / "packages/orion-tokenizer/releases/v1.0/STATUS.json"
    if not root_valid:
        record("tokenizer-release-status", "NOT TESTED", "workspace root is unavailable")
        record("tokenizer-owner-approval", "NOT TESTED", "workspace root is unavailable")
        record("trained-tokenizer-artifact", "NOT TESTED", "workspace root is unavailable")
    else:
        try:
            release_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record("tokenizer-release-status", "FAIL", f"{type(exc).__name__}: {exc}")
            release_status = {}
        if not isinstance(release_status, dict):
            record("tokenizer-release-status", "FAIL", "release status must be a JSON object")
            release_status = {}
        elif all(
            key in release_status
            for key in (
                "status",
                "tokenizer_version",
                "vocab_size",
                "trained_tokenizer_present",
                "owner_approval",
                "special_token_map_sha256",
            )
        ):
            record("tokenizer-release-status", "PASS", "required release fields are present")
        else:
            record("tokenizer-release-status", "FAIL", "required release status fields are missing")

        try:
            from orion_tokenizer.spec import special_token_map_sha256
        except ImportError as exc:
            record("tokenizer-owner-approval", "NOT TESTED", f"token registry import failed: {exc}")
        else:
            current_digest = special_token_map_sha256()
            if release_status.get("special_token_map_sha256") != current_digest:
                record(
                    "tokenizer-registry-digest",
                    "FAIL",
                    "release status digest differs from the installed canonical registry",
                )
                record(
                    "tokenizer-owner-approval",
                    "FAIL",
                    "owner approval cannot be trusted until the registry digest matches",
                )
            else:
                record("tokenizer-registry-digest", "PASS", current_digest)
                try:
                    from orion_tokenizer.approval import require_tokenizer_owner_approval

                    require_tokenizer_owner_approval(root, current_digest)
                except ConfigurationError as exc:
                    owner_pending = release_status.get("owner_approval") != "approved"
                    record(
                        "tokenizer-owner-approval",
                        "OWNER APPROVAL REQUIRED" if owner_pending else "FAIL",
                        str(exc),
                    )
                except ImportError as exc:
                    record(
                        "tokenizer-owner-approval",
                        "NOT TESTED",
                        f"approval checker import failed: {exc}",
                    )
                else:
                    record(
                        "tokenizer-owner-approval",
                        "PASS",
                        "local approval record and registry digest match; identity is not authenticated",
                    )

        trained = (
            release_status.get("status") == "trained"
            and release_status.get("trained_tokenizer_present") is True
        )
        record(
            "trained-tokenizer-artifact",
            "PASS" if trained else "OWNER APPROVAL REQUIRED",
            "release status marks the trained artifact present"
            if trained
            else "no production-trained shared tokenizer artifact is recorded",
        )

    for distribution, module in _DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            record(f"package:{distribution}", "FAIL", "distribution is not installed")
            continue
        if version != "0.1.0":
            record(f"package:{distribution}", "FAIL", f"unexpected version {version}")
            continue
        try:
            importlib.import_module(module)
        except Exception as exc:
            record(
                f"package:{distribution}",
                "FAIL",
                f"version={version}; import failed: {type(exc).__name__}: {exc}",
            )
        else:
            record(f"package:{distribution}", "PASS", f"version={version}; import succeeded")

    uv_path = shutil.which("uv")
    if uv_path is None:
        record("uv", "FAIL", "uv executable is not on PATH")
    else:
        record("uv", "NOT TESTED", f"executable found at {uv_path}; it was not executed")

    lockfile = root / "uv.lock"
    if not root_valid:
        record("uv-lock", "NOT TESTED", "workspace root is unavailable")
    elif not lockfile.is_file():
        record("uv-lock", "FAIL", "uv.lock is missing")
    else:
        record(
            "uv-lock",
            "NOT TESTED",
            "file exists; run `uv lock --check` to verify it matches workspace metadata",
        )

    no_index = os.environ.get("PIP_NO_INDEX", "").strip().lower() in {"1", "true", "yes", "on"}
    record(
        "pip-index-policy",
        "FAIL" if no_index else "PASS",
        "PIP_NO_INDEX disables package-index resolution"
        if no_index
        else "PIP_NO_INDEX does not disable package-index resolution",
    )
    proxy_configured = any(
        os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    )
    record(
        "proxy-configuration",
        "WARNING" if proxy_configured else "PASS",
        "proxy variables are configured; values intentionally omitted"
        if proxy_configured
        else "no proxy variables are configured",
    )
    record("package-index-connectivity", "NOT TESTED", "doctor never makes network requests")

    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        record("pip-check", "NOT TESTED", f"{type(exc).__name__}: {exc}")
    else:
        output = (pip_check.stdout.strip() or pip_check.stderr.strip()).strip()
        record(
            "pip-check",
            "PASS" if pip_check.returncode == 0 else "FAIL",
            output or "no dependency conflicts reported",
        )

    blocking_statuses = {
        "FAIL",
        "NOT TESTED",
        "OWNER APPROVAL REQUIRED",
        "EXTERNAL HARDWARE REQUIRED",
    }
    readiness = (
        "ready" if all(check["status"] not in blocking_statuses for check in checks) else "blocked"
    )
    return {
        "status": readiness,
        "training_started": False,
        "network_access_attempted": False,
        "checks": checks,
        "summary": {
            status: sum(check["status"] == status for check in checks)
            for status in (
                "PASS",
                "FAIL",
                "WARNING",
                "NOT TESTED",
                "OWNER APPROVAL REQUIRED",
                "EXTERNAL HARDWARE REQUIRED",
            )
        },
    }


def render_doctor_text(result: dict[str, Any]) -> str:
    """Render the doctor report without losing machine-readable status values."""
    lines = [f"ORION doctor: {result['status']}"]
    for check in result["checks"]:
        lines.append(f"[{check['status']}] {check['name']}: {check['detail']}")
    lines.append("No training started; no network request was made.")
    return "\n".join(lines)


__all__ = ["render_doctor_text", "run_doctor"]
