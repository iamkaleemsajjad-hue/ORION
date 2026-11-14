"""Safe local tensor serialization and hash-first model state loading."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from orion_core.errors import ArtifactIntegrityError, ModelWeightsNotFound
from orion_core.hashing import sha256_file, verify_sha256


def save_safetensors(state: dict[str, torch.Tensor], path: str | Path) -> str:
    """Atomically save detached CPU tensors and return their SHA-256 digest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu_state: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State value {name!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise TypeError(f"State tensor {name!r} must have strided layout")
        # Clone each entry so tied-weight aliases are emitted as independent safe tensors.
        cpu_state[name] = tensor.detach().contiguous().cpu().clone()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(cpu_state, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(destination)
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    _write_json_atomic(checksum_path, {"sha256": digest, "file": destination.name})
    return digest


def load_safetensors(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Load only safetensors data after checking a neighboring SHA-256 record."""
    source = Path(path)
    if not source.is_file():
        raise ModelWeightsNotFound(f"Model weights were not found at {source}")
    checksum_path = source.with_suffix(source.suffix + ".sha256")
    if not checksum_path.is_file():
        raise ArtifactIntegrityError(f"Missing checksum manifest for {source.name}")
    try:
        checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Invalid checksum manifest: {checksum_path}") from exc
    expected = checksum.get("sha256")
    if not isinstance(expected, str):
        raise ArtifactIntegrityError("Checksum manifest must contain a SHA-256 string")
    verify_sha256(source, expected)
    try:
        return load_file(str(source), device=str(device))
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Could not load safetensors file {source.name}: {exc}"
        ) from exc


def load_model_state(
    module: torch.nn.Module,
    weights_path: str | Path,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> None:
    """Verify and load a model state dict from safetensors, never from pickle."""
    state = load_safetensors(weights_path, device=device)
    incompatible = module.load_state_dict(state, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise ArtifactIntegrityError(
            "Weight keys do not match the configured model",
            details={
                "missing": incompatible.missing_keys,
                "unexpected": incompatible.unexpected_keys,
            },
        )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
