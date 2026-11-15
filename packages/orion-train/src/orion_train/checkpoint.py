"""Atomic safe checkpoints with optimizer, scheduler, RNG, and sampler recovery."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from orion_core.errors import ArtifactIntegrityError, ModelWeightsNotFound
from orion_core.hashing import sha256_file, verify_sha256
from orion_core.serialization import load_safetensors, save_safetensors

_CHECKPOINT_SCHEMA_VERSION = "1.2"
_NumpyRNGState = tuple[str, NDArray[np.uint32], int, int, float]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    raise TypeError(f"Training state value {type(value).__name__} is not JSON serializable")


def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuplify(item) for key, item in value.items()}
    return value


def _metadata_contains(actual: Any, expected: Any) -> bool:
    """Check an expected metadata subset recursively without ignoring leaf values."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _metadata_contains(actual[key], value)
            for key, value in expected.items()
        )
    return bool(actual == expected)


def load_checkpoint_model_weights(
    checkpoint: str | Path,
    *,
    expected_model: str,
    expected_model_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    """Read model tensors from a verified checkpoint with matching model identity/config."""
    supplied = Path(checkpoint)
    if supplied.is_file() and supplied.name == "model.safetensors":
        path = supplied.parent
    elif supplied.is_dir() and (supplied / "latest.json").is_file():
        latest = CheckpointManager(supplied).latest()
        if latest is None:
            raise ModelWeightsNotFound(f"No checkpoint exists in {supplied}")
        path = latest
    else:
        path = supplied
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("Initial training checkpoint has an invalid manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ArtifactIntegrityError("Initial training checkpoint uses an unsupported schema")
    files = manifest["files"]
    expected_files = {
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "training_state.json",
    }
    if set(files) != expected_files:
        raise ArtifactIntegrityError("Initial checkpoint manifest has an invalid file list")
    for filename, digest in files.items():
        target = path / filename
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
            or target.is_symlink()
            or target.resolve().parent != path.resolve()
        ):
            raise ArtifactIntegrityError(f"Initial checkpoint has an unsafe {filename} entry")
        verify_sha256(target, digest)
    expected = files["model.safetensors"]
    model_path = path / "model.safetensors"
    try:
        training_state = json.loads((path / "training_state.json").read_text(encoding="utf-8"))
        identity = training_state["metadata"]["run_identity"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactIntegrityError("Initial checkpoint has no training run identity") from exc
    if not isinstance(identity, dict) or not isinstance(identity.get("training_config"), dict):
        raise ArtifactIntegrityError("Initial checkpoint training identity is malformed")
    if (
        not isinstance(training_state, dict)
        or training_state.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        or training_state.get("step") != manifest.get("step")
    ):
        raise ArtifactIntegrityError("Initial checkpoint state and manifest do not agree")
    source_config = identity["training_config"].get("model")
    if identity.get("model") != expected_model or source_config != expected_model_config:
        raise ArtifactIntegrityError("Initial checkpoint model/configuration does not match")
    return load_safetensors(model_path), expected


class CheckpointManager:
    """Save/restore model and training state without pickle deserialization."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int = 0,
        data_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        runtime_state: dict[str, Any] | None = None,
        distributed_states: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Write an immutable numbered checkpoint and update the verified latest pointer."""
        if step < 0 or epoch < 0:
            raise ValueError("step and epoch must be non-negative")
        destination = self.directory / f"step-{step:012d}"
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        temporary = self.directory / f".step-{step:012d}.tmp"
        if temporary.exists():
            raise FileExistsError(f"Checkpoint staging directory already exists: {temporary}")
        temporary.mkdir()
        try:
            model_hash = save_safetensors(model.state_dict(), temporary / "model.safetensors")
            optimizer_tensors, optimizer_metadata = _optimizer_state(model, optimizer)
            optimizer_hash = save_safetensors(
                optimizer_tensors, temporary / "optimizer.safetensors"
            )
            scheduler_state = scheduler.state_dict() if scheduler is not None else None
            numpy_rng = cast(_NumpyRNGState, np.random.get_state())
            state = {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "step": step,
                "epoch": epoch,
                "scheduler_state": _to_jsonable(scheduler_state),
                "optimizer_state": optimizer_metadata,
                "data_state": _to_jsonable(data_state or {}),
                "python_rng_state": _to_jsonable(random.getstate()),
                "numpy_rng_state": {
                    "algorithm": numpy_rng[0],
                    "position": int(numpy_rng[2]),
                    "has_gauss": int(numpy_rng[3]),
                    "cached_gaussian": float(numpy_rng[4]),
                },
                "torch_rng_tensor": "torch_rng",
                "metadata": _to_jsonable(metadata or {}),
                "runtime_state": _to_jsonable(runtime_state or {}),
                "distributed_states": _to_jsonable(distributed_states),
            }
            rng_tensors: dict[str, torch.Tensor] = {
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": torch.from_numpy(numpy_rng[1].astype(np.int64)),
            }
            cuda_rng_count = 0
            if torch.cuda.is_available():
                for index, rng_state in enumerate(torch.cuda.get_rng_state_all()):
                    rng_tensors[f"cuda_rng_{index}"] = rng_state
                    cuda_rng_count += 1
            rng_hash = save_safetensors(rng_tensors, temporary / "rng.safetensors")
            state["cuda_rng_count"] = cuda_rng_count
            _write_json(temporary / "training_state.json", state)
            hashes = {
                "model.safetensors": model_hash,
                "optimizer.safetensors": optimizer_hash,
                "rng.safetensors": rng_hash,
                "training_state.json": sha256_file(temporary / "training_state.json"),
            }
            for filename, expected in hashes.items():
                verify_sha256(temporary / filename, expected)
            _write_json(
                temporary / "manifest.json",
                {"schema_version": _CHECKPOINT_SCHEMA_VERSION, "step": step, "files": hashes},
            )
            os.replace(temporary, destination)
            _write_json(
                self.directory / "latest.json", {"checkpoint": destination.name, "step": step}
            )
        except Exception:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def latest(self) -> Path | None:
        """Return the latest checkpoint only if its pointer and manifest are valid."""
        pointer = self.directory / "latest.json"
        if not pointer.is_file():
            return None
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))
            name = value["checkpoint"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactIntegrityError("Invalid latest checkpoint pointer") from exc
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"step-\d{12}", name)
            or (self.directory / name).resolve().parent != self.directory.resolve()
        ):
            raise ArtifactIntegrityError("Checkpoint pointer contains an unsafe path")
        path = self.directory / name
        if not path.is_dir():
            raise ModelWeightsNotFound(f"Checkpoint directory is missing: {path}")
        return path

    def restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        checkpoint: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        expected_metadata: dict[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore all saved state and random generators from a verified checkpoint."""
        if checkpoint is None:
            path = self.latest()
        else:
            supplied = Path(checkpoint)
            path = supplied if supplied.is_absolute() else self.directory / supplied
            if (
                not re.fullmatch(r"step-\d{12}", path.name)
                or path.resolve().parent != self.directory.resolve()
            ):
                raise ArtifactIntegrityError("Checkpoint path must name a direct child checkpoint")
        if path is None:
            raise ModelWeightsNotFound(f"No checkpoint exists in {self.directory}")
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            state = json.loads((path / "training_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"Checkpoint metadata is invalid at {path}") from exc
        if not isinstance(manifest, dict) or not isinstance(state, dict):
            raise ArtifactIntegrityError("Checkpoint metadata must contain JSON objects")
        if (
            manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
            or state.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        ):
            raise ArtifactIntegrityError("Unsupported checkpoint schema version")
        if manifest.get("step") != state.get("step"):
            raise ArtifactIntegrityError("Checkpoint manifest and training step disagree")
        files = manifest.get("files")
        expected_files = {
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
            "training_state.json",
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ArtifactIntegrityError("Checkpoint manifest has an invalid file list")
        for filename, expected in files.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ArtifactIntegrityError(f"Invalid SHA-256 entry for {filename}")
            target = path / filename
            if target.is_symlink() or target.resolve().parent != path.resolve():
                raise ArtifactIntegrityError(f"Checkpoint file path is unsafe: {filename}")
            verify_sha256(target, expected)
        if expected_metadata:
            actual_metadata = state.get("metadata", {})
            mismatches = [
                key
                for key, value in expected_metadata.items()
                if not _metadata_contains(actual_metadata.get(key), value)
            ]
            if mismatches:
                raise ArtifactIntegrityError(
                    "Checkpoint run metadata does not match: " + ", ".join(sorted(mismatches))
                )
        scheduler_state = state.get("scheduler_state")
        if (scheduler is None) != (scheduler_state is None):
            raise ArtifactIntegrityError("Checkpoint scheduler configuration changed")
        numpy_state = state.get("numpy_rng_state")
        if not isinstance(numpy_state, dict):
            raise ArtifactIntegrityError("Checkpoint is missing NumPy RNG metadata")
        model_state = load_safetensors(path / "model.safetensors", device=map_location)
        model.load_state_dict(model_state, strict=True)
        optimizer_state = load_safetensors(path / "optimizer.safetensors", device=map_location)
        _restore_optimizer(model, optimizer, optimizer_state, state["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if restore_rng:
            rng = load_safetensors(path / "rng.safetensors", device="cpu")
            torch.set_rng_state(rng["torch_rng"].to(dtype=torch.uint8, device="cpu"))
            cuda_count = int(state.get("cuda_rng_count", 0))
            actual_cuda_keys = {key for key in rng if key.startswith("cuda_rng_")}
            if cuda_count != len(actual_cuda_keys):
                raise ArtifactIntegrityError("Checkpoint CUDA RNG state is incomplete")
            if cuda_count and torch.cuda.is_available():
                cuda_states = [
                    rng[f"cuda_rng_{index}"].to(dtype=torch.uint8, device="cpu")
                    for index in range(cuda_count)
                ]
                torch.cuda.set_rng_state_all(cuda_states)
            random.setstate(_tuplify(state["python_rng_state"]))
            np.random.set_state(
                (
                    str(numpy_state["algorithm"]),
                    rng["numpy_rng"].cpu().numpy().astype(np.uint32, copy=False),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        return state


def _optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    tensors: dict[str, torch.Tensor] = {}
    scalar_state: dict[str, dict[str, Any]] = {}
    for parameter, values in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None:
            raise ValueError("optimizer contains a parameter not owned by the model")
        for key, value in values.items():
            tensor_name = f"state/{name}/{key}"
            if isinstance(value, torch.Tensor):
                tensors[tensor_name] = value
            elif isinstance(value, int | float | bool):
                scalar_state.setdefault(name, {})[key] = value
            else:
                raise TypeError(
                    f"Unsupported optimizer state {type(value).__name__} at {name}.{key}"
                )
    groups = []
    for group in optimizer.param_groups:
        group_names = [names[id(parameter)] for parameter in group["params"]]
        options = {key: value for key, value in group.items() if key != "params"}
        groups.append({"names": group_names, "options": _to_jsonable(options)})
    return tensors, {"scalar_state": scalar_state, "param_groups": groups}


def _restore_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    parameters = dict(model.named_parameters())
    optimizer.state.clear()
    for name, scalars in metadata["scalar_state"].items():
        if name not in parameters:
            raise ArtifactIntegrityError(f"Optimizer references unknown parameter {name}")
        optimizer.state[parameters[name]].update(scalars)
    for tensor_name, value in tensors.items():
        if not tensor_name.startswith("state/"):
            raise ArtifactIntegrityError(f"Unexpected optimizer tensor key {tensor_name}")
        _, name, key = tensor_name.split("/", maxsplit=2)
        if name not in parameters:
            raise ArtifactIntegrityError(f"Optimizer references unknown parameter {name}")
        optimizer.state[parameters[name]][key] = value
    saved_groups = metadata["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ArtifactIntegrityError("Optimizer parameter group count changed since checkpoint")
    for saved, current in zip(saved_groups, optimizer.param_groups, strict=True):
        current_names = [
            next(name for name, parameter in parameters.items() if parameter is item)
            for item in current["params"]
        ]
        if saved["names"] != current_names:
            raise ArtifactIntegrityError("Optimizer parameter group order changed since checkpoint")
        for key, value in saved["options"].items():
            current[key] = value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
