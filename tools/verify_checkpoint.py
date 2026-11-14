"""Verify a checkpoint directory's safe tensor hashes and metadata consistency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from orion_core.hashing import sha256_file

_SCHEMA_VERSION = "1.2"


def verify(path: Path) -> dict[str, object]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((path / "training_state.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint metadata must contain JSON objects")
    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or state.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("unsupported checkpoint schema")
    files = manifest.get("files")
    expected_files = {
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "training_state.json",
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("checkpoint manifest does not enumerate the expected state files")
    for name, expected in files.items():
        target = path / name
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not target.is_file()
            or target.is_symlink()
            or target.resolve().parent != path.resolve()
            or sha256_file(target) != expected
        ):
            raise ValueError(f"checkpoint file is missing or has a bad SHA-256: {name}")
    if manifest.get("step") != state.get("step"):
        raise ValueError("checkpoint step differs between manifest and training state")
    return {
        "verified": True,
        "step": state["step"],
        "epoch": state["epoch"],
        "tensor_files": sorted(name for name in files if name.endswith(".safetensors")),
        "training_state_verified": True,
        "path": str(path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="checkpoint directory or parent with latest.json")
    args = parser.parse_args(argv)
    path = Path(args.checkpoint)
    if (path / "latest.json").is_file():
        pointer = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        child = pointer.get("checkpoint")
        if (
            not isinstance(child, str)
            or not re.fullmatch(r"step-\d{12}", child)
            or (path / child).resolve().parent != path.resolve()
        ):
            raise ValueError("latest checkpoint pointer contains an unsafe path")
        path = path / child
    print(json.dumps(verify(path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
