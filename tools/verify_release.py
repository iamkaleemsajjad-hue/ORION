"""Verify local release metadata and hashes; this tool never publishes artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion_core.hashing import sha256_file


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release metadata file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"release metadata must be a JSON object: {path.name}")
    return value


def _check_sha(path: Path, expected: object) -> None:
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected.lower())
        or sha256_file(path) != expected.lower()
    ):
        raise ValueError(f"missing file or SHA-256 mismatch: {path.name}")


def verify_release(root: Path, model: str) -> dict[str, object]:
    if model not in {"orion-100m", "orion-300m"}:
        raise ValueError("model must be orion-100m or orion-300m")
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if model == "orion-300m":
        manifest_path = root / "artifact.json"
        manifest = _json_object(manifest_path)
        if manifest.get("model_name") != model or manifest.get("trained") is not True:
            raise ValueError("artifact manifest is not a trained ORION-300M release")
        if manifest.get("status") not in {"preview", "accepted"}:
            raise ValueError("release status must be preview or accepted")
        for report in ("training_report", "evaluation_report", "data_manifest_sha256"):
            if not manifest.get(report):
                raise ValueError(f"release manifest is missing {report}")
        _check_sha(root / "model.safetensors", manifest.get("weights_sha256"))
        _check_sha(root / "config.json", manifest.get("config_sha256"))
        tokenizer_root = root / "tokenizer"
        if not tokenizer_root.exists():
            tokenizer_root = root
        tokenizer_manifest = _json_object(tokenizer_root / "vocab_manifest.json")
        if tokenizer_manifest.get("status") != "trained":
            raise ValueError("release tokenizer is not marked trained")
        if tokenizer_manifest.get("vocab_size") != 32768:
            raise ValueError("release tokenizer vocabulary must contain exactly 32,768 IDs")
        if tokenizer_manifest.get("tokenizer_version") != manifest.get("tokenizer_version"):
            raise ValueError("release model and tokenizer versions do not match")
        _check_sha(tokenizer_root / "tokenizer.json", manifest.get("tokenizer_sha256"))
    else:
        manifest_path = root / "config.json"
        manifest = _json_object(manifest_path)
        if (
            manifest.get("artifact_status") != "trained"
            or manifest.get("training_completed") is not True
        ):
            raise ValueError("config does not identify a completed trained ORION-100M model")
        if not manifest.get("training_data_manifest_sha256"):
            raise ValueError("release config has no training data provenance hash")
        weights = root / "model.safetensors"
        sidecar = weights.with_suffix(weights.suffix + ".sha256")
        sidecar_record = _json_object(sidecar)
        _check_sha(weights, sidecar_record.get("sha256"))
        _check_sha(weights, manifest.get("weights_sha256"))
        tokenizer_root = root / "tokenizer"
        if not tokenizer_root.exists():
            tokenizer_root = root
        tokenizer_manifest = _json_object(tokenizer_root / "vocab_manifest.json")
        if tokenizer_manifest.get("status") != "trained":
            raise ValueError("release tokenizer is not marked trained")
        if tokenizer_manifest.get("vocab_size") != 32768:
            raise ValueError("release tokenizer vocabulary must contain exactly 32,768 IDs")
        model_config = manifest.get("model_config")
        if not isinstance(model_config, dict) or tokenizer_manifest.get(
            "tokenizer_version"
        ) != model_config.get("tokenizer_version"):
            raise ValueError("release model and tokenizer versions do not match")
        _check_sha(tokenizer_root / "tokenizer.json", manifest.get("tokenizer_sha256"))
    forbidden = [root / "pytorch_model.bin", root / "training_state.pt"]
    if any(item.exists() for item in forbidden):
        raise ValueError("unsafe pickle-based model or training files are forbidden in releases")
    return {
        "verified": True,
        "model": model,
        "status": manifest.get("status", "trained"),
        "manifest": str(manifest_path.resolve()),
        "weights_sha256": manifest.get("weights_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("orion-100m", "orion-300m"), required=True)
    parser.add_argument("release_dir")
    args = parser.parse_args(argv)
    print(json.dumps(verify_release(Path(args.release_dir), args.model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
