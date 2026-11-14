"""Explicit Hugging Face Hub artifact operations with local release verification."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from tools.verify_release import verify_release

_RELEASE_FILES = (
    "artifact.json",
    "config.json",
    "model.safetensors",
    "model.safetensors.sha256",
    "vocab_manifest.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens.json",
    "golden_encodings.jsonl",
    "training_report.json",
    "evaluation_report.json",
    "data_manifest.json",
    "README.md",
    "LICENSE",
)


def _allow_patterns() -> list[str]:
    """Match root release files and their tokenizer-subdirectory counterparts."""
    return [*_RELEASE_FILES, *(f"tokenizer/{name}" for name in _RELEASE_FILES)]


def _hub_api() -> type:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Hub operations require `pip install orion-data[hub]`") from exc
    return HfApi


def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def _release_paths(root: Path, model: str) -> list[Path]:
    tokenizer_root = root / "tokenizer"
    paths = []
    for name in _RELEASE_FILES:
        for base in (root, tokenizer_root):
            path = base / name
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def upload_release(root: str | Path, *, model: str, repo_id: str, confirm: bool) -> str:
    """Verify and upload a deliberately selected private model release."""
    if not confirm:
        raise ValueError("Upload was not started; pass --confirm-upload to authorize it")
    source = Path(root)
    verify_release(source, model)
    files = _release_paths(source, model)
    required = {"config.json", "model.safetensors", "model.safetensors.sha256"}
    available = {path.name for path in files}
    if model == "orion-300m":
        required.add("artifact.json")
    if not required.issubset(available):
        raise ValueError(
            f"release upload is missing required files: {sorted(required - available)}"
        )
    api_class = _hub_api()
    api = api_class(token=_token())
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(source),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[str(path.relative_to(source)).replace("\\", "/") for path in files],
        commit_message=f"Upload verified {model} artifact",
        token=_token(),
    )
    return str(getattr(commit, "oid", "upload-complete"))


def download_release(repo_id: str, *, model: str, revision: str, destination: str | Path) -> Path:
    """Download an explicit pinned Hub revision and verify it before use."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("download requires a full immutable 40-character Hub commit hash")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hub operations require `pip install orion-data[hub]`") from exc
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to mix Hub files into non-empty destination {target}")
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(target),
            token=_token(),
            allow_patterns=_allow_patterns(),
        )
    )
    verify_release(local, model)
    return local


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--model", choices=("orion-100m", "orion-300m"), required=True)
    upload.add_argument("--release-dir", required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--confirm-upload", action="store_true")
    download = commands.add_parser("download")
    download.add_argument("--model", choices=("orion-100m", "orion-300m"), required=True)
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    if args.command == "upload":
        print(
            upload_release(
                args.release_dir,
                model=args.model,
                repo_id=args.repo_id,
                confirm=args.confirm_upload,
            )
        )
    else:
        print(
            download_release(
                args.repo_id, model=args.model, revision=args.revision, destination=args.destination
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
