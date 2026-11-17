"""Opt-in revision-pinned Hugging Face dataset snapshot acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from orion_data.manifest import DatasetManifest


def acquire_manifest_sources(
    manifest: DatasetManifest,
    output_root: str | Path,
    *,
    execute: bool = False,
    # This is the environment variable name; the credential value is read at call time.
    token_env_var: str = "HF_TOKEN",  # noqa: S107
    license_mode: str = "research",
) -> list[Path]:
    """Download only explicitly requested HF dataset snapshots at pinned revisions."""
    if not execute:
        raise PermissionError(
            "Dataset acquisition is disabled unless execute=True is explicitly supplied"
        )
    manifest.validate_license_policy(license_mode)
    sources = [
        source
        for source in manifest.sources
        if not source.synthetic and source.kind == "huggingface-dataset"
    ]
    if not sources:
        raise ValueError("manifest contains no explicitly downloadable Hugging Face datasets")
    target_by_source = {}
    for source in sources:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", source.revision):
            raise ValueError(f"Source {source.source_id} does not pin a 40-character commit hash")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
            source.source_id,
        ):
            raise ValueError(f"Invalid Hugging Face dataset repository ID: {source.source_id!r}")
        target_by_source[source.source_id] = source.source_id.replace("/", "--")
    if len(set(target_by_source.values())) != len(target_by_source):
        raise ValueError("Distinct source IDs resolve to the same local target name")
    destination_root = Path(output_root)
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(
            f"Refusing to mix acquired datasets into non-empty directory {destination_root}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hub acquisition requires `pip install orion-data[hub]`") from exc
    destination_root.mkdir(parents=True, exist_ok=True)
    results = []
    source_records = []
    for source in sources:
        target = destination_root / target_by_source[source.source_id]
        if target.resolve().parent != destination_root.resolve():
            raise ValueError("Resolved dataset target escapes the requested output directory")
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"Refusing to mix dataset files into non-empty directory {target}"
            )
        result = snapshot_download(
            repo_id=source.source_id,
            repo_type="dataset",
            revision=source.revision,
            local_dir=target,
            token=os.environ.get(token_env_var),
            local_files_only=False,
        )
        local = Path(result).resolve()
        if local != target.resolve():
            raise ValueError("Dataset client returned a snapshot outside its requested directory")
        files = []
        for path in sorted(local.rglob("*")):
            if not path.is_file() or path.is_symlink() or ".cache" in path.relative_to(local).parts:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "path": path.relative_to(local).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        source_records.append(
            {
                "source_id": source.source_id,
                "revision": source.revision,
                "license": source.license.model_dump(mode="json"),
                "files": files,
            }
        )
        results.append(local)
    acquisition_record = {
        "schema_version": "1.0",
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest(),
        "sources": source_records,
    }
    temporary = destination_root / ".acquisition_manifest.json.tmp"
    temporary.write_text(
        json.dumps(acquisition_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination_root / "acquisition_manifest.json")
    return results
