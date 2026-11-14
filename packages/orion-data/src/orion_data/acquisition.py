"""Optional, pinned-revision Hub snapshot helper (never invoked implicitly)."""

from __future__ import annotations

from pathlib import Path


def fetch_pinned_snapshot(
    repo_id: str, revision: str, destination: str | Path, *, allow_network: bool = False
) -> Path:
    if not allow_network:
        raise PermissionError("Network acquisition is disabled; explicitly pass allow_network=True")
    if not repo_id or not revision or len(revision) < 7:
        raise ValueError("repo_id and a pinned commit revision are required")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional `orion-data[hub]` extra to enable acquisition"
        ) from exc
    local_dir = Path(destination)
    local_dir.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        local_files_only=False,
    )
    return Path(result)
