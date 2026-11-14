"""Local sealed-set evaluation ledger that prevents accidental repeat scoring."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from orion_core.errors import OrionError


class SealedSetAlreadyEvaluated(OrionError):
    """Raised when a test split/release pair is being scored a second time."""

    code = "ORION-EVAL-0001"


class SealedEvaluationGuard:
    """Record each sealed split hash and require an explicit release tag."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, evaluation_id: str, split_sha256: str, release_tag: str | None) -> None:
        if not evaluation_id or len(split_sha256) != 64:
            raise ValueError("evaluation_id and a 64-character split hash are required")
        if not release_tag or not release_tag.strip():
            raise ValueError("sealed evaluation requires an explicit release tag")
        entries = []
        if self.path.exists():
            entries = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("sealed evaluation ledger must contain a JSON list")
        duplicate = any(
            item.get("evaluation_id") == evaluation_id
            and item.get("split_sha256") == split_sha256
            and item.get("release_tag") == release_tag
            for item in entries
            if isinstance(item, dict)
        )
        if duplicate:
            raise SealedSetAlreadyEvaluated(
                "This sealed evaluation split has already been scored for this release tag"
            )
        entries.append(
            {
                "evaluation_id": evaluation_id,
                "split_sha256": split_sha256,
                "release_tag": release_tag,
            }
        )
        fd, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(fd)
        temporary = Path(name)
        try:
            temporary.write_text(
                json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
