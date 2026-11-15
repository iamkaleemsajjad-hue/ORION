"""Local verification of the documented owner approval gate for tokenizer v1."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from orion_core.errors import ConfigurationError
from orion_tokenizer.spec import special_token_map_sha256


def require_tokenizer_owner_approval(
    project_root: str | Path,
    expected_token_map_sha256: str,
) -> dict[str, Any]:
    """Require matching approval in the owner record and machine release status.

    This checks local records and digest consistency; it cannot authenticate the
    identity of the person who edited them.
    """
    root = Path(project_root).resolve(strict=True)
    if expected_token_map_sha256 != special_token_map_sha256():
        raise ConfigurationError("Requested token map differs from the current v1 registry")
    status_path = root / "packages/orion-tokenizer/releases/v1.0/STATUS.json"
    approval_path = root / "docs/tokenizer/tokenizer_approval.md"
    status = _read_status(status_path)
    if status.get("owner_approval") != "approved":
        raise ConfigurationError("owner approval is pending in tokenizer v1 STATUS.json")
    if status.get("special_token_map_sha256") != expected_token_map_sha256:
        raise ConfigurationError("STATUS.json token-map digest does not match the training recipe")
    if status.get("status") != "pending" or status.get("trained_tokenizer_present") is not False:
        raise ConfigurationError("tokenizer v1 status is not in the approved pre-training state")

    fields = _read_approval_fields(approval_path)
    required = (
        "Owner name or authorized role",
        "Decision date (ISO 8601)",
        "Approved special-token map SHA-256",
        "Approved byte mapping and normalization",
        "Approved generation-emittable controls",
        "Notes or required changes",
    )
    if fields.get("Decision", "").casefold() != "approved":
        raise ConfigurationError("tokenizer_approval.md decision must be APPROVED")
    if any(not fields.get(key) or fields[key].casefold() == "pending" for key in required):
        raise ConfigurationError("tokenizer_approval.md has incomplete owner decision fields")
    try:
        date.fromisoformat(fields["Decision date (ISO 8601)"])
    except ValueError as exc:
        raise ConfigurationError("owner decision date must use ISO 8601 YYYY-MM-DD") from exc
    if fields["Approved special-token map SHA-256"].lower() != expected_token_map_sha256:
        raise ConfigurationError("owner approval digest does not match the current token map")
    return status


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"could not read tokenizer release status: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("tokenizer release status must be a JSON object")
    return value


def _read_approval_fields(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"could not read tokenizer approval record: {exc}") from exc
    fields: dict[str, str] = {}
    for line in content.splitlines():
        columns = [column.strip().strip("`") for column in line.strip().strip("|").split("|")]
        if len(columns) == 2:
            fields[columns[0]] = columns[1]
    return fields


__all__ = ["require_tokenizer_owner_approval"]
