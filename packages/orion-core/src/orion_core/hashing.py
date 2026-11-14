"""Content hashing and checksum validation for local artifacts and data shards."""

from __future__ import annotations

import hashlib
from pathlib import Path

from orion_core.errors import ArtifactIntegrityError


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hash UTF-8 text without platform-dependent newline conversion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Verify an artifact digest, raising on malformed or mismatching hashes."""
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ArtifactIntegrityError("Expected SHA-256 value is not a 64-character hex digest")
    actual = sha256_file(path)
    if actual != normalized:
        raise ArtifactIntegrityError(
            f"SHA-256 mismatch for {Path(path).name}",
            details={"expected": normalized, "actual": actual},
        )
    return actual
