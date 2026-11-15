"""Integrity validation and explicit local export for trained tokenizer releases."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from orion_core.errors import TokenizerMismatchError
from orion_tokenizer.spec import (
    SPECIAL_TOKENS,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
    special_token_map_sha256,
)
from orion_tokenizer.tokenizer import OrionTokenizer

_RELEASE_FILES = (
    "tokenizer.json",
    "vocab_manifest.json",
    "tokenizer_config.json",
    "special_tokens.json",
    "golden_encodings.jsonl",
)


def verify_tokenizer_release(directory: str | Path) -> dict[str, object]:
    """Verify a complete trained tokenizer, its ID registry, and golden vectors."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise TokenizerMismatchError("Tokenizer release path must be a directory")
    for filename in _RELEASE_FILES:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise TokenizerMismatchError(f"Tokenizer release is missing a regular {filename}")

    manifest = _read_object(root / "vocab_manifest.json")
    if manifest.get("status") != "trained":
        raise TokenizerMismatchError("Tokenizer manifest does not identify a trained release")
    if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
        raise TokenizerMismatchError(f"Expected tokenizer version {TOKENIZER_VERSION}")
    if manifest.get("vocab_size") != VOCAB_SIZE:
        raise TokenizerMismatchError(f"Tokenizer vocabulary must contain {VOCAB_SIZE} IDs")
    training_config = manifest.get("training_config")
    if (
        not isinstance(training_config, dict)
        or training_config.get("token_map_approved") is not True
    ):
        raise TokenizerMismatchError(
            "Release manifest does not record reviewed token-map assertion"
        )
    special_digest = special_token_map_sha256()
    if training_config.get("approved_token_map_sha256") != special_digest:
        raise TokenizerMismatchError(
            "Release training config is not bound to the current token-map digest"
        )

    special_map = {str(index): SPECIAL_TOKENS[index] for index in range(256)}
    if manifest.get("special_tokens_sha256") != special_digest:
        raise TokenizerMismatchError(
            "Tokenizer manifest special-token map digest does not match v1"
        )
    if _read_object(root / "special_tokens.json") != special_map:
        raise TokenizerMismatchError("special_tokens.json does not match the frozen v1 registry")

    config = _read_object(root / "tokenizer_config.json")
    expected_config = {
        "tokenizer_version": TOKENIZER_VERSION,
        "vocab_size": VOCAB_SIZE,
        "normalization": "NFC",
        "byte_level": True,
        "add_prefix_space": False,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise TokenizerMismatchError("tokenizer_config.json does not match the v1 contract")

    tokenizer = OrionTokenizer.load(root)
    backend_vocab = tokenizer._tokenizer.get_vocab(with_added_tokens=True)
    if set(backend_vocab.values()) != set(range(VOCAB_SIZE)):
        raise TokenizerMismatchError("Tokenizer backend IDs are not a contiguous 0..32767 range")
    if tokenizer._byte_ids != tuple(range(256, 512)):
        raise TokenizerMismatchError("Tokenizer byte alphabet does not occupy IDs 256 through 511")
    golden_count = _verify_golden_encodings(root / "golden_encodings.jsonl", tokenizer)
    return {
        "verified": True,
        "status": "trained",
        "tokenizer_version": tokenizer.version,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.sha256,
        "special_tokens_sha256": special_digest,
        "golden_vector_count": golden_count,
        "tokenizers_version": manifest.get("tokenizers_version"),
        "directory": str(root),
    }


def export_tokenizer_release(source: str | Path, destination: str | Path) -> dict[str, object]:
    """Copy a verified local release into a new, atomically created directory."""
    verified = verify_tokenizer_release(source)
    source_root = Path(str(verified["directory"]))
    target = Path(destination).absolute()
    target_resolved = target.resolve(strict=False)
    if target_resolved == source_root or target_resolved.is_relative_to(source_root):
        raise ValueError("tokenizer export destination must be outside its source directory")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("tokenizer export destination must be a regular directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty tokenizer directory {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for filename in _RELEASE_FILES:
            shutil.copyfile(source_root / filename, staging / filename)
        staged_verification = verify_tokenizer_release(staging)
        if staged_verification["tokenizer_sha256"] != verified["tokenizer_sha256"]:
            raise TokenizerMismatchError("Exported tokenizer hash differs from its source")
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**verified, "directory": str(target.resolve()), "exported": True}


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerMismatchError(f"Invalid tokenizer release metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise TokenizerMismatchError(f"Tokenizer metadata must be a JSON object: {path.name}")
    return value


def _verify_golden_encodings(path: Path, tokenizer: OrionTokenizer) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or value.get("fixture_kind") != "encoding-contract"
                    or not isinstance(value.get("text"), str)
                    or not isinstance(value.get("input_ids"), list)
                    or any(type(token_id) is not int for token_id in value["input_ids"])
                ):
                    raise ValueError(f"invalid golden vector at line {line_number}")
                encoded = tokenizer.encode(value["text"], allow_special_tokens=True)
                if list(encoded.input_ids) != value["input_ids"]:
                    raise ValueError(f"golden encoding mismatch at line {line_number}")
                count += 1
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TokenizerMismatchError(f"Tokenizer golden encodings failed: {exc}") from exc
    if count == 0:
        raise TokenizerMismatchError("Tokenizer release has no golden encoding vectors")
    return count


__all__ = ["export_tokenizer_release", "verify_tokenizer_release"]
