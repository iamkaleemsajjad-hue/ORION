"""Tokenizer directory validation without network access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import TokenizerMismatchError
from .hashing import sha256_file


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    version: str
    vocab_size: int
    sha256: str


def inspect_tokenizer_dir(directory: str | Path) -> TokenizerIdentity:
    path = Path(directory)
    model_file = path / "tokenizer.json" if path.is_dir() else path
    if not model_file.is_file():
        raise FileNotFoundError(f"Tokenizer file was not found: {model_file}")
    config_file = model_file.parent / "tokenizer_config.json"
    if not config_file.is_file():
        raise TokenizerMismatchError("tokenizer_config.json is required beside tokenizer.json")
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
        size = int(config["vocab_size"])
        version = str(config["tokenizer_version"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise TokenizerMismatchError(f"Invalid tokenizer_config.json: {exc}") from exc
    return TokenizerIdentity(version=version, vocab_size=size, sha256=sha256_file(model_file))


def verify_tokenizer_dir(
    directory: str | Path,
    expected_sha256: str,
    expected_vocab_size: int,
    expected_version: str,
) -> TokenizerIdentity:
    identity = inspect_tokenizer_dir(directory)
    if identity.sha256 != expected_sha256:
        raise TokenizerMismatchError(
            f"Tokenizer SHA-256 mismatch: expected {expected_sha256}, got {identity.sha256}"
        )
    if identity.vocab_size != expected_vocab_size or identity.version != expected_version:
        raise TokenizerMismatchError(
            "Tokenizer identity mismatch: expected "
            f"version={expected_version}, vocab_size={expected_vocab_size}; got "
            f"version={identity.version}, vocab_size={identity.vocab_size}"
        )
    return identity
