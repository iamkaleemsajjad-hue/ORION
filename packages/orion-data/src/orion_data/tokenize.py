"""Local text-to-token-shard preparation with whole-word alignment and provenance."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from orion_core.hashing import sha256_file
from orion_data.normalize import normalize_text
from orion_data.shards import write_token_shard, write_word_id_shard


class Encoder(Protocol):
    """Minimal tokenizer interface needed by the local data pipeline."""

    version: str
    sha256: str

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special_tokens: bool = False,
        max_length: int | None = None,
        truncation: str | None = None,
    ) -> EncodedText: ...

    def special_token_id(self, token: str) -> int: ...


class EncodedText(Protocol):
    """Minimal typed result returned by the tokenizer backends used here."""

    @property
    def input_ids(self) -> Sequence[int]: ...

    @property
    def word_ids(self) -> Sequence[int | None]: ...


def tokenize_texts(
    texts: Iterable[str], tokenizer: Encoder, *, add_eos: bool = False
) -> tuple[list[int], list[int]]:
    """Encode local text in order and maintain globally distinct word IDs."""
    token_ids: list[int] = []
    word_ids: list[int] = []
    next_word = 0
    for text in texts:
        normalized = normalize_text(text)
        if not normalized.strip():
            continue
        encoded = tokenizer.encode(normalized, add_eos=add_eos)
        token_ids.extend(encoded.input_ids)
        present = [value for value in encoded.word_ids if value is not None]
        for value in encoded.word_ids:
            word_ids.append(-1 if value is None else next_word + value)
        if present:
            next_word += max(present) + 1
    if len(token_ids) != len(word_ids):
        raise RuntimeError("tokenizer returned misaligned token and word ID arrays")
    return token_ids, word_ids


def write_aligned_token_shards(
    texts: Iterable[str],
    tokenizer: Encoder,
    output_dir: str | Path,
    *,
    add_eos: bool = False,
    synthetic_fixture: bool = False,
) -> dict[str, object]:
    """Write token IDs, whole-word IDs, and a tokenizer-hash manifest locally."""
    tokens, words = tokenize_texts(texts, tokenizer, add_eos=add_eos)
    if not tokens:
        raise ValueError("no non-empty text was tokenized")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty shard directory {root}")
    root.mkdir(parents=True, exist_ok=True)
    token_record = write_token_shard(root / "tokens.u16", tokens)
    word_record = write_word_id_shard(root / "word_ids.i32", words)
    manifest = {
        "schema_version": "1.0",
        "status": "locally_prepared",
        "synthetic_fixture": synthetic_fixture,
        "tokenizer_version": tokenizer.version,
        "tokenizer_sha256": tokenizer.sha256,
        "token_count": len(tokens),
        "token_shard": {**token_record, "sha256": sha256_file(root / "tokens.u16")},
        "word_id_shard": {**word_record, "sha256": sha256_file(root / "word_ids.i32")},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["Encoder", "tokenize_texts", "write_aligned_token_shards"]
