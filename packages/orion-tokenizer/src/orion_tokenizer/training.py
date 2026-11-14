"""Explicit local-corpus byte-level BPE training; no acquisition is performed here."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import tokenizers
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from orion_core.hashing import sha256_file
from orion_tokenizer.spec import (
    BYTE_TOKEN_END,
    BYTE_TOKEN_START,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
)
from orion_tokenizer.tokenizer import _bytes_to_unicode

_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?(?:\p{L}\p{M}*)+| ?\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"
)


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    """Reproducible tokenizer training parameters; data selection remains external."""

    vocab_size: int = VOCAB_SIZE
    tokenizer_version: str = TOKENIZER_VERSION
    min_frequency: int = 2
    seed: int = 1729
    status: str = "trained"
    token_map_approved: bool = False

    def validate(self) -> None:
        if self.vocab_size < 512 or self.vocab_size > VOCAB_SIZE:
            raise ValueError(
                "vocab_size must be between the 512 reserved byte/special IDs and 32768"
            )
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if self.status not in {"trained", "synthetic-test-fixture"}:
            raise ValueError("status must be trained or synthetic-test-fixture")
        if self.status == "trained" and not self.token_map_approved:
            raise ValueError(
                "the provisional special-token map must be explicitly reviewed before "
                "a production tokenizer can be marked trained"
            )


def iter_local_text(paths: Iterable[str | Path], *, field: str = "text") -> Iterator[str]:
    """Stream deterministic UTF-8 lines or JSONL text fields from explicit local paths."""
    files = sorted((Path(path) for path in paths), key=lambda path: str(path.resolve()).casefold())
    if not files:
        raise ValueError("At least one local corpus file is required")
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(f"Corpus input is not a regular file: {source}")
        with source.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                if source.suffix.lower() == ".jsonl":
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
                    value = row.get(field) if isinstance(row, dict) else None
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Expected a string field {field!r} at {source}:{line_number}"
                        )
                    stripped = value
                normalized = unicodedata.normalize("NFC", stripped)
                if normalized:
                    yield normalized


def _make_backend() -> Tokenizer:
    backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.normalizer = normalizers.NFC()
    regex_split = pre_tokenizers.Split(pre_tokenizers.Regex(_PATTERN), behavior="isolated")
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.pre_tokenizer = pre_tokenizers.Sequence([regex_split, byte_level])
    backend.decoder = decoders.ByteLevel()
    return backend


def train_tokenizer(
    paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: TokenizerTrainingConfig | None = None,
    jsonl_text_field: str = "text",
) -> Path:
    """Train a byte-level BPE tokenizer from local corpora and write a complete release.

    This function never downloads data. Production release training should only be invoked
    after the corpus/license manifest and provisional token-ID decisions are approved.
    """
    training_config = config or TokenizerTrainingConfig()
    training_config.validate()
    corpus_files = sorted(
        (Path(path) for path in paths), key=lambda path: str(path.resolve()).casefold()
    )
    if not corpus_files:
        raise ValueError("At least one local corpus file is required")
    # Materialize path validation and checksums before mutating the destination.
    corpus_records = []
    for source in corpus_files:
        if not source.is_file():
            raise FileNotFoundError(f"Corpus input is not a regular file: {source}")
        corpus_records.append({"name": source.name, "sha256": sha256_file(source)})
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty tokenizer release directory {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    backend = _make_backend()
    special_tokens = [
        AddedToken(token, special=True, normalized=False, lstrip=False, rstrip=False)
        for token in SPECIAL_TOKENS
    ]
    byte_map = _bytes_to_unicode()
    byte_alphabet = [byte_map[byte] for byte in range(256)]
    trainer = trainers.BpeTrainer(
        vocab_size=training_config.vocab_size,
        min_frequency=training_config.min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=byte_alphabet,
    )
    backend.train_from_iterator(
        iter_local_text(corpus_files, field=jsonl_text_field),
        trainer=trainer,
        length=None,
    )
    actual_vocab_size = backend.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != training_config.vocab_size:
        raise ValueError(
            f"BPE training produced {actual_vocab_size} entries; expected {training_config.vocab_size}. "
            "Provide a larger and more varied permitted corpus before freezing the tokenizer."
        )
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        if backend.token_to_id(token) != expected_id:
            raise ValueError(f"Special token {token!r} has an unexpected ID")
    vocab = backend.get_vocab(with_added_tokens=True)
    byte_ids = {vocab.get(token) for token in byte_alphabet}
    expected_byte_ids = set(range(BYTE_TOKEN_START, BYTE_TOKEN_END + 1))
    if byte_ids != expected_byte_ids:
        raise ValueError("Byte alphabet does not occupy frozen IDs 256 through 511")

    tokenizer_path = destination / "tokenizer.json"
    backend.save(str(tokenizer_path), pretty=True)
    tokenizer_digest = sha256_file(tokenizer_path)
    special_map = {str(index): SPECIAL_TOKENS[index] for index in range(len(SPECIAL_TOKENS))}
    special_digest = hashlib.sha256(
        json.dumps(special_map, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "status": training_config.status,
        "tokenizer_version": training_config.tokenizer_version,
        "algorithm": "byte-level BPE",
        "normalization": "NFC",
        "vocab_size": actual_vocab_size,
        "special_tokens_sha256": special_digest,
        "tokenizer_sha256": tokenizer_digest,
        "corpora": corpus_records,
        "seed": training_config.seed,
        "determinism_note": (
            "Tokenizers BPE trainer is deterministic for fixed ordered input and versions; "
            "seed is recorded for upstream sampling and is not a trainer RNG argument."
        ),
        "tokenizers_version": tokenizers.__version__,
        "training_config": asdict(training_config),
    }
    (destination / "vocab_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_version": training_config.tokenizer_version,
                "vocab_size": actual_vocab_size,
                "normalization": "NFC",
                "byte_level": True,
                "add_prefix_space": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "special_tokens.json").write_text(
        json.dumps(special_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_golden_encodings(backend, destination / "golden_encodings.jsonl")
    return destination


def _write_golden_encodings(backend: Tokenizer, path: Path) -> None:
    """Create small deterministic release vectors; no training examples are embedded."""
    examples = [
        "A short English string.",
        "اردو متن اور database systems",
        "x = a_{i}^2 + \\alpha",
        "def f(x):\n    return x + 1",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for text in examples:
            encoded = backend.encode(text, add_special_tokens=False)
            record = {
                "fixture_kind": "encoding-contract",
                "text": text,
                "input_ids": encoded.ids,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
