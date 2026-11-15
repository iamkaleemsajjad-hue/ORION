"""Local tokenizer loading, safe untrusted-text encoding, and batch padding."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tokenizers import Tokenizer

from orion_core.errors import InputLimitError, TokenizerMismatchError
from orion_core.hashing import sha256_file
from orion_tokenizer.spec import (
    ACTIVE_SPECIAL_TOKENS,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
)

_SPECIAL_PATTERN = re.compile(
    "("
    + "|".join(re.escape(token) for token in sorted(ACTIVE_SPECIAL_TOKENS, key=len, reverse=True))
    + ")"
)


@dataclass(frozen=True)
class TokenizedText:
    """Encoding with optional word alignment from the Rust Tokenizers backend."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    word_ids: tuple[int | None, ...]
    offset_mapping: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TokenBatch:
    """Padded batch of token IDs and masks, preserving caller order."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    word_ids: tuple[tuple[int | None, ...], ...]
    offset_mapping: tuple[tuple[tuple[int, int], ...], ...]


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte alphabet map used by the Tokenizers ByteLevel pre-tokenizer."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAD))
    visible += list(range(0xAE, 0x100))
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(visible, codepoints, strict=True)}


class OrionTokenizer:
    """A loaded, hash-checked tokenizer. Runtime loading never trains or downloads."""

    def __init__(self, tokenizer: Tokenizer, *, path: Path, manifest: dict[str, object]) -> None:
        version = manifest.get("tokenizer_version")
        vocab_size = manifest.get("vocab_size")
        digest = manifest.get("tokenizer_sha256")
        if (
            not isinstance(version, str)
            or type(vocab_size) is not int
            or not isinstance(digest, str)
        ):
            raise TokenizerMismatchError(
                "Tokenizer manifest has invalid version, vocabulary, or hash"
            )
        self._tokenizer = tokenizer
        self.path = path
        self.manifest = manifest
        self.version = version
        self.vocab_size = vocab_size
        self.sha256 = digest
        self._byte_ids = self._build_byte_ids()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_version: str = TOKENIZER_VERSION,
        expected_sha256: str | None = None,
    ) -> OrionTokenizer:
        """Load a production tokenizer only from a complete local release directory."""
        root = Path(directory)
        tokenizer_path = root / "tokenizer.json"
        manifest_path = root / "vocab_manifest.json"
        if not tokenizer_path.is_file() or not manifest_path.is_file():
            raise TokenizerMismatchError(
                f"A trained ORION tokenizer is not installed at {root}; tokenizer training is pending"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerMismatchError(f"Invalid tokenizer manifest at {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise TokenizerMismatchError("Tokenizer manifest must be a JSON object")
        if manifest.get("status") != "trained":
            raise TokenizerMismatchError("Tokenizer manifest does not identify a trained artifact")
        if manifest.get("tokenizer_version") != expected_version:
            raise TokenizerMismatchError(
                f"Expected tokenizer {expected_version}, found {manifest.get('tokenizer_version')}"
            )
        if manifest.get("vocab_size") != VOCAB_SIZE:
            raise TokenizerMismatchError(f"Expected vocabulary size {VOCAB_SIZE}")
        actual_sha256 = sha256_file(tokenizer_path)
        if manifest.get("tokenizer_sha256") != actual_sha256:
            raise TokenizerMismatchError("tokenizer.json SHA-256 does not match its manifest")
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise TokenizerMismatchError("Tokenizer hash does not match the model configuration")
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise TokenizerMismatchError(f"Could not read tokenizer.json: {exc}") from exc
        if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
            raise TokenizerMismatchError("Tokenizer backend vocabulary does not contain 32,768 IDs")
        for token, expected_id in SPECIAL_TOKEN_IDS.items():
            if tokenizer.token_to_id(token) != expected_id:
                raise TokenizerMismatchError(f"Special-token ID mismatch for {token}")
        return cls(tokenizer, path=tokenizer_path, manifest=manifest)

    @classmethod
    def from_backend_for_testing(
        cls,
        tokenizer: Tokenizer,
        *,
        tokenizer_sha256: str = "0" * 64,
    ) -> OrionTokenizer:
        """Construct a test-only wrapper around a synthetic tokenizer backend."""
        manifest: dict[str, object] = {
            "status": "synthetic-test-fixture",
            "tokenizer_version": TOKENIZER_VERSION,
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "tokenizer_sha256": tokenizer_sha256,
        }
        instance = object.__new__(cls)
        instance._tokenizer = tokenizer
        instance.path = Path("<synthetic-test-fixture>")
        instance.manifest = manifest
        instance.version = TOKENIZER_VERSION
        instance.vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        instance.sha256 = tokenizer_sha256
        instance._byte_ids = instance._build_byte_ids()
        return instance

    def _build_byte_ids(self) -> tuple[int, ...]:
        mapping = _bytes_to_unicode()
        ids = tuple(self._tokenizer.token_to_id(mapping[byte]) for byte in range(256))
        if any(token_id is None for token_id in ids):
            raise TokenizerMismatchError("Tokenizer is missing one or more byte-alphabet tokens")
        return tuple(token_id for token_id in ids if token_id is not None)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special_tokens: bool = False,
        max_length: int | None = None,
        truncation: str | None = None,
    ) -> TokenizedText:
        """Encode NFC text. Special-token spellings in untrusted text become byte IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFC", text)
        if not normalized:
            raise ValueError("text must not be empty")
        ids: list[int] = []
        word_ids: list[int | None] = []
        offsets: list[tuple[int, int]] = []
        current_word = 0
        cursor = 0
        for match in _SPECIAL_PATTERN.finditer(normalized):
            if match.start() > cursor:
                current_word = _append_regular_text(
                    self._tokenizer,
                    normalized[cursor : match.start()],
                    base_offset=cursor,
                    current_word=current_word,
                    ids=ids,
                    word_ids=word_ids,
                    offsets=offsets,
                )
            part = match.group(0)
            if allow_special_tokens:
                ids.append(SPECIAL_TOKEN_IDS[part])
                word_ids.append(None)
                offsets.append((match.start(), match.end()))
            else:
                encoded_bytes = part.encode("utf-8")
                ids.extend(self._byte_ids[byte] for byte in encoded_bytes)
                word_ids.extend([current_word] * len(encoded_bytes))
                offsets.extend([(match.start(), match.end())] * len(encoded_bytes))
                current_word += 1
            cursor = match.end()
        if cursor < len(normalized):
            current_word = _append_regular_text(
                self._tokenizer,
                normalized[cursor:],
                base_offset=cursor,
                current_word=current_word,
                ids=ids,
                word_ids=word_ids,
                offsets=offsets,
            )
        if add_bos:
            ids.insert(0, SPECIAL_TOKEN_IDS["<|bos|>"])
            word_ids.insert(0, None)
            offsets.insert(0, (0, 0))
        if add_eos:
            ids.append(SPECIAL_TOKEN_IDS["<|eos|>"])
            word_ids.append(None)
            offsets.append((0, 0))
        if max_length is not None and len(ids) > max_length:
            if truncation is None:
                raise InputLimitError(f"Encoding has {len(ids)} tokens; maximum is {max_length}")
            if truncation == "head":
                ids, word_ids, offsets = (
                    ids[:max_length],
                    word_ids[:max_length],
                    offsets[:max_length],
                )
            elif truncation == "tail":
                ids, word_ids, offsets = (
                    ids[-max_length:],
                    word_ids[-max_length:],
                    offsets[-max_length:],
                )
            else:
                raise ValueError("truncation must be 'head', 'tail', or None")
        return TokenizedText(tuple(ids), (1,) * len(ids), tuple(word_ids), tuple(offsets))

    def encode_batch(
        self,
        texts: list[str] | tuple[str, ...],
        *,
        max_length: int | None = None,
        padding: str = "longest",
        truncation: str | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special_tokens: bool = False,
    ) -> TokenBatch:
        """Encode texts in order and pad with the frozen padding token ID."""
        if not texts:
            raise ValueError("texts must contain at least one string")
        encodings = [
            self.encode(
                text,
                max_length=max_length,
                truncation=truncation,
                add_bos=add_bos,
                add_eos=add_eos,
                allow_special_tokens=allow_special_tokens,
            )
            for text in texts
        ]
        if padding == "longest":
            width = max(len(item.input_ids) for item in encodings)
        elif padding == "max_length" and max_length is not None:
            width = max_length
            if any(len(item.input_ids) > width for item in encodings):
                raise InputLimitError("An encoding is longer than requested max_length")
        elif padding == "none":
            if len({len(item.input_ids) for item in encodings}) != 1:
                raise ValueError("padding='none' requires equal-length encodings")
            width = len(encodings[0].input_ids)
        else:
            raise ValueError("padding must be 'longest', 'max_length', or 'none'")
        pad_id = SPECIAL_TOKEN_IDS["<|pad|>"]
        ids, masks, words = [], [], []
        offsets = []
        for item in encodings:
            amount = width - len(item.input_ids)
            ids.append(item.input_ids + (pad_id,) * amount)
            masks.append(item.attention_mask + (0,) * amount)
            words.append(item.word_ids + (None,) * amount)
            offsets.append(item.offset_mapping + ((0, 0),) * amount)
        return TokenBatch(tuple(ids), tuple(masks), tuple(words), tuple(offsets))

    def encode_with_offsets(
        self,
        text: str,
        *,
        allow_special_tokens: bool = False,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        """Safe encoding with character offsets into NFC-normalized input text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for match in _SPECIAL_PATTERN.finditer(normalized):
            if match.start() > cursor:
                segment = normalized[cursor : match.start()]
                encoded = self._tokenizer.encode(segment, add_special_tokens=False)
                ids.extend(encoded.ids)
                offsets.extend((cursor + start, cursor + end) for start, end in encoded.offsets)
            spelling = match.group(0)
            if allow_special_tokens:
                ids.append(SPECIAL_TOKEN_IDS[spelling])
                offsets.append((match.start(), match.end()))
            else:
                for char_offset, char in enumerate(spelling, start=match.start()):
                    for byte in char.encode("utf-8"):
                        ids.append(self._byte_ids[byte])
                        offsets.append((char_offset, char_offset + 1))
            cursor = match.end()
        if cursor < len(normalized):
            segment = normalized[cursor:]
            encoded = self._tokenizer.encode(segment, add_special_tokens=False)
            ids.extend(encoded.ids)
            offsets.extend((cursor + start, cursor + end) for start, end in encoded.offsets)
        if not normalized:
            raise ValueError("text must not be empty")
        if len(ids) != len(offsets):
            raise RuntimeError("Tokenizer backend returned inconsistent offsets")
        return tuple(ids), tuple(offsets)

    def encode_parts(
        self,
        parts: list[tuple[str, bool]] | tuple[tuple[str, bool], ...],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> TokenizedText:
        """Encode text segments with per-segment trust for template control tokens.

        A true flag is reserved for literals supplied by trusted application code. User
        questions, document labels, and document text must use false so marker-looking
        strings are encoded as ordinary bytes and cannot forge a role or citation token.
        """
        if not parts:
            raise ValueError("parts must not be empty")
        all_ids: list[int] = []
        all_words: list[int | None] = []
        all_offsets: list[tuple[int, int]] = []
        word_offset = 0
        for text, trusted in parts:
            item = self.encode(text, allow_special_tokens=trusted)
            all_ids.extend(item.input_ids)
            all_offsets.extend(item.offset_mapping)
            for word_id in item.word_ids:
                all_words.append(None if word_id is None else word_id + word_offset)
            present = [value for value in item.word_ids if value is not None]
            word_offset += (max(present) + 1) if present else 0
        if add_bos:
            all_ids.insert(0, SPECIAL_TOKEN_IDS["<|bos|>"])
            all_words.insert(0, None)
            all_offsets.insert(0, (0, 0))
        if add_eos:
            all_ids.append(SPECIAL_TOKEN_IDS["<|eos|>"])
            all_words.append(None)
            all_offsets.append((0, 0))
        return TokenizedText(
            tuple(all_ids), (1,) * len(all_ids), tuple(all_words), tuple(all_offsets)
        )

    def decode(
        self,
        ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs, optionally removing active and reserved control tokens."""
        return cast(str, self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens))

    def special_token_id(self, token: str) -> int:
        """Return a frozen special-token ID or report an unknown token spelling."""
        try:
            return SPECIAL_TOKEN_IDS[token]
        except KeyError as exc:
            raise KeyError(f"Unknown ORION v{self.version} special token {token!r}") from exc


def _append_regular_text(
    tokenizer: Tokenizer,
    text: str,
    *,
    base_offset: int,
    current_word: int,
    ids: list[int],
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
) -> int:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids.extend(encoding.ids)
    local_ids: dict[int, int] = {}
    for word_id, (start, end) in zip(encoding.word_ids, encoding.offsets, strict=True):
        if word_id is None:
            word_ids.append(None)
        else:
            if word_id not in local_ids:
                local_ids[word_id] = current_word
                current_word += 1
            word_ids.append(local_ids[word_id])
        offsets.append((base_offset + start, base_offset + end))
    return current_word
