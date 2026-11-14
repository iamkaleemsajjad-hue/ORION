"""Packed uint16 token shards and document-boundary metadata."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PackedSequence:
    input_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


def pack_documents(
    documents: Iterable[tuple[str, Iterable[int]]],
    *,
    sequence_length: int,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """Greedily pack document token IDs and mark boundaries for block attention masks."""
    if sequence_length <= 0 or not 0 <= pad_token_id <= 65535:
        raise ValueError("invalid sequence length or pad token ID")
    packed: list[PackedSequence] = []
    ids: list[int] = []
    doc_ids: list[int] = []

    def flush() -> None:
        if not ids:
            return
        padding = sequence_length - len(ids)
        packed.append(
            PackedSequence(
                tuple(ids + [pad_token_id] * padding),
                tuple(doc_ids + [-1] * padding),
                tuple([1] * len(ids) + [0] * padding),
            )
        )
        ids.clear()
        doc_ids.clear()

    for next_document, (document_id, token_ids) in enumerate(documents):
        if not document_id:
            raise ValueError("document IDs must be non-empty")
        tokens = [int(token) for token in token_ids]
        if any(token < 0 or token > 65535 for token in tokens):
            raise ValueError(f"token IDs for {document_id} do not fit uint16")
        offset = 0
        while offset < len(tokens):
            available = sequence_length - len(ids)
            if available == 0:
                flush()
                available = sequence_length
            chunk = tokens[offset : offset + available]
            ids.extend(chunk)
            doc_ids.extend([next_document] * len(chunk))
            offset += len(chunk)
            if offset < len(tokens):
                flush()
    flush()
    return packed


def write_memmap_shard(sequences: list[PackedSequence], output_dir: str | Path) -> Path:
    """Write typed NumPy memmaps and a JSON index with checksums and shape metadata."""
    if not sequences:
        raise ValueError("cannot write an empty shard")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows, width = len(sequences), len(sequences[0].input_ids)
    if any(len(item.input_ids) != width for item in sequences):
        raise ValueError("all packed sequences must have equal lengths")
    arrays = {
        "input_ids": np.memmap(
            root / "input_ids.u16", dtype=np.uint16, mode="w+", shape=(rows, width)
        ),
        "document_ids": np.memmap(
            root / "document_ids.i32", dtype=np.int32, mode="w+", shape=(rows, width)
        ),
        "attention_mask": np.memmap(
            root / "attention_mask.u8", dtype=np.uint8, mode="w+", shape=(rows, width)
        ),
    }
    for index, sequence in enumerate(sequences):
        arrays["input_ids"][index] = sequence.input_ids
        arrays["document_ids"][index] = sequence.document_ids
        arrays["attention_mask"][index] = sequence.attention_mask
    for array in arrays.values():
        array.flush()
    from orion_core.hashing import sha256_file

    metadata = {
        "storage": "numpy-memmap",
        "dtype": {"input_ids": "uint16", "document_ids": "int32", "attention_mask": "uint8"},
        "shape": [rows, width],
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in (
                ("input_ids", root / "input_ids.u16"),
                ("document_ids", root / "document_ids.i32"),
                ("attention_mask", root / "attention_mask.u8"),
            )
        },
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def write_parquet_records(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    """Write columnar Parquet when the optional pyarrow extra is installed."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires `pip install orion-data[parquet]`") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        raise ValueError("cannot write an empty Parquet shard")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, destination, compression="zstd", use_dictionary=True)
    return destination
