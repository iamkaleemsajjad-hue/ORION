"""Atomic JSONL and uint16 token shard writing/reading."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: str | Path) -> Iterable[Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_token_shard(path: str | Path, token_ids: Iterable[int]) -> dict[str, int | str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.fromiter(token_ids, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() > 65535):
        raise ValueError("uint16 token shards require IDs in [0, 65535]")
    little_endian = values.astype("<u2", copy=False)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        little_endian.tofile(temp_name)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "dtype": "uint16-le",
        "token_count": int(values.size),
        "bytes": path.stat().st_size,
    }


def write_word_id_shard(path: str | Path, word_ids: Iterable[int]) -> dict[str, int | str]:
    """Write int32 whole-word alignment IDs; -1 marks special/non-word tokens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.fromiter(word_ids, dtype=np.int64)
    if values.size and (values.min() < -1 or values.max() > 2**31 - 1):
        raise ValueError("word IDs must fit signed int32 and use -1 for no word")
    little_endian = values.astype("<i4", copy=False)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        little_endian.tofile(temp_name)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "dtype": "int32-le",
        "value_count": int(values.size),
        "bytes": path.stat().st_size,
    }


def read_token_shard(
    path: str | Path, *, offset: int = 0, count: int | None = None
) -> NDArray[np.uint16]:
    path = Path(path)
    byte_count = path.stat().st_size
    if byte_count % 2:
        raise ValueError("uint16 token shard has an odd byte length")
    total = byte_count // 2
    if offset < 0 or offset > total:
        raise ValueError("offset is outside token shard")
    length = total - offset if count is None else count
    if length < 0 or offset + length > total:
        raise ValueError("requested token range is outside shard")
    return np.memmap(path, dtype="<u2", mode="r", offset=offset * 2, shape=(length,))
