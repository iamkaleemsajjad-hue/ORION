"""Map-style datasets over local, checksummed uint16 token shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from orion_core.hashing import sha256_file


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Read fixed contiguous token blocks from an explicit local shard."""

    def __init__(
        self,
        path: str | Path,
        *,
        sequence_length: int,
        expected_sha256: str | None = None,
        word_ids_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Token shard does not exist: {self.path}")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two tokens")
        actual = sha256_file(self.path)
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ValueError("Token shard SHA-256 does not match the supplied manifest")
        byte_length = self.path.stat().st_size
        if byte_length % 2:
            raise ValueError("uint16 token shard has an odd byte length")
        self.token_count = byte_length // 2
        self.sequence_length = sequence_length
        self.sha256 = actual
        self._tokens = np.memmap(self.path, dtype="<u2", mode="r")
        self.word_ids_path = Path(word_ids_path) if word_ids_path is not None else None
        self._word_ids = None
        self.word_ids_sha256: str | None = None
        if self.word_ids_path is not None:
            if not self.word_ids_path.is_file():
                raise FileNotFoundError(
                    f"Whole-word alignment shard does not exist: {self.word_ids_path}"
                )
            if self.word_ids_path.stat().st_size != self.token_count * 4:
                raise ValueError("word ID shard must contain one int32 value per token ID")
            self.word_ids_sha256 = sha256_file(self.word_ids_path)
            self._word_ids = np.memmap(self.word_ids_path, dtype="<i4", mode="r")
        self.sample_count = self.token_count // sequence_length
        if self.sample_count == 0:
            raise ValueError("Token shard is shorter than one complete sequence block")

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        start = index * self.sequence_length
        # Copy prevents Torch from sharing a read-only numpy mapping.
        ids = np.asarray(self._tokens[start : start + self.sequence_length], dtype=np.int64).copy()
        input_ids = torch.from_numpy(ids)
        result = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }
        if self._word_ids is not None:
            values = np.asarray(
                self._word_ids[start : start + self.sequence_length], dtype=np.int64
            ).copy()
            result["word_ids"] = torch.from_numpy(values)
        return result


__all__ = ["TokenBlockDataset"]
