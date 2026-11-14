"""Deterministic random sampling with a resumable cursor."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sized
from typing import Any

from torch.utils.data import Sampler


class StatefulRandomSampler(Sampler[int]):
    """Epoch-wise permutation sampler whose state can be saved as JSON values."""

    def __init__(self, data_source: Sized, *, seed: int = 1729) -> None:
        if len(data_source) <= 0:
            raise ValueError("data_source must not be empty")
        self.size = len(data_source)
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._indices: list[int] = []
        self._make_order()

    def _make_order(self) -> None:
        generator = random.Random(self.seed + self.epoch)
        self._indices = list(range(self.size))
        generator.shuffle(self._indices)
        self.cursor = 0

    def __iter__(self) -> Iterator[int]:
        while self.cursor < self.size:
            value = self._indices[self.cursor]
            self.cursor += 1
            yield value

    def __len__(self) -> int:
        return self.size - self.cursor

    def next_epoch(self) -> None:
        self.epoch += 1
        self._make_order()

    def state_dict(self) -> dict[str, Any]:
        return {"size": self.size, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("size") != self.size or state.get("seed") != self.seed:
            raise ValueError("sampler state does not match dataset size and seed")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self.size:
            raise ValueError("invalid sampler epoch or cursor")
        self.epoch = epoch
        self._make_order()
        self.cursor = cursor


class StatefulRandomBatchSampler(Sampler[list[int]]):
    """Batch-level resumable permutation that drops only incomplete epoch tails."""

    def __init__(self, data_source: Sized, *, batch_size: int, seed: int = 1729) -> None:
        if len(data_source) <= 0 or batch_size <= 0:
            raise ValueError("data_source and batch_size must be positive")
        self.size = len(data_source)
        self.batch_size = batch_size
        self.usable_size = self.size - (self.size % self.batch_size)
        if self.usable_size == 0:
            raise ValueError("data_source must contain at least one complete batch")
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._indices: list[int] = []
        self._make_order()

    def _make_order(self) -> None:
        generator = random.Random(self.seed + self.epoch)
        self._indices = list(range(self.size))
        generator.shuffle(self._indices)
        self.cursor = 0

    def __iter__(self) -> Iterator[list[int]]:
        while self.cursor < self.usable_size:
            start = self.cursor
            self.cursor += self.batch_size
            yield self._indices[start : self.cursor]

    def __len__(self) -> int:
        return (self.usable_size - self.cursor) // self.batch_size

    def next_epoch(self) -> None:
        self.epoch += 1
        self._make_order()

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("size") != self.size
            or state.get("batch_size") != self.batch_size
            or state.get("seed") != self.seed
        ):
            raise ValueError("batch sampler state does not match dataset, batch size, or seed")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self.usable_size or cursor % self.batch_size:
            raise ValueError("invalid batch sampler epoch or cursor")
        self.epoch = epoch
        self._make_order()
        self.cursor = cursor
