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
        self.rank = 0
        self.world_size = 1
        self._local_size = self.size
        self._indices: list[int] = []
        self._make_order()

    def _make_order(self) -> None:
        generator = random.Random(self.seed + self.epoch)
        self._indices = list(range(self.size))
        generator.shuffle(self._indices)
        usable_size = self.size - (self.size % self.world_size)
        self._indices = self._indices[self.rank : usable_size : self.world_size]
        self._local_size = len(self._indices)
        self.cursor = 0

    def configure_distributed(self, rank: int, world_size: int) -> None:
        """Shard one deterministic global permutation evenly across DDP ranks."""
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("distributed sampler requires 0 <= rank < world_size")
        if (rank, world_size) == (self.rank, self.world_size):
            return
        if self.cursor:
            raise ValueError("distributed sampling must be configured before consuming batches")
        if self.size < world_size:
            raise ValueError("dataset must contain at least one item per distributed rank")
        self.rank = rank
        self.world_size = world_size
        self._make_order()

    def __iter__(self) -> Iterator[int]:
        while self.cursor < self._local_size:
            value = self._indices[self.cursor]
            self.cursor += 1
            yield value

    def __len__(self) -> int:
        return self._local_size - self.cursor

    def next_epoch(self) -> None:
        self.epoch += 1
        self._make_order()

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("size") != self.size or state.get("seed") != self.seed:
            raise ValueError("sampler state does not match dataset size and seed")
        rank, world_size = int(state.get("rank", 0)), int(state.get("world_size", 1))
        if (rank, world_size) != (self.rank, self.world_size):
            raise ValueError("sampler state does not match distributed rank and world size")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self._local_size:
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
        self.rank = 0
        self.world_size = 1
        self._local_size = self.usable_size
        self._indices: list[int] = []
        self._make_order()

    def _make_order(self) -> None:
        generator = random.Random(self.seed + self.epoch)
        self._indices = list(range(self.size))
        generator.shuffle(self._indices)
        global_usable_size = self.size - (self.size % (self.batch_size * self.world_size))
        self._indices = self._indices[self.rank : global_usable_size : self.world_size]
        self._local_size = len(self._indices)
        self.cursor = 0

    def configure_distributed(self, rank: int, world_size: int) -> None:
        """Shard samples so every rank yields the same number of full local batches."""
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("distributed sampler requires 0 <= rank < world_size")
        if (rank, world_size) == (self.rank, self.world_size):
            return
        if self.cursor:
            raise ValueError("distributed sampling must be configured before consuming batches")
        if self.size < self.batch_size * world_size:
            raise ValueError("dataset must contain one full batch per distributed rank")
        self.rank = rank
        self.world_size = world_size
        self._make_order()

    def __iter__(self) -> Iterator[list[int]]:
        while self.cursor < self._local_size:
            start = self.cursor
            self.cursor += self.batch_size
            yield self._indices[start : self.cursor]

    def __len__(self) -> int:
        return (self._local_size - self.cursor) // self.batch_size

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
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("size") != self.size
            or state.get("batch_size") != self.batch_size
            or state.get("seed") != self.seed
        ):
            raise ValueError("batch sampler state does not match dataset, batch size, or seed")
        rank, world_size = int(state.get("rank", 0)), int(state.get("world_size", 1))
        if (rank, world_size) != (self.rank, self.world_size):
            raise ValueError("batch sampler state does not match distributed rank and world size")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self._local_size or cursor % self.batch_size:
            raise ValueError("invalid batch sampler epoch or cursor")
        self.epoch = epoch
        self._make_order()
        self.cursor = cursor
