"""Preallocated inference KV cache for decoder-only generation."""

from orion_core.blocks.attention import StaticKVCache


class KVCache(StaticKVCache):
    """Public ORION decoder cache with a fixed maximum sequence and batch size."""
