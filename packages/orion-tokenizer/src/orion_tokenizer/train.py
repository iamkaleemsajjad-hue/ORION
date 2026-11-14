"""Backward-compatible import path for the canonical local-only trainer."""

from orion_tokenizer.training import TokenizerTrainingConfig, iter_local_text, train_tokenizer

__all__ = ["TokenizerTrainingConfig", "iter_local_text", "train_tokenizer"]
