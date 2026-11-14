"""Compatibility exports for the canonical ORION tokenizer-v1 specification."""

from orion_tokenizer.spec import (
    BYTE_TOKEN_END,
    BYTE_TOKEN_START,
    SPECIAL_TOKENS,
)

BYTE_START = BYTE_TOKEN_START
BYTE_END = BYTE_TOKEN_END + 1
MERGE_START = BYTE_END
BYTE_TOKENS = tuple(range(BYTE_START, BYTE_END))
PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, ABSTAIN_TOKEN = (
    SPECIAL_TOKENS[:8]
)
