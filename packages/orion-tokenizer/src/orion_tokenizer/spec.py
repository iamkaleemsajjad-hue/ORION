"""Frozen token ID ranges from the plan; undocumented names are provisional."""

from __future__ import annotations

TOKENIZER_VERSION = "1.0"
VOCAB_SIZE = 32768
BYTE_TOKEN_START = 256
BYTE_TOKEN_END = 511
MERGE_TOKEN_START = 512
SPECIAL_BLOCK_END = 255

# The plan's 0-7 and 16-31 token text was lost in the supplied DOCX table.
# Stable model-required tokens are placed first; unspecified entries stay reserved.
_SPECIALS: dict[int, str] = {
    0: "<|pad|>",
    1: "<|unk|>",
    2: "<|cls|>",
    3: "<|sep|>",
    4: "<|mask|>",
    5: "<|bos|>",
    6: "<|eos|>",
    7: "<|abstain|>",
    8: "<meta:filename>",
    9: "<meta:title>",
    10: "<meta:heading>",
    11: "<meta:text>",
    12: "<meta:ocr_low>",
    13: "<meta:ext>",
    14: "<meta:lang>",
    15: "<field_sep>",
    16: "<|system|>",
    17: "<|user|>",
    18: "<|assistant|>",
    19: "<|end|>",
    20: "<|context|>",
    21: "<|/context|>",
    22: "<|json|>",
    23: "<|/json|>",
}
for _index in range(24, 32):
    _SPECIALS[_index] = f"<|reserved_{_index:03d}|>"
for _index in range(32, 48):
    _SPECIALS[_index] = f"<|doc_{_index - 31}|>"
for _index in range(48, 64):
    _SPECIALS[_index] = f"<|cite_{_index - 47}|>"
for _index in range(64, 256):
    _SPECIALS[_index] = f"<|reserved_{_index:03d}|>"

SPECIAL_TOKEN_IDS = {token: index for index, token in _SPECIALS.items()}
SPECIAL_ID_TO_TOKEN = dict(_SPECIALS)
SPECIAL_TOKENS = tuple(_SPECIALS[index] for index in range(256))
ACTIVE_SPECIAL_TOKENS = frozenset(
    token for token in SPECIAL_TOKENS if not token.startswith("<|reserved_")
)
DECODER_FORBIDDEN_TOKEN_IDS = frozenset(
    SPECIAL_TOKEN_IDS[token]
    for token in SPECIAL_TOKENS
    if token
    not in {
        "<|eos|>",
        "<|abstain|>",
        "<|end|>",
        *(f"<|cite_{index}|>" for index in range(1, 17)),
    }
)

if len(SPECIAL_TOKEN_IDS) != 256 or set(SPECIAL_TOKEN_IDS.values()) != set(range(256)):
    raise RuntimeError("ORION special-token table must cover IDs 0 through 255 exactly once")
