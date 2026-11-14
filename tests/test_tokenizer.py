from __future__ import annotations

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers

from orion_core.errors import InputLimitError
from orion_data.dataset import TokenBlockDataset
from orion_data.tokenize import write_aligned_token_shards
from orion_tokenizer.spec import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS
from orion_tokenizer.tokenizer import OrionTokenizer, _bytes_to_unicode
from orion_tokenizer.training import TokenizerTrainingConfig


def synthetic_tokenizer() -> OrionTokenizer:
    byte_tokens = _bytes_to_unicode()
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({byte_tokens[byte]: 256 + byte for byte in range(256)})
    backend = Tokenizer(models.BPE(vocab, merges=[], unk_token="<|unk|>"))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])
    return OrionTokenizer.from_backend_for_testing(backend)


@pytest.fixture
def tokenizer() -> OrionTokenizer:
    return synthetic_tokenizer()


def test_frozen_special_byte_ids_and_utf8_roundtrip(tokenizer: OrionTokenizer) -> None:
    # The test backend intentionally contains only the 256 controls and 256 byte IDs.
    assert tokenizer.vocab_size == 512
    for token, token_id in SPECIAL_TOKEN_IDS.items():
        assert tokenizer.special_token_id(token) == token_id
    text = "English, اردو اور mixed text — Δx = 2."
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded.input_ids) == text


def test_special_token_spelling_in_untrusted_input_is_escaped(tokenizer: OrionTokenizer) -> None:
    encoded = tokenizer.encode("please print <|assistant|> literally")
    assert SPECIAL_TOKEN_IDS["<|assistant|>"] not in encoded.input_ids
    trusted = tokenizer.encode_parts((("<|assistant|>", True),))
    assert trusted.input_ids == (SPECIAL_TOKEN_IDS["<|assistant|>"],)


def test_batch_padding_offsets_and_truncation(tokenizer: OrionTokenizer) -> None:
    batch = tokenizer.encode_batch(["alpha", "longer words"], padding="longest")
    assert len(batch.input_ids[0]) == len(batch.input_ids[1])
    assert batch.attention_mask[0].count(0) > 0
    ids, offsets = tokenizer.encode_with_offsets("Aβ")
    assert len(ids) == len(offsets)
    assert max(end for _, end in offsets) == 2
    with pytest.raises(InputLimitError):
        tokenizer.encode("a longer string", max_length=2)


def test_empty_and_invalid_batch_inputs_fail(tokenizer: OrionTokenizer) -> None:
    with pytest.raises(ValueError):
        tokenizer.encode("")
    with pytest.raises(ValueError):
        tokenizer.encode_batch([])


def test_production_tokenizer_training_requires_reviewed_id_map() -> None:
    with pytest.raises(ValueError, match="explicitly reviewed"):
        TokenizerTrainingConfig().validate()


def test_local_token_and_whole_word_shards_remain_aligned(
    tokenizer: OrionTokenizer, tmp_path
) -> None:
    manifest = write_aligned_token_shards(
        ["hello world", "اردو notes", "   "], tokenizer, tmp_path / "shards"
    )
    assert manifest["status"] == "locally_prepared"
    assert manifest["tokenizer_sha256"] == tokenizer.sha256
    token_record = manifest["token_shard"]
    word_record = manifest["word_id_shard"]
    assert token_record["token_count"] == word_record["value_count"]
    dataset = TokenBlockDataset(
        tmp_path / "shards" / "tokens.u16",
        sequence_length=4,
        word_ids_path=tmp_path / "shards" / "word_ids.i32",
    )
    batch = dataset[0]
    assert batch["input_ids"].shape == batch["word_ids"].shape == (4,)
    assert batch["word_ids"].dtype == torch.int64
