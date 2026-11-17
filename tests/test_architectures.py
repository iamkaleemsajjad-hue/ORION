from __future__ import annotations

import copy
from typing import cast

import pytest
import torch

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import Orion100MEncoder
from orion_100m.model.pretraining import Orion100MRTDPretrainer
from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.model.encoder_decoder import (
    DecoderBlock as EncoderDecoderBlock,
)
from orion_300m.model.encoder_decoder import (
    Orion300MEncoderDecoderPilot,
)
from orion_core.blocks.rotary import RotaryEmbedding, apply_rotary


def test_planned_default_body_parameter_counts_on_meta_device() -> None:
    with torch.device("meta"):
        encoder = Orion100MEncoder(Orion100MConfig())
        decoder = Orion300MDecoder(Orion300MConfig())
    assert encoder.parameter_count() == 100_291_840
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 306_763_200
    assert decoder.lm_head.weight is decoder.token_embeddings.weight


def test_decoder_rejects_out_of_vocabulary_generation_control_ids() -> None:
    with pytest.raises(ValueError, match="special token IDs"):
        Orion300MConfig(vocab_size=512, end_token_id=512)


def _small_encoder_config() -> Orion100MConfig:
    return Orion100MConfig(
        vocab_size=512,
        num_layers=2,
        hidden_size=16,
        num_heads=4,
        intermediate_size=24,
        max_sequence_length=16,
        base_training_length=8,
        local_window=4,
        global_attention_every=2,
    )


def test_encoder_forward_backward_and_padding_isolation() -> None:
    torch.manual_seed(11)
    model = Orion100MEncoder(_small_encoder_config())
    ids = torch.tensor([[2, 20, 21, 0], [2, 22, 23, 0]])
    mask = ids.ne(0)
    docs = torch.tensor([[0, 0, 1, -1], [2, 2, 2, -1]])
    output = model(ids, attention_mask=mask, document_ids=docs, output_hidden_states=True)
    assert output.last_hidden_state.shape == (2, 4, 16)
    assert output.pooled_output.shape == (2, 16)
    assert len(output.hidden_states or ()) == 3
    output.pooled_output.square().mean().backward()
    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()
    short = model(ids[:1, :3], attention_mask=torch.ones((1, 3), dtype=torch.bool))
    padded = model(
        ids[:1],
        attention_mask=torch.tensor([[True, True, True, False]]),
    )
    torch.testing.assert_close(short.pooled_output, padded.pooled_output)
    torch.testing.assert_close(short.last_hidden_state, padded.last_hidden_state[:, :3])


def test_encoder_activation_checkpointing_preserves_dropout_outputs_and_gradients() -> None:
    config = _small_encoder_config().model_copy(update={"dropout": 0.15})
    torch.manual_seed(111)
    reference = Orion100MEncoder(config).train()
    checkpointed = copy.deepcopy(reference).train()
    checkpointed.gradient_checkpointing_enable()
    ids = torch.tensor([[2, 300, 301, 302]])
    mask = ids.ne(0)

    torch.manual_seed(222)
    reference_output = reference(ids, attention_mask=mask)
    reference_output.pooled_output.square().mean().backward()
    expected_rng = torch.get_rng_state()

    torch.manual_seed(222)
    checkpointed_output = checkpointed(ids, attention_mask=mask)
    checkpointed_output.pooled_output.square().mean().backward()

    torch.testing.assert_close(
        checkpointed_output.last_hidden_state, reference_output.last_hidden_state
    )
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)


def test_rotary_embedding_matches_pairwise_rotation() -> None:
    rotary = RotaryEmbedding(head_dim=4, theta=10000.0)
    positions = torch.tensor([[1]])
    cosine, sine = rotary(positions, dtype=torch.float32)
    vector = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]])
    result = apply_rotary(vector, cosine, sine)
    expected = torch.tensor(
        [
            [
                [
                    [
                        torch.cos(torch.tensor(1.0)),
                        torch.sin(torch.tensor(1.0)),
                        -torch.sin(torch.tensor(0.01)),
                        torch.cos(torch.tensor(0.01)),
                    ]
                ]
            ]
        ]
    )
    torch.testing.assert_close(result, expected)


def test_rtd_objective_has_real_generator_and_discriminator_gradients() -> None:
    torch.manual_seed(7)
    model = Orion100MRTDPretrainer(
        _small_encoder_config(),
        generator_layers=1,
        generator_hidden_size=8,
        generator_heads=2,
        generator_intermediate_size=12,
        mask_rate=0.5,
    )
    ids = torch.tensor([[2, 300, 301, 302, 3]])
    result = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.generator.input_projection.weight.grad is not None
    assert model.discriminator_head.output.weight.grad is not None


def _small_decoder_config() -> Orion300MConfig:
    return Orion300MConfig(
        vocab_size=512,
        num_layers=2,
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=24,
        pretraining_length=4,
        context_length=12,
        max_generation_tokens=4,
    )


def test_decoder_causal_mask_cache_equivalence_and_backward() -> None:
    torch.manual_seed(19)
    model = Orion300MDecoder(_small_decoder_config()).eval()
    ids = torch.tensor([[5, 300, 301, 302]])
    full = model(ids)
    cache = None
    incremental = []
    for position in range(ids.shape[1]):
        output = model(ids[:, position : position + 1], cache=cache, use_cache=True)
        cache = output.cache
        incremental.append(output.logits)
    cached = torch.cat(incremental, dim=1)
    torch.testing.assert_close(cached, full.logits, rtol=2e-5, atol=2e-5)
    changed = ids.clone()
    changed[:, -1] = 400
    changed_prefix = model(changed).logits[:, :3]
    torch.testing.assert_close(changed_prefix, full.logits[:, :3], rtol=2e-5, atol=2e-5)
    loss, _ = model(ids, labels=ids).loss, model(ids, labels=ids).z_loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert model.lm_head.weight is model.token_embeddings.weight


def test_decoder_activation_checkpointing_preserves_gradients_and_rejects_kv_cache() -> None:
    config = _small_decoder_config().model_copy(update={"dropout": 0.15})
    torch.manual_seed(333)
    reference = Orion300MDecoder(config).train()
    checkpointed = copy.deepcopy(reference).train()
    checkpointed.gradient_checkpointing_enable()
    ids = torch.tensor([[5, 300, 301, 302]])

    torch.manual_seed(444)
    reference_output = reference(ids, labels=ids)
    assert reference_output.loss is not None
    reference_output.loss.backward()
    expected_rng = torch.get_rng_state()

    torch.manual_seed(444)
    checkpointed_output = checkpointed(ids, labels=ids)
    assert checkpointed_output.loss is not None
    checkpointed_output.loss.backward()

    torch.testing.assert_close(checkpointed_output.logits, reference_output.logits)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="cannot be combined with a KV cache"):
        checkpointed(ids, use_cache=True)
    checkpointed.eval()
    assert checkpointed(ids, use_cache=True).cache is not None


def test_prefix_lm_allows_bidirectional_prefix_and_causal_targets() -> None:
    torch.manual_seed(31)
    model = Orion300MDecoder(_small_decoder_config()).eval()
    tokens = torch.tensor([[5, 300, 301, 302]])
    prefix = torch.tensor([2])
    output = model(tokens, prefix_lengths=prefix)

    changed_target_future = tokens.clone()
    changed_target_future[0, 3] = 401
    future_output = model(changed_target_future, prefix_lengths=prefix)
    torch.testing.assert_close(
        output.last_hidden_state[:, :3], future_output.last_hidden_state[:, :3]
    )

    changed_prefix = tokens.clone()
    changed_prefix[0, 1] = 402
    prefix_output = model(changed_prefix, prefix_lengths=prefix)
    assert not torch.allclose(
        output.last_hidden_state[:, :1], prefix_output.last_hidden_state[:, :1]
    )


def test_encoder_decoder_pilot_cross_attention_forward_and_backward() -> None:
    torch.manual_seed(37)
    config = _small_decoder_config()
    reference = Orion300MEncoderDecoderPilot(config, encoder_layers=1, decoder_layers=1)
    model = copy.deepcopy(reference)
    model.gradient_checkpointing_enable()
    source = torch.tensor([[5, 300, 0]])
    source_mask = source.ne(0)
    decoder_input = torch.tensor([[5, 301, 0]])
    target_mask = decoder_input.ne(0)
    labels = torch.tensor([[301, 6, -100]])
    reference_output = reference(
        source,
        decoder_input,
        source_attention_mask=source_mask,
        target_attention_mask=target_mask,
        labels=labels,
    )
    output = model(
        source,
        decoder_input,
        source_attention_mask=source_mask,
        target_attention_mask=target_mask,
        labels=labels,
    )
    assert output.encoder_last_hidden_state.shape == (1, 3, config.hidden_size)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert reference_output.loss is not None
    torch.testing.assert_close(output.logits, reference_output.logits)
    output.loss.backward()
    reference_output.loss.backward()
    assert model.token_embeddings.weight.grad is not None
    decoder_layer = cast(EncoderDecoderBlock, model.decoder[0])
    assert decoder_layer.cross_attention.query.weight.grad is not None
    assert model.lm_head.weight is model.token_embeddings.weight
    for expected, actual in zip(reference.parameters(), model.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
