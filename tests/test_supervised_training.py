from __future__ import annotations

import json

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers

from orion_100m.config import Orion100MConfig
from orion_100m.contrastive import ContrastiveTrainingConfig, train_contrastive
from orion_100m.course_matching import (
    CourseMatchingTrainingConfig,
    evaluate_course_matching,
    train_course_matching,
)
from orion_100m.finetuning import FineTuningConfig, train_multitask
from orion_100m.model.document_model import Orion100MDocumentModel, TaskLabelMap
from orion_100m.model.losses import course_open_set_loss
from orion_100m.model.pretraining import Orion100MRTDPretrainer
from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder
from orion_300m.sft import (
    SFTTrainingConfig,
    render_chat_messages,
    train_sft_jsonl,
)
from orion_core.errors import ArtifactIntegrityError
from orion_tokenizer.spec import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS
from orion_tokenizer.tokenizer import OrionTokenizer, _bytes_to_unicode
from orion_train.checkpoint import CheckpointManager, load_checkpoint_model_weights
from orion_train.config import TrainerConfig


def _small_tokenizer() -> OrionTokenizer:
    byte_tokens = _bytes_to_unicode()
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({byte_tokens[byte]: 256 + byte for byte in range(256)})
    backend = Tokenizer(models.BPE(vocab, merges=[], unk_token="<|unk|>"))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])
    return OrionTokenizer.from_backend_for_testing(backend)


def test_sft_renderer_marks_only_assistant_content_and_end_tokens() -> None:
    tokenizer = _small_tokenizer()
    result = render_chat_messages(
        [
            {"role": "user", "content": "Explain <|assistant|> literally"},
            {"role": "assistant", "content": "Evidence based answer."},
        ],
        tokenizer,
        max_length=512,
    )
    assert result["input_ids"][0] == SPECIAL_TOKEN_IDS["<|bos|>"]
    assert SPECIAL_TOKEN_IDS["<|assistant|>"] in result["input_ids"]
    assert result["input_ids"].count(SPECIAL_TOKEN_IDS["<|assistant|>"]) == 1
    assert result["input_ids"][-1] == SPECIAL_TOKEN_IDS["<|end|>"]
    assistant_indexes = [index for index, value in enumerate(result["assistant_mask"]) if value]
    assert assistant_indexes
    assert assistant_indexes[0] > result["input_ids"].index(SPECIAL_TOKEN_IDS["<|assistant|>"])
    assert all(
        not result["assistant_mask"][index]
        for index in range(
            result["input_ids"].index(SPECIAL_TOKEN_IDS["<|user|>"]), assistant_indexes[0]
        )
    )
    with pytest.raises(ValueError, match="assistant response"):
        render_chat_messages([{"role": "user", "content": "Only user"}], tokenizer, max_length=64)


def test_multitask_finetuning_runs_and_checkpoints_one_synthetic_step(tmp_path) -> None:
    data = tmp_path / "documents.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"input_ids": [2, token, 3], "subject": label})
            for token, label in ((300, 0), (301, 1))
        )
        + "\n",
        encoding="utf-8",
    )
    labels = TaskLabelMap(subjects=("math", "physics"), document_types=("notes", "slides"))
    config = FineTuningConfig(
        model=Orion100MConfig(
            vocab_size=512,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=24,
            max_sequence_length=8,
            base_training_length=4,
            local_window=4,
            global_attention_every=1,
        ),
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1, log_every_steps=1),
        sequence_length=4,
        batch_size=2,
        warmup_steps=0,
    )
    pretrainer = Orion100MRTDPretrainer(
        config.model,
        generator_layers=1,
        generator_hidden_size=8,
        generator_heads=2,
        generator_intermediate_size=12,
    )
    pretrain_optimizer = torch.optim.SGD(pretrainer.parameters(), lr=0.01)
    initial_checkpoint = CheckpointManager(tmp_path / "rtd-checkpoint")
    initial_checkpoint.save(
        model=pretrainer,
        optimizer=pretrain_optimizer,
        scheduler=None,
        step=0,
        metadata={
            "run_identity": {
                "model": "orion-100m",
                "training_config": {"model": config.model.model_dump(mode="json")},
            }
        },
    )
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        load_checkpoint_model_weights(
            tmp_path / "rtd-checkpoint",
            expected_model="orion-300m",
            expected_model_config=config.model.model_dump(mode="json"),
        )
    result = train_multitask(
        data,
        tmp_path / "run",
        labels,
        config,
        initial_checkpoint=tmp_path / "rtd-checkpoint",
        device="cpu",
    )
    assert result["objective"] == "multitask_document_finetuning"
    assert result["step"] == 1
    assert torch.isfinite(torch.tensor(result["observed_losses"])).all()
    assert (tmp_path / "run" / "checkpoints" / "latest.json").is_file()


def test_matryoshka_contrastive_training_runs_one_synthetic_step(tmp_path) -> None:
    data = tmp_path / "pairs.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "anchor_ids": [2, anchor, 3],
                    "positive_ids": [2, positive, 3],
                }
            )
            for anchor, positive in ((300, 310), (301, 311), (302, 312))
        )
        + "\n",
        encoding="utf-8",
    )
    model_config = Orion100MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=16,
        num_heads=4,
        intermediate_size=24,
        max_sequence_length=8,
        base_training_length=4,
        local_window=4,
        global_attention_every=1,
    )
    config = ContrastiveTrainingConfig(
        model=model_config,
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1, log_every_steps=1),
        sequence_length=4,
        batch_size=2,
        warmup_steps=0,
        dimensions=(16, 8, 4),
        dimension_weights=(1.0, 0.5, 0.25),
    )
    pretrainer = Orion100MRTDPretrainer(
        model_config,
        generator_layers=1,
        generator_hidden_size=8,
        generator_heads=2,
        generator_intermediate_size=12,
    )
    source = CheckpointManager(tmp_path / "rtd-checkpoint")
    source.save(
        model=pretrainer,
        optimizer=torch.optim.SGD(pretrainer.parameters(), lr=0.01),
        scheduler=None,
        step=0,
        metadata={
            "run_identity": {
                "model": "orion-100m",
                "training_config": {"model": model_config.model_dump(mode="json")},
            }
        },
    )
    result = train_contrastive(
        data,
        tmp_path / "contrastive-run",
        config,
        initial_checkpoint=tmp_path / "rtd-checkpoint",
        device="cpu",
    )
    assert result["objective"] == "matryoshka_contrastive"
    assert result["step"] == 1
    assert torch.isfinite(torch.tensor(result["observed_losses"])).all()


def test_course_open_set_loss_trains_known_and_unknown_examples() -> None:
    logits = torch.tensor([[4.0, 0.2], [2.5, 0.1]], requires_grad=True)
    cosine = torch.tensor([[0.8, 0.1], [0.7, 0.2]], requires_grad=True)
    loss, parts = course_open_set_loss(
        logits,
        torch.tensor([0, -1]),
        cosine_scores=cosine,
        unknown_margin=0.3,
        unknown_weight=2.0,
    )
    loss.backward()
    assert set(parts) == {"known_course", "unknown_margin"}
    assert torch.isfinite(loss)
    assert logits.grad is not None and logits.grad[0].abs().sum() > 0
    assert cosine.grad is not None and cosine.grad[1, 0] > 0


def test_course_matching_training_runs_known_unknown_step_and_checkpoints(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "courses": [
                    {"course_id": "MATH-101", "input_ids": [2, 300, 3]},
                    {"course_id": "PHYS-101", "input_ids": [2, 301, 3]},
                ]
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "course-documents.jsonl"
    data.write_text(
        "\n".join(
            (
                json.dumps({"document_ids": [2, 302, 3], "course_id": "MATH-101"}),
                json.dumps({"document_ids": [2, 303, 3], "course_id": None}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    model_config = Orion100MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=16,
        num_heads=4,
        intermediate_size=24,
        max_sequence_length=8,
        base_training_length=4,
        local_window=4,
        global_attention_every=1,
    )
    config = CourseMatchingTrainingConfig(
        model=model_config,
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1, log_every_steps=1),
        sequence_length=4,
        batch_size=2,
        negative_courses=1,
        warmup_steps=0,
    )
    labels = TaskLabelMap(
        subjects=("math", "physics"),
        document_types=("notes", "slides"),
        topics=("algebra", "mechanics"),
        tags=("exam",),
    )
    pretrained = Orion100MDocumentModel(labels, model_config)
    source = CheckpointManager(tmp_path / "multitask-checkpoint")
    source.save(
        model=pretrained,
        optimizer=torch.optim.SGD(pretrained.parameters(), lr=0.01),
        scheduler=None,
        step=0,
        metadata={
            "run_identity": {
                "model": "orion-100m-multitask",
                "training_config": {"model": model_config.model_dump(mode="json")},
                "labels": {
                    "subjects": list(labels.subjects),
                    "document_types": list(labels.document_types),
                    "topics": list(labels.topics),
                    "tags": list(labels.tags),
                    "bio_tags": list(labels.bio_tags),
                },
            }
        },
    )
    result = train_course_matching(
        registry,
        data,
        tmp_path / "course-matching-run",
        labels,
        config,
        initial_checkpoint=tmp_path / "multitask-checkpoint",
        device="cpu",
    )
    assert result["objective"] == "registry_prototype_open_set"
    assert result["step"] == 1
    assert torch.isfinite(torch.tensor(result["observed_losses"])).all()
    assert (tmp_path / "course-matching-run" / "checkpoints" / "latest.json").is_file()
    evaluated = evaluate_course_matching(
        registry,
        data,
        tmp_path / "course-matching-run",
        labels,
        config,
        device="cpu",
    )
    assert evaluated["status"] == "evaluated"
    assert torch.isfinite(torch.tensor(evaluated["loss"]))


def test_assistant_only_sft_runs_and_checkpoints_one_synthetic_step(tmp_path) -> None:
    data = tmp_path / "sft.jsonl"
    record = {
        "input_ids": [5, 17, 300, 19, 18, 301, 302, 19],
        "assistant_mask": [False, False, False, False, False, True, True, True],
    }
    data.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = SFTTrainingConfig(
        model=Orion300MConfig(
            vocab_size=512,
            num_layers=1,
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=12,
            pretraining_length=4,
            context_length=12,
            max_generation_tokens=4,
        ),
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1, log_every_steps=1),
        sequence_length=8,
        batch_size=1,
        warmup_steps=0,
    )
    pretrainer = Orion300MDecoder(config.model)
    pretrain_optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=0.001)
    initial_checkpoint = CheckpointManager(tmp_path / "causal-checkpoint")
    initial_checkpoint.save(
        model=pretrainer,
        optimizer=pretrain_optimizer,
        scheduler=None,
        step=0,
        metadata={
            "run_identity": {
                "model": "orion-300m",
                "training_config": {"model": config.model.model_dump(mode="json")},
            }
        },
    )
    result = train_sft_jsonl(
        data,
        tmp_path / "sft-run",
        config,
        initial_checkpoint=tmp_path / "causal-checkpoint",
        device="cpu",
    )
    assert result["objective"] == "assistant_only_sft"
    assert result["step"] == 1
    assert torch.isfinite(torch.tensor(result["observed_losses"])).all()
    assert (tmp_path / "sft-run" / "checkpoints" / "latest.json").is_file()


def test_sft_jsonl_rejects_records_without_assistant_targets(tmp_path) -> None:
    data = tmp_path / "invalid.jsonl"
    data.write_text(
        json.dumps({"input_ids": [5, 17, 300], "assistant_mask": [False, False, False]}),
        encoding="utf-8",
    )
    config = SFTTrainingConfig(
        model=Orion300MConfig(
            vocab_size=512,
            num_layers=1,
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=12,
            pretraining_length=4,
            context_length=12,
            max_generation_tokens=4,
        ),
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1),
        sequence_length=8,
        warmup_steps=0,
    )
    with pytest.raises(ValueError, match="invalid tokens or target mask"):
        from orion_300m.sft import SFTJsonlDataset

        SFTJsonlDataset(data, config=config)
