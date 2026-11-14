from __future__ import annotations

import json

import torch

from orion_300m.architecture_pilot import ArchitecturePilotConfig, run_architecture_pilot
from orion_300m.config import Orion300MConfig
from orion_train.config import TrainerConfig


def _write_pairs(path, values: tuple[tuple[int, int], ...]) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"source_ids": [5, source], "target_ids": [target, 6]})
            for source, target in values
        )
        + "\n",
        encoding="utf-8",
    )


def test_e3001_runner_executes_all_candidates_on_tiny_synthetic_splits(tmp_path) -> None:
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_pairs(training, ((300, 310), (301, 311)))
    _write_pairs(validation, ((302, 312), (303, 313)))
    model = Orion300MConfig(
        vocab_size=512,
        num_layers=2,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=12,
        pretraining_length=4,
        context_length=8,
        max_generation_tokens=4,
    )
    config = ArchitecturePilotConfig(
        model=model,
        trainer=TrainerConfig(max_steps=1, checkpoint_every_steps=1, log_every_steps=1),
        candidates=("decoder", "prefix_lm", "encoder_decoder"),
        encoder_decoder_layers=(1, 1),
        sequence_length=4,
        batch_size=2,
        warmup_steps=0,
    )
    report = run_architecture_pilot(
        training,
        validation,
        tmp_path / "pilot",
        config,
        device="cpu",
    )
    assert report["experiment"] == "E-300-1"
    assert set(report["results"]) == {"decoder", "prefix_lm", "encoder_decoder"}
    for result in report["results"].values():
        assert result["steps"] == 1
        assert result["parameters"] > 0
        assert torch.isfinite(torch.tensor(result["validation_loss"]))
    assert (tmp_path / "pilot" / "architecture-pilot-report.json").is_file()
