from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from orion_core.errors import ArtifactIntegrityError
from orion_data.acquire import acquire_manifest_sources
from orion_data.dataset import TokenBlockDataset
from orion_data.manifest import DatasetManifest, DatasetSource, LicenseRecord
from orion_data.normalize import mask_pii, normalize_text
from orion_data.pipeline import detect_language_hint, exact_dedup_key
from orion_data.shards import read_token_shard, write_token_shard
from orion_train.checkpoint import CheckpointManager
from orion_train.cli import main as train_cli_main
from orion_train.config import TrainerConfig
from orion_train.sampler import StatefulRandomBatchSampler, StatefulRandomSampler
from orion_train.trainer import Trainer


def _linear_regression_loss(
    network: torch.nn.Linear, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    prediction = cast(torch.Tensor, network(batch["x"]))
    return (prediction - batch["y"]).square().mean()


class _DictBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def test_local_token_shard_and_memmap_blocks(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tokens"
    record = write_token_shard(path, range(24))
    dataset = TokenBlockDataset(path, sequence_length=6, expected_sha256=None)
    assert record["token_count"] == 24
    assert len(dataset) == 4
    assert dataset[2]["input_ids"].tolist() == [12, 13, 14, 15, 16, 17]
    assert dataset[0]["attention_mask"].all()
    assert read_token_shard(path, offset=2, count=3).tolist() == [2, 3, 4]


def test_resume_sampler_restores_same_remaining_order() -> None:
    sampler = StatefulRandomSampler(list(range(20)), seed=13)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(7)]
    state = sampler.state_dict()
    expected = list(iterator)
    restored = StatefulRandomSampler(list(range(20)), seed=13)
    restored.load_state_dict(state)
    assert consumed + list(restored) == consumed + expected


def test_resumable_batch_sampler_and_trainer_advance_after_boundary_checkpoint(
    tmp_path: Path,
) -> None:
    sampler = StatefulRandomBatchSampler(list(range(5)), batch_size=2, seed=27)
    iterator = iter(sampler)
    first_batch = next(iterator)
    state = sampler.state_dict()
    expected_remaining = list(iterator)
    restored_sampler = StatefulRandomBatchSampler(list(range(5)), batch_size=2, seed=27)
    restored_sampler.load_state_dict(state)
    assert list(restored_sampler) == expected_remaining
    assert len(first_batch) == 2 and all(len(batch) == 2 for batch in expected_remaining)

    dataset = [
        {"x": torch.tensor([1.0, -1.0]), "y": torch.tensor([0.25])},
        {"x": torch.tensor([-0.5, 2.0]), "y": torch.tensor([-0.1])},
    ]
    batch_dataset = _DictBatchDataset(dataset)
    train_sampler = StatefulRandomBatchSampler(dataset, batch_size=2, seed=7)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        batch_dataset, batch_sampler=train_sampler
    )
    manager = CheckpointManager(tmp_path / "boundary-checkpoint")
    config = TrainerConfig(max_steps=2, checkpoint_every_steps=1, log_every_steps=1)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer: Trainer[torch.nn.Linear, dict[str, torch.Tensor]] = Trainer(
        model,
        optimizer,
        config=config,
        device="cpu",
        loss_function=_linear_regression_loss,
        checkpoint_manager=manager,
        data_state_provider=lambda: {"sampler": train_sampler.state_dict()},
        data_state_restorer=lambda value: train_sampler.load_state_dict(value["sampler"]),
        checkpoint_metadata={"dataset": "boundary-fixture"},
    )
    trainer.fit(loader, max_steps=1)
    assert train_sampler.cursor == train_sampler.usable_size

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.05)
    resumed_sampler = StatefulRandomBatchSampler(dataset, batch_size=2, seed=7)
    resumed_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        batch_dataset, batch_sampler=resumed_sampler
    )
    resumed: Trainer[torch.nn.Linear, dict[str, torch.Tensor]] = Trainer(
        resumed_model,
        resumed_optimizer,
        config=config,
        device="cpu",
        loss_function=_linear_regression_loss,
        checkpoint_manager=manager,
        data_state_provider=lambda: {"sampler": resumed_sampler.state_dict()},
        data_state_restorer=lambda value: resumed_sampler.load_state_dict(value["sampler"]),
        checkpoint_metadata={"dataset": "boundary-fixture"},
    )
    resumed.restore_checkpoint()
    assert resumed_sampler.cursor == resumed_sampler.usable_size
    resumed.fit(resumed_loader)
    assert resumed.state.step == 2
    assert resumed.state.epoch == 1


def test_safe_checkpoint_restores_parameters_optimizer_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(17)
    random.seed(31)
    np.random.seed(47)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=1,
        data_state={"cursor": 4},
        metadata={"run_identity": {"dataset": "fixture"}},
        runtime_state={"grad_scaler": {}},
    )
    saved_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    next_random = torch.rand(1)
    next_python = random.random()
    next_numpy = np.random.random()

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    state = manager.restore(
        model=restored_model,
        optimizer=restored_optimizer,
        expected_metadata={"run_identity": {"dataset": "fixture"}},
    )
    assert state["data_state"] == {"cursor": 4}
    assert state["runtime_state"] == {"grad_scaler": {}}
    assert restored_optimizer.param_groups[0]["lr"] == 0.01
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, saved_parameters[name])
    torch.testing.assert_close(torch.rand(1), next_random)
    assert random.random() == next_python
    assert np.random.random() == next_numpy


def test_trainer_executes_tiny_real_updates_and_checkpoint_resume(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "trainer-checkpoints")
    config = TrainerConfig(max_steps=2, checkpoint_every_steps=1, log_every_steps=1)
    trainer: Trainer[torch.nn.Linear, dict[str, torch.Tensor]] = Trainer(
        model,
        optimizer,
        config=config,
        device="cpu",
        loss_function=lambda network, batch: (network(batch["x"]) - batch["y"]).square().mean(),
        checkpoint_manager=manager,
    )
    batch = [{"x": torch.ones(2, 2), "y": torch.zeros(2, 1)}]
    observed = trainer.fit(batch)
    assert trainer.state.step == 2
    assert len(observed) == 2 and all(np.isfinite(observed))
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    second_model = torch.nn.Linear(2, 1)
    second_optimizer = torch.optim.SGD(second_model.parameters(), lr=0.1)
    second_trainer: Trainer[torch.nn.Linear, dict[str, torch.Tensor]] = Trainer(
        second_model,
        second_optimizer,
        config=config,
        device="cpu",
        loss_function=lambda network, item: network(item["x"]).square().mean(),
        checkpoint_manager=manager,
    )
    second_trainer.restore_checkpoint()
    assert second_trainer.state.step == 2
    for name, value in second_model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_interrupted_training_resumes_with_equivalent_parameters_and_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(404)
    random.seed(505)
    np.random.seed(606)
    initial_model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Dropout(p=0.25), torch.nn.Linear(4, 1)
    )
    initial_state = {
        name: tensor.detach().clone() for name, tensor in initial_model.state_dict().items()
    }
    batches = [{"x": torch.tensor([[1.0, -0.5], [0.25, 2.0]]), "y": torch.tensor([[0.2], [-0.1]])}]

    def loss_function(network: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = cast(torch.Tensor, network(batch["x"]))
        return (prediction - batch["y"]).square().mean()

    uninterrupted_manager = CheckpointManager(tmp_path / "uninterrupted")
    interrupted_model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Dropout(p=0.25), torch.nn.Linear(4, 1)
    )
    interrupted_model.load_state_dict(initial_state)
    interrupted_optimizer = torch.optim.AdamW(interrupted_model.parameters(), lr=0.01)
    interrupted_trainer: Trainer[torch.nn.Sequential, dict[str, torch.Tensor]] = Trainer(
        interrupted_model,
        interrupted_optimizer,
        config=TrainerConfig(max_steps=4, checkpoint_every_steps=1, log_every_steps=4),
        device="cpu",
        loss_function=loss_function,
        checkpoint_manager=uninterrupted_manager,
        checkpoint_metadata={"dataset": "synthetic-resume-equivalence"},
    )
    interrupted_trainer.fit(batches, max_steps=2)

    resumed_manager = CheckpointManager(tmp_path / "resumed")
    checkpoint = uninterrupted_manager.latest()
    assert checkpoint is not None
    shutil.copytree(checkpoint, resumed_manager.directory / checkpoint.name)
    (resumed_manager.directory / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": 2}), encoding="utf-8"
    )

    uninterrupted_trainer = Trainer(
        interrupted_model,
        interrupted_optimizer,
        config=TrainerConfig(max_steps=4, checkpoint_every_steps=1, log_every_steps=4),
        device="cpu",
        loss_function=loss_function,
        checkpoint_manager=uninterrupted_manager,
        checkpoint_metadata={"dataset": "synthetic-resume-equivalence"},
    )
    uninterrupted_trainer.state = interrupted_trainer.state
    uninterrupted_trainer.fit(batches)
    expected_parameters = {
        name: value.detach().clone()
        for name, value in uninterrupted_trainer.model.state_dict().items()
    }
    expected_rng = (torch.rand(3), random.random(), np.random.random())

    resumed_model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Dropout(p=0.25), torch.nn.Linear(4, 1)
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_trainer: Trainer[torch.nn.Sequential, dict[str, torch.Tensor]] = Trainer(
        resumed_model,
        resumed_optimizer,
        config=TrainerConfig(max_steps=4, checkpoint_every_steps=1, log_every_steps=4),
        device="cpu",
        loss_function=loss_function,
        checkpoint_manager=resumed_manager,
        checkpoint_metadata={"dataset": "synthetic-resume-equivalence"},
    )
    resumed_trainer.restore_checkpoint()
    assert resumed_trainer.state.step == 2
    resumed_trainer.fit(batches)
    for name, value in resumed_trainer.model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3), expected_rng[0], rtol=0, atol=0)
    assert random.random() == expected_rng[1]
    assert np.random.random() == expected_rng[2]


def test_model_specific_trainers_run_one_synthetic_step_without_release_claims(
    tmp_path: Path,
) -> None:
    from orion_100m.config import Orion100MConfig
    from orion_100m.training import RTDTrainingConfig, evaluate_rtd, train_rtd
    from orion_300m.config import Orion300MConfig
    from orion_300m.training import CausalTrainingConfig, evaluate_causal_lm, train_causal_lm

    ids = [300, 301, 302, 303, 304, 305, 306, 307]
    token_path = tmp_path / "tokens.u16"
    write_token_shard(token_path, ids)
    words_path = tmp_path / "word_ids.i32"
    from orion_data.shards import write_word_id_shard

    write_word_id_shard(words_path, [0, 1, 2, 3, 4, 5, 6, 7])
    trainer_config = TrainerConfig(
        max_steps=1,
        checkpoint_every_steps=1,
        log_every_steps=1,
    )
    encoder_config = Orion100MConfig(
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
    rtd = RTDTrainingConfig(
        model=encoder_config,
        trainer=trainer_config,
        sequence_length=4,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        generator_layers=1,
        generator_hidden_size=8,
        generator_heads=2,
        generator_intermediate_size=12,
        mask_rate=1.0,
    )
    rtd_result = train_rtd(
        token_path, tmp_path / "rtd-run", rtd, word_ids_path=words_path, device="cpu"
    )
    assert rtd_result["status"] == "training_completed"
    assert rtd_result["step"] == 1
    assert np.isfinite(rtd_result["observed_losses"]).all()
    assert (tmp_path / "rtd-run" / "checkpoints" / "latest.json").is_file()
    heldout_path = tmp_path / "heldout.u16"
    write_token_shard(heldout_path, [308, 309, 310, 311])
    heldout_words = tmp_path / "heldout-word-ids.i32"
    write_word_id_shard(heldout_words, [0, 1, 2, 3])
    rtd_evaluation = evaluate_rtd(
        heldout_path,
        tmp_path / "rtd-run",
        rtd,
        word_ids_path=heldout_words,
        device="cpu",
    )
    assert np.isfinite(rtd_evaluation["loss"])

    decoder_config = Orion300MConfig(
        vocab_size=512,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=12,
        pretraining_length=4,
        context_length=16,
        max_generation_tokens=4,
    )
    causal = CausalTrainingConfig(
        model=decoder_config,
        trainer=trainer_config,
        sequence_length=4,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
    )
    causal_result = train_causal_lm(token_path, tmp_path / "causal-run", causal, device="cpu")
    assert causal_result["status"] == "training_completed"
    assert causal_result["step"] == 1
    assert np.isfinite(causal_result["observed_losses"]).all()
    assert (tmp_path / "causal-run" / "checkpoints" / "latest.json").is_file()
    causal_evaluation = evaluate_causal_lm(
        heldout_path, tmp_path / "causal-run", causal, device="cpu"
    )
    assert np.isfinite(causal_evaluation["loss"])


def test_text_normalization_pii_language_and_hashing() -> None:
    assert normalize_text("a\u200cb\x00") == "a\u200cb"
    masked, counts = mask_pii("contact person@example.org")
    assert "person@example.org" not in masked and counts["email"] == 1
    assert detect_language_hint("\u0627\u0631\u062f\u0648 language") == "mixed-en-ur"
    assert exact_dedup_key("caf\u00e9") == exact_dedup_key("cafe\u0301")


def test_checkpoint_pointer_rejects_path_traversal(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    (manager.directory / "latest.json").write_text('{"checkpoint":"../secret"}', encoding="utf-8")
    from orion_core.errors import ArtifactIntegrityError

    with pytest.raises(ArtifactIntegrityError):
        manager.latest()


def test_checkpoint_identity_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=1,
        metadata={"run_identity": {"dataset": "a"}},
    )
    restored = torch.nn.Linear(2, 1)
    before = {key: value.detach().clone() for key, value in restored.state_dict().items()}
    with pytest.raises(ArtifactIntegrityError, match="run metadata"):
        manager.restore(
            model=restored,
            optimizer=torch.optim.AdamW(restored.parameters()),
            expected_metadata={"run_identity": {"dataset": "b"}},
        )
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_checkpoint_verification_tool_accepts_complete_checkpoint(tmp_path: Path) -> None:
    from tools.verify_checkpoint import verify

    model = torch.nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    path = manager.save(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        step=3,
        epoch=1,
    )
    result = verify(path)
    assert result["verified"] is True
    assert result["step"] == 3
    assert result["epoch"] == 1


def test_training_cli_reports_the_actual_checkpoint_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = torch.nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        step=1,
    )

    train_cli_main(["info"])
    response = json.loads(capsys.readouterr().out)

    assert sorted(response["checkpoint_format"]) == sorted(
        path.name for path in checkpoint.iterdir()
    )


def test_checkpoint_manifest_detects_training_state_tampering(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=1,
        metadata={"run_identity": {"data": "fixture"}},
    )
    state_path = checkpoint / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["metadata"]["run_identity"]["data"] = "tampered"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    restored = torch.nn.Linear(2, 1)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        manager.restore(
            model=restored,
            optimizer=torch.optim.AdamW(restored.parameters(), lr=0.01),
        )


def test_acquisition_is_opt_in_and_rejects_unsafe_repository_ids(tmp_path: Path) -> None:
    synthetic = DatasetManifest.load(
        Path(__file__).resolve().parents[1] / "datasets" / "manifests" / "synthetic-example.json"
    )
    with pytest.raises(PermissionError, match="execute=True"):
        acquire_manifest_sources(synthetic, tmp_path / "disabled")
    unsafe = DatasetManifest(
        dataset_id="fixture",
        created_at="2026-01-01",
        sources=(
            DatasetSource(
                source_id="../../outside",
                kind="huggingface-dataset",
                revision="a" * 40,
                license=LicenseRecord(
                    spdx_id="MIT", attribution="Synthetic test record", verified=True
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Invalid Hugging Face"):
        acquire_manifest_sources(unsafe, tmp_path / "unsafe", execute=True)
