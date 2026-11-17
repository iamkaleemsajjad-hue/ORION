from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.sampler import StatefulRandomBatchSampler, StatefulRandomSampler
from orion_train.trainer import Trainer


class _Rows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.values = [0.25, -0.5, 0.75, 1.0, 1.5, -1.25, 0.5, -0.75]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = self.values[index]
        return {
            "x": torch.tensor([value], dtype=torch.float32),
            "y": torch.tensor([value * 0.4], dtype=torch.float32),
        }


def _dropout_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction = cast(torch.Tensor, model(batch["x"]))
    return (prediction - batch["y"]).square().mean()


def _run_distributed_resume(
    rank: int,
    world_size: int,
    init_method: str,
    directory: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    root = Path(directory)
    source_manager = CheckpointManager(root / "source")
    resume_manager = CheckpointManager(root / "resume")
    rows = _Rows()
    config = TrainerConfig(
        max_steps=2,
        checkpoint_every_steps=1,
        log_every_steps=1,
        seed=811,
        distributed_strategy="ddp",
        distributed_backend="gloo",
    )

    def create_trainer(
        manager: CheckpointManager,
    ) -> tuple[
        Trainer[nn.Sequential, dict[str, torch.Tensor]],
        DataLoader[dict[str, torch.Tensor]],
        StatefulRandomSampler,
    ]:
        torch.manual_seed(119)
        model = nn.Sequential(nn.Linear(1, 5), nn.Dropout(0.35), nn.Linear(5, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
        sampler = StatefulRandomSampler(rows, seed=config.seed)
        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            rows, batch_size=1, sampler=sampler, num_workers=0
        )
        trainer: Trainer[nn.Sequential, dict[str, torch.Tensor]] = Trainer(
            model,
            optimizer,
            config=config,
            device="cpu",
            loss_function=_dropout_loss,
            checkpoint_manager=manager,
            data_state_provider=lambda: {"sampler": sampler.state_dict()},
            data_state_restorer=lambda state: sampler.load_state_dict(state["sampler"]),
            checkpoint_metadata={"fixture": "ddp-exact-resume"},
        )
        return trainer, loader, sampler

    try:
        first, first_loader, _ = create_trainer(source_manager)
        first.fit(first_loader, max_steps=1)
        checkpoint = source_manager.latest()
        assert checkpoint is not None
        if rank == 0:
            shutil.copytree(checkpoint, resume_manager.directory / checkpoint.name)
            (resume_manager.directory / "latest.json").write_text(
                json.dumps({"checkpoint": checkpoint.name, "step": 1}), encoding="utf-8"
            )
        dist.barrier()

        first.fit(first_loader, max_steps=2)
        expected = {
            name: tensor.detach().clone() for name, tensor in first.model.state_dict().items()
        }

        resumed, resumed_loader, sampler = create_trainer(resume_manager)
        resumed.restore_checkpoint()
        resumed.fit(resumed_loader, max_steps=2)
        assert sampler.rank == rank and sampler.world_size == world_size
        for name, tensor in resumed.model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        first.close()
        resumed.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_distributed_samplers_have_equal_disjoint_shards() -> None:
    rows = list(range(19))
    sample_orders: list[list[int]] = []
    for rank in range(3):
        sampler = StatefulRandomSampler(rows, seed=73)
        sampler.configure_distributed(rank, 3)
        sample_orders.append(list(sampler))
    flattened = [item for shard in sample_orders for item in shard]
    assert len(flattened) == 18
    assert len(set(flattened)) == 18
    assert len({len(shard) for shard in sample_orders}) == 1

    batch_orders: list[list[list[int]]] = []
    for rank in range(2):
        sampler = StatefulRandomBatchSampler(rows, batch_size=2, seed=91)
        sampler.configure_distributed(rank, 2)
        batch_orders.append(list(sampler))
    flattened_batches = [item for shard in batch_orders for batch in shard for item in batch]
    assert len(flattened_batches) == 16
    assert len(set(flattened_batches)) == 16
    assert all(len(batch) == 2 for shard in batch_orders for batch in shard)
    assert len(batch_orders[0]) == len(batch_orders[1])


@pytest.mark.skipif(
    (not dist.is_gloo_available() and os.getenv("ORION_REQUIRE_GLOO") != "1")
    or sys.platform == "win32",
    reason=(
        "this Windows PyTorch build cannot create a Gloo network device; "
        "run the process-group test on a supported Linux runner"
    ),
)
def test_ddp_gradient_sync_and_exact_rank_state_resume(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    init_method = f"tcp://127.0.0.1:{port}"
    mp.spawn(
        _run_distributed_resume,
        args=(2, init_method, str(tmp_path)),
        nprocs=2,
        join=True,
    )
