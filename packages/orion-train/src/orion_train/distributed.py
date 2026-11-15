"""Process-group setup and rank utilities for opt-in DDP training."""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from orion_core.device import select_device


@dataclass(frozen=True)
class DistributedContext:
    """The active process-group coordinates and ownership for one training process."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    owns_process_group: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        """Destroy a process group created by ORION; leave caller-owned groups alone."""
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def initialize_ddp(
    device: str = "auto", *, backend: str = "auto", timeout_seconds: int = 1800
) -> tuple[DistributedContext, torch.device]:
    """Initialize env:// DDP or attach to an existing process group.

    A multi-process invocation must be launched with ``torchrun`` or an equivalent
    launcher that sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT.
    """
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")

    selected_device = select_device(device)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for distributed training but is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK is outside the visible CUDA device range")
        selected_device = torch.device("cuda", local_rank)
        torch.cuda.set_device(selected_device)

    owned = False
    if not dist.is_initialized():
        required_environment = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        if not all(os.environ.get(key) for key in required_environment):
            raise RuntimeError(
                "DDP requires torchrun rendezvous variables; launch with "
                "torchrun --standalone --nproc_per_node=N"
            )
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        chosen_backend = backend
        if chosen_backend == "auto":
            chosen_backend = "nccl" if selected_device.type == "cuda" else "gloo"
        if chosen_backend not in {"gloo", "nccl"}:
            raise ValueError(f"unsupported distributed backend {chosen_backend!r}")
        if chosen_backend == "nccl" and selected_device.type != "cuda":
            raise ValueError("NCCL requires a CUDA device")
        dist.init_process_group(
            backend=chosen_backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
        owned = True
        atexit.register(dist.destroy_process_group)
    else:
        active_backend = dist.get_backend()
        chosen_backend = str(active_backend)
        if backend != "auto" and backend != chosen_backend:
            raise ValueError(
                f"initialized process group uses {chosen_backend}, requested {backend}"
            )

    context = DistributedContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        backend=chosen_backend,
        owns_process_group=owned,
    )
    return context, selected_device


def wrap_ddp(model: nn.Module, context: DistributedContext, device: torch.device) -> nn.Module:
    """Wrap a module for synchronized gradients, including conditionally used heads."""
    if not dist.is_initialized():
        raise RuntimeError("DDP wrapping requires an initialized process group")
    if device.type == "cuda":
        wrapped = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
    else:
        wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    if context.world_size != dist.get_world_size() or context.rank != dist.get_rank():
        raise RuntimeError("distributed context no longer matches the active process group")
    return wrapped
