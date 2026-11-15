"""Small explicit trainer loop for supervised, self-supervised, and LM objectives."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import NDArray
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from orion_core.device import autocast_dtype, select_device
from orion_core.runtime import seed_everything as _seed_everything
from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig
from orion_train.distributed import DistributedContext, initialize_ddp, wrap_ddp

logger = logging.getLogger("orion.train")
ModelT = TypeVar("ModelT", bound=nn.Module)
BatchT = TypeVar("BatchT")
_NumpyRNGState = tuple[str, NDArray[np.uint32], int, int, float]


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    loss_ema: float | None = None


def seed_everything(seed: int) -> None:
    """Set Python and PyTorch RNGs; callers record the seed in their run config."""
    _seed_everything(seed)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _epoch_sampler(data_loader: Iterable[Any]) -> Any | None:
    """Find a sampler that exposes resumable epoch boundaries, including batch samplers."""
    for name in ("sampler", "batch_sampler"):
        sampler = getattr(data_loader, name, None)
        if callable(getattr(sampler, "next_epoch", None)):
            return sampler
    return None


class _ObjectiveModule(nn.Module, Generic[ModelT, BatchT]):
    """Put objective computation inside DDP's forward so every model path is reduced."""

    def __init__(
        self,
        model: ModelT,
        loss_function: Callable[[ModelT, BatchT], torch.Tensor],
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_function = loss_function

    def forward(self, batch: BatchT) -> torch.Tensor:
        return self.loss_function(self.model, batch)


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class Trainer(Generic[ModelT, BatchT]):
    """Gradient accumulation, autocast, clipping, logging, and periodic checkpoints."""

    def __init__(
        self,
        model: ModelT,
        optimizer: torch.optim.Optimizer,
        *,
        config: TrainerConfig,
        scheduler: Any = None,
        device: str = "auto",
        loss_function: Callable[[ModelT, BatchT], torch.Tensor],
        checkpoint_manager: CheckpointManager | None = None,
        data_state_provider: Callable[[], dict[str, Any]] | None = None,
        data_state_restorer: Callable[[dict[str, Any]], None] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.distributed_context: DistributedContext | None = None
        if config.distributed_strategy == "ddp":
            self.distributed_context, self.device = initialize_ddp(
                device,
                backend=config.distributed_backend,
                timeout_seconds=config.distributed_timeout_seconds,
            )
        else:
            self.device = select_device(device)
        self.precision_dtype = autocast_dtype(config.precision, self.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.loss_function = loss_function
        self.checkpoint_manager = checkpoint_manager
        self.data_state_provider = data_state_provider
        self.data_state_restorer = data_state_restorer
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.state = TrainerState()
        self._pending_data_state: dict[str, Any] | None = None
        self._distributed_model: nn.Module | None = None
        if self.distributed_context is not None:
            objective = _ObjectiveModule(self.model, loss_function)
            self._distributed_model = wrap_ddp(objective, self.distributed_context, self.device)
            # Keep model initialization identical across ranks, then give each rank
            # independent dropout and augmentation streams.
            seed_everything(config.seed + self.distributed_context.rank)
        self.scaler = GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and config.precision == "fp16")
        )

    def fit(self, data_loader: Iterable[BatchT], *, max_steps: int | None = None) -> list[float]:
        """Train until the configured step count; returns actual observed step losses."""
        step_limit = self.config.max_steps if max_steps is None else max_steps
        if step_limit <= 0:
            raise ValueError("max_steps must be positive")
        self._configure_data_sampler(data_loader)
        if self._pending_data_state is not None and self.data_state_restorer is not None:
            self.data_state_restorer(self._pending_data_state)
            self._pending_data_state = None
        self.model.train()
        if self._distributed_model is not None:
            self._distributed_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        accumulated = 0
        while self.state.step < step_limit:
            saw_batch = False
            epoch_complete = True
            for raw_batch in self._iterate(data_loader):
                saw_batch = True
                batch = cast(BatchT, _move_to_device(raw_batch, self.device))
                synchronize_now = accumulated + 1 >= self.config.grad_accum_steps
                sync_context = (
                    self._distributed_model.no_sync()
                    if isinstance(self._distributed_model, DistributedDataParallel)
                    and not synchronize_now
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.precision_dtype,
                        enabled=self.precision_dtype is not None,
                    ):
                        loss = self._compute_loss(batch)
                        if loss.ndim != 0:
                            raise ValueError("loss_function must return a scalar tensor")
                        scaled_loss = loss / self.config.grad_accum_steps
                    if not self._loss_is_finite(loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.state.step + 1} on at least one rank"
                        )
                    torch.autograd.backward(self.scaler.scale(scaled_loss))
                accumulated += 1
                value, token_count = self._reduce_step_metrics(loss, _count_tokens(batch))
                self.state.tokens_seen += token_count
                if accumulated < self.config.grad_accum_steps:
                    continue
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {self.state.step + 1}"
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                self.state.step += 1
                losses.append(value)
                self._update_loss_ema(value)
                if self.state.step % self.config.log_every_steps == 0 and self._is_primary:
                    logger.info(
                        "step=%d loss=%.6f grad_norm=%.5f tokens_seen=%d",
                        self.state.step,
                        value,
                        float(grad_norm.detach().float().cpu()),
                        self.state.tokens_seen,
                    )
                if (
                    self.checkpoint_manager is not None
                    and self.state.step % self.config.checkpoint_every_steps == 0
                ):
                    self.save_checkpoint()
                if self.state.step >= step_limit:
                    epoch_complete = False
                    break
            if not saw_batch:
                sampler = _epoch_sampler(data_loader)
                if sampler is not None and len(sampler) == 0:
                    self.state.epoch += 1
                    sampler.next_epoch()
                    continue
                raise ValueError("data_loader yielded no training batches")
            if epoch_complete:
                self.state.epoch += 1
                sampler = _epoch_sampler(data_loader)
                next_epoch = getattr(sampler, "next_epoch", None)
                if callable(next_epoch):
                    next_epoch()
            if self.state.step >= step_limit:
                break
        if accumulated:
            # A short final accumulation is normalized by its actual number of batches.
            factor = self.config.grad_accum_steps / accumulated
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1
            if losses:
                losses[-1] = float(losses[-1])
        return losses

    def evaluate(self, data_loader: Iterable[BatchT], *, max_batches: int | None = None) -> float:
        """Return mean observed evaluation loss without updating training state."""
        was_training = self.model.training
        self.model.eval()
        if self._distributed_model is not None:
            self._distributed_model.eval()
        values = []
        try:
            with torch.no_grad():
                for index, raw_batch in enumerate(self._iterate(data_loader)):
                    if max_batches is not None and index >= max_batches:
                        break
                    batch = cast(BatchT, _move_to_device(raw_batch, self.device))
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.precision_dtype,
                        enabled=self.precision_dtype is not None,
                    ):
                        loss = self._compute_loss(batch)
                    if not self._loss_is_finite(loss):
                        raise FloatingPointError("Non-finite evaluation loss on at least one rank")
                    values.append(float(loss.float().cpu()))
        finally:
            self.model.train(was_training)
            if self._distributed_model is not None:
                self._distributed_model.train(was_training)
        if not values:
            raise ValueError("evaluation loader yielded no batches")
        local_total = sum(values)
        if self.distributed_context is not None:
            totals = torch.tensor(
                [local_total, float(len(values))],
                dtype=torch.float64,
                device=self._collective_device,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total = float(totals[0].item())
            count = float(totals[1].item())
            if count == 0:
                raise ValueError("distributed evaluation loader yielded no batches")
            return total / count
        return local_total / len(values)

    def save_checkpoint(self, *, metadata: dict[str, Any] | None = None) -> None:
        """Persist the full recoverable state when a checkpoint manager is configured."""
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint manager is not configured")
        data_state = self.data_state_provider() if self.data_state_provider else {}
        checkpoint_kwargs: dict[str, Any] = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "step": self.state.step,
            "epoch": self.state.epoch,
            "data_state": data_state,
            "metadata": {
                "tokens_seen": self.state.tokens_seen,
                "loss_ema": self.state.loss_ema,
                **(metadata or {}),
                "run_identity": self.checkpoint_metadata,
            },
            "runtime_state": {"grad_scaler": self.scaler.state_dict()},
        }
        context = self.distributed_context
        if context is None:
            self.checkpoint_manager.save(**checkpoint_kwargs)
            return

        rank_states = [None] * context.world_size
        dist.all_gather_object(rank_states, self._capture_rank_state(data_state))
        error: list[str | None] = [None]
        if context.is_primary:
            try:
                checkpoint_kwargs["distributed_states"] = rank_states
                self.checkpoint_manager.save(**checkpoint_kwargs)
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"rank-zero checkpoint save failed: {error[0]}")
        dist.barrier()

    def restore_checkpoint(
        self,
        *,
        checkpoint: str | None = None,
        expected_run_identity: dict[str, Any] | None = None,
    ) -> None:
        """Restore a complete training checkpoint and its RNG states."""
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint manager is not configured")
        state = self.checkpoint_manager.restore(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint=checkpoint,
            map_location=self.device,
            expected_metadata={
                "run_identity": (
                    self.checkpoint_metadata
                    if expected_run_identity is None
                    else expected_run_identity
                )
            },
            restore_rng=self.distributed_context is None,
        )
        runtime_state = state.get("runtime_state", {})
        if self.distributed_context is not None:
            rank_states = state.get("distributed_states")
            if (
                not isinstance(rank_states, list)
                or len(rank_states) != self.distributed_context.world_size
            ):
                raise ValueError("checkpoint does not contain matching per-rank recovery state")
            rank_state = rank_states[self.distributed_context.rank]
            if (
                not isinstance(rank_state, dict)
                or rank_state.get("rank") != self.distributed_context.rank
            ):
                raise ValueError("checkpoint per-rank recovery state is malformed")
            self._restore_rank_state(rank_state)
            runtime_state = rank_state.get("runtime_state", {})
            self._pending_data_state = rank_state.get("data_state", {})
        if "grad_scaler" in runtime_state:
            self.scaler.load_state_dict(runtime_state["grad_scaler"])
        self.state.step = int(state["step"])
        self.state.epoch = int(state["epoch"])
        if self.distributed_context is None and self.data_state_restorer is not None:
            self.data_state_restorer(state.get("data_state", {}))
        metadata = state.get("metadata", {})
        self.state.tokens_seen = int(metadata.get("tokens_seen", 0))
        ema = metadata.get("loss_ema")
        self.state.loss_ema = float(ema) if ema is not None else None

    def _update_loss_ema(self, loss: float) -> None:
        previous = self.state.loss_ema
        if previous is not None and loss > previous * self.config.loss_spike_factor:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.rollback_on_spike and self.checkpoint_manager is not None:
                latest = self.checkpoint_manager.latest()
                if latest is not None:
                    self.restore_checkpoint(checkpoint=str(latest))
                    for group in self.optimizer.param_groups:
                        group["lr"] = float(group["lr"]) * self.config.lower_lr_on_rollback
            raise FloatingPointError(
                f"Loss spike at step {self.state.step}: {loss:.6g} exceeds "
                f"{self.config.loss_spike_factor:g}× EMA {previous:.6g}"
            )
        self.state.loss_ema = (
            loss
            if previous is None
            else self.config.loss_ema_decay * previous + (1 - self.config.loss_ema_decay) * loss
        )

    @property
    def _is_primary(self) -> bool:
        return self.distributed_context is None or self.distributed_context.is_primary

    @property
    def _collective_device(self) -> torch.device:
        if self.distributed_context is not None and self.distributed_context.backend != "nccl":
            return torch.device("cpu")
        return self.device

    def _compute_loss(self, batch: BatchT) -> torch.Tensor:
        if self._distributed_model is not None:
            return cast(torch.Tensor, self._distributed_model(batch))
        return self.loss_function(self.model, batch)

    def _loss_is_finite(self, loss: torch.Tensor) -> bool:
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if self.distributed_context is not None:
            finite = finite.to(self._collective_device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite.item())

    def _reduce_step_metrics(self, loss: torch.Tensor, tokens: int) -> tuple[float, int]:
        if self.distributed_context is None:
            return float(loss.detach().float().cpu()), tokens
        metrics = torch.tensor(
            [float(loss.detach().float().cpu()), float(tokens)],
            dtype=torch.float64,
            device=self._collective_device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        return float(metrics[0].item() / self.distributed_context.world_size), int(
            metrics[1].item()
        )

    def _configure_data_sampler(self, data_loader: Iterable[BatchT]) -> None:
        context = self.distributed_context
        if context is None:
            return
        candidates = [
            getattr(data_loader, "sampler", None),
            getattr(data_loader, "batch_sampler", None),
        ]
        configured = False
        for sampler in candidates:
            configure = getattr(sampler, "configure_distributed", None)
            if callable(configure):
                configure(context.rank, context.world_size)
                configured = True
        if not configured:
            raise ValueError(
                "DDP training requires an ORION stateful sampler that can shard data by rank"
            )

    @staticmethod
    def _iterate(data_loader: Iterable[BatchT]) -> Iterable[BatchT]:
        # DataLoader iterator creation draws a base seed from the global CPU RNG
        # when no generator is supplied. That would shift dropout after resume.
        if isinstance(data_loader, DataLoader) and data_loader.num_workers == 0:
            rng_state = torch.get_rng_state()
            iterator = iter(data_loader)
            torch.set_rng_state(rng_state)
            return cast(Iterable[BatchT], iterator)
        return data_loader

    def _capture_rank_state(self, data_state: dict[str, Any]) -> dict[str, Any]:
        numpy_state = cast(_NumpyRNGState, np.random.get_state())
        cuda_state = (
            torch.cuda.get_rng_state(self.device).tolist() if self.device.type == "cuda" else None
        )
        return {
            "rank": self.distributed_context.rank if self.distributed_context else 0,
            "world_size": self.distributed_context.world_size if self.distributed_context else 1,
            "device_type": self.device.type,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "algorithm": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu_rng_state": torch.get_rng_state().tolist(),
            "torch_cuda_rng_state": cuda_state,
            "data_state": data_state,
            "runtime_state": {"grad_scaler": self.scaler.state_dict()},
        }

    def _restore_rank_state(self, rank_state: dict[str, Any]) -> None:
        context = self.distributed_context
        if context is None:
            raise RuntimeError("rank-specific state requires distributed training")
        if rank_state.get("world_size") != context.world_size:
            raise ValueError("checkpoint world size differs from the active DDP world size")
        if rank_state.get("device_type") != self.device.type:
            raise ValueError("checkpoint device type differs from the active training device")
        random.setstate(_tuple_tree(rank_state["python_rng_state"]))
        numpy_state = rank_state["numpy_rng_state"]
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(
            torch.tensor(rank_state["torch_cpu_rng_state"], dtype=torch.uint8, device="cpu")
        )
        cuda_rng = rank_state.get("torch_cuda_rng_state")
        if cuda_rng is not None:
            if self.device.type != "cuda":
                raise ValueError("CUDA checkpoint RNG cannot be restored on a CPU device")
            torch.cuda.set_rng_state(
                torch.tensor(cuda_rng, dtype=torch.uint8, device="cpu"), self.device
            )

    def close(self) -> None:
        """Release a process group initialized by this trainer."""
        if self.distributed_context is not None:
            self.distributed_context.close()


def _count_tokens(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in ("attention_mask", "input_ids", "labels"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.numel() if key != "attention_mask" else value.sum().item())
    return 0
