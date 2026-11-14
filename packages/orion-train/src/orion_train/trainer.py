"""Small explicit trainer loop for supervised, self-supervised, and LM objectives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from orion_core.device import autocast_dtype, select_device
from orion_core.runtime import seed_everything as _seed_everything
from orion_train.checkpoint import CheckpointManager
from orion_train.config import TrainerConfig

logger = logging.getLogger("orion.train")
LossFunction = Callable[[nn.Module, Any], torch.Tensor]


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


class Trainer:
    """Gradient accumulation, autocast, clipping, logging, and periodic checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        config: TrainerConfig,
        scheduler: Any = None,
        device: str = "auto",
        loss_function: LossFunction,
        checkpoint_manager: CheckpointManager | None = None,
        data_state_provider: Callable[[], dict[str, Any]] | None = None,
        data_state_restorer: Callable[[dict[str, Any]], None] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
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
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and config.precision == "fp16")
        )

    def fit(self, data_loader: Iterable[Any], *, max_steps: int | None = None) -> list[float]:
        """Train until the configured step count; returns actual observed step losses."""
        step_limit = self.config.max_steps if max_steps is None else max_steps
        if step_limit <= 0:
            raise ValueError("max_steps must be positive")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        accumulated = 0
        while self.state.step < step_limit:
            saw_batch = False
            epoch_complete = True
            for raw_batch in data_loader:
                saw_batch = True
                batch = _move_to_device(raw_batch, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.precision_dtype,
                    enabled=self.precision_dtype is not None,
                ):
                    loss = self.loss_function(self.model, batch)
                    if loss.ndim != 0:
                        raise ValueError("loss_function must return a scalar tensor")
                    scaled_loss = loss / self.config.grad_accum_steps
                if not torch.isfinite(loss.detach()):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"Non-finite loss at step {self.state.step + 1}")
                self.scaler.scale(scaled_loss).backward()
                accumulated += 1
                self.state.tokens_seen += _count_tokens(batch)
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
                value = float(loss.detach().float().cpu())
                losses.append(value)
                self._update_loss_ema(value)
                if self.state.step % self.config.log_every_steps == 0:
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

    def evaluate(self, data_loader: Iterable[Any], *, max_batches: int | None = None) -> float:
        """Return mean observed evaluation loss without updating training state."""
        was_training = self.model.training
        self.model.eval()
        values = []
        try:
            with torch.no_grad():
                for index, raw_batch in enumerate(data_loader):
                    if max_batches is not None and index >= max_batches:
                        break
                    batch = _move_to_device(raw_batch, self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.precision_dtype,
                        enabled=self.precision_dtype is not None,
                    ):
                        loss = self.loss_function(self.model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite evaluation loss")
                    values.append(float(loss.float().cpu()))
        finally:
            self.model.train(was_training)
        if not values:
            raise ValueError("evaluation loader yielded no batches")
        return sum(values) / len(values)

    def save_checkpoint(self, *, metadata: dict[str, Any] | None = None) -> None:
        """Persist the full recoverable state when a checkpoint manager is configured."""
        if self.checkpoint_manager is None:
            raise ValueError("checkpoint manager is not configured")
        data_state = self.data_state_provider() if self.data_state_provider else {}
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.state.step,
            epoch=self.state.epoch,
            data_state=data_state,
            metadata={
                "tokens_seen": self.state.tokens_seen,
                "loss_ema": self.state.loss_ema,
                **(metadata or {}),
                "run_identity": self.checkpoint_metadata,
            },
            runtime_state={"grad_scaler": self.scaler.state_dict()},
        )

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
        )
        runtime_state = state.get("runtime_state", {})
        if "grad_scaler" in runtime_state:
            self.scaler.load_state_dict(runtime_state["grad_scaler"])
        self.state.step = int(state["step"])
        self.state.epoch = int(state["epoch"])
        if self.data_state_restorer is not None:
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


def _count_tokens(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in ("attention_mask", "input_ids", "labels"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.numel() if key != "attention_mask" else value.sum().item())
    return 0
