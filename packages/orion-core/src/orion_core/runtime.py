"""Device, precision, and deterministic setup helpers."""

from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext

import numpy as np
import torch

from orion_core.device import select_device as _select_device
from orion_core.errors import DeviceUnavailableError


def select_device(device: str = "auto") -> torch.device:
    """Compatibility wrapper around the shared device selection policy."""
    return _select_device(device)


def precision_dtype(name: str, device: torch.device | str) -> torch.dtype:
    device = torch.device(device)
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise DeviceUnavailableError(
                "BF16 was requested but this CUDA device does not support it"
            )
        return torch.bfloat16
    if name == "fp16":
        if device.type == "cpu":
            raise DeviceUnavailableError("FP16 autocast is not supported on the CPU training path")
        return torch.float16
    raise ValueError("precision must be one of: fp32, bf16, fp16")


def autocast_context(
    device: torch.device | str, dtype: torch.dtype
) -> AbstractContextManager[None]:
    device = torch.device(device)
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
