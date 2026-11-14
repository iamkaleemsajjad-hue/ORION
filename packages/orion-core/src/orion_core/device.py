"""Explicit local device and precision selection helpers."""

from __future__ import annotations

from typing import Literal

import torch

from orion_core.errors import DeviceUnavailableError

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def select_device(device: DeviceName | str = "auto") -> torch.device:
    """Choose a requested available device without probing external services."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    candidate = torch.device(device)
    if candidate.type == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA was requested but is not available")
    mps = getattr(torch.backends, "mps", None)
    if candidate.type == "mps" and (mps is None or not mps.is_available()):
        raise DeviceUnavailableError("MPS was requested but is not available")
    if candidate.type not in {"cpu", "cuda", "mps"}:
        raise DeviceUnavailableError(f"Unsupported device type: {candidate.type}")
    return candidate


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """Return a supported autocast dtype, or ``None`` for full precision."""
    if precision == "fp32":
        return None
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise DeviceUnavailableError("BF16 was requested but this CUDA device lacks support")
        if device.type == "cpu" or device.type == "cuda":
            return torch.bfloat16
        raise DeviceUnavailableError(f"BF16 autocast is not configured for {device.type}")
    if precision == "fp16":
        if device.type not in {"cuda", "mps"}:
            raise DeviceUnavailableError("FP16 autocast requires CUDA or MPS")
        return torch.float16
    raise ValueError("precision must be one of: fp32, bf16, fp16")
