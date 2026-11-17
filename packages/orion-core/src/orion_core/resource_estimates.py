"""Transparent lower-bound estimates for persistent training tensor state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdamWMemoryFloor:
    """Persistent model/gradient/AdamW state bytes, excluding activation memory."""

    parameter_count: int
    parameter_bytes_per_element: int
    gradient_bytes_per_element: int
    adam_moment_bytes_per_element: int

    @property
    def parameter_bytes(self) -> int:
        return self.parameter_count * self.parameter_bytes_per_element

    @property
    def gradient_bytes(self) -> int:
        return self.parameter_count * self.gradient_bytes_per_element

    @property
    def optimizer_state_bytes(self) -> int:
        # Standard AdamW keeps first and second moments; ORION uses FP32 states.
        return self.parameter_count * self.adam_moment_bytes_per_element * 2

    @property
    def total_bytes(self) -> int:
        """Return the persistent lower bound; this is not a peak fit estimate."""
        return self.parameter_bytes + self.gradient_bytes + self.optimizer_state_bytes

    @property
    def gib(self) -> float:
        return self.total_bytes / (1024**3)


def estimate_adamw_memory_floor(
    parameter_count: int,
    *,
    parameter_bytes_per_element: int = 4,
    gradient_bytes_per_element: int = 4,
    adam_moment_bytes_per_element: int = 4,
) -> AdamWMemoryFloor:
    """Estimate persistent parameter, gradient, and standard AdamW moment storage.

    The default matches ORION's current trainer: parameters remain FP32 while
    autocast selects lower precision for eligible operations, gradients remain
    FP32, and ordinary ``torch.optim.AdamW`` stores two FP32 moment tensors.
    This intentionally excludes activations, temporary kernels, allocator
    reserve, RTD generator/heads, dataloader memory, and any master copy beyond
    the FP32 parameters. It must not be used as a device-fit guarantee.
    """
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    sizes = (
        parameter_bytes_per_element,
        gradient_bytes_per_element,
        adam_moment_bytes_per_element,
    )
    if any(size <= 0 for size in sizes):
        raise ValueError("all element sizes must be positive")
    return AdamWMemoryFloor(
        parameter_count=parameter_count,
        parameter_bytes_per_element=parameter_bytes_per_element,
        gradient_bytes_per_element=gradient_bytes_per_element,
        adam_moment_bytes_per_element=adam_moment_bytes_per_element,
    )


__all__ = ["AdamWMemoryFloor", "estimate_adamw_memory_floor"]
