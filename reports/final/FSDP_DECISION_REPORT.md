# FSDP decision report

## Decision

FSDP is **not implemented and is not exposed as supported**. The supplied implementation plan says DDP for up to eight GPUs and FSDP only if memory-bound. Its recommended development configuration is a single A100/H100 80-GB device; it also gives a 24-GB minimum experimental device with reduced training scope. The plan does not require FSDP for the recommended 80-GB configuration.

The model sizes alone do not establish an FSDP need: 100.3M and 306.8M parameter tensors are small relative to an 80-GB device, but sequence activations, batch/packing, optimizer state, precision, and checkpoint overhead determine peak usage. No target GPU was available, so no measured fit claim is made. The current preflight labels target configuration peak memory `NOT TESTED`.

## Why a wrapper-only implementation is unsafe

PyTorch's [FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) and [distributed checkpoint API](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) define sharded state handling separately. ORION's builders currently create optimizers before `Trainer`; FSDP wrapping changes parameter ownership and requires the optimizer/state lifecycle to be designed around the wrapped model. Safetensors checkpointing would also need full/sharded model and optimizer state semantics and distributed restore tests. No CUDA/NCCL environment exists here to verify this.

## Gate

Measure the approved batch/sequence/precision on target hardware first. If it does not fit after gradient accumulation and activation checkpointing, implement FSDP wrapping before optimizer creation plus sharded save/restore, then verify on CUDA. Until then use only the documented DDP or single-device paths within measured limits; no FSDP capability is claimed.
