# Distributed training audit

## Implemented

`orion_train.distributed` and its trainer/sampler/checkpoint integration provide
opt-in process-group setup, rank/local-device handling, DDP wrapping, rank-aware
deterministic sharding, sampler epoch updates, synchronized training, rank-zero
checkpoint writes, per-rank RNG/sampler/scaler state and world-size validation
on restore. Single-process recovery includes model, optimizer, scheduler,
scaler where used, Python/NumPy/Torch RNG, step/epoch, sampler and resolved
configuration metadata.

The stateful rank sampler test passes. The full CPU suite has a two-process
Gloo gradient/recovery integration test, but that test is **skipped** on the
current Windows build because this PyTorch Gloo backend cannot create its
network device. This is not a passing DDP runtime result.

## Missing or unverified

- FSDP wrapping, sharded state-dict and optimizer-state recovery are not
  implemented. This is a real integration gap, not just a missing wrapper:
  every model-specific builder currently creates AdamW and its scheduler before
  it constructs `Trainer`; FSDP requires wrapping first and then constructing
  the optimizer. The existing `CheckpointManager` serializes full
  `model.state_dict()` and maps optimizer tensors by the names/identities of
  ordinary model parameters. FSDP needs an explicit full or sharded state-dict
  and optimizer-state lifecycle, coordinated across ranks.
- No CUDA/NCCL or multi-GPU test was possible.
- Exact resume under DataLoader worker prefetch is not verified; sampler cursor
  state alone cannot establish that every prefetched batch is replayed exactly.
- Rank failure/restart and elastic world-size changes are not claimed.

On the available PyTorch 2.9.1+cpu installation, constructing
`FullyShardedDataParallel(nn.Linear(2, 2))` fails immediately with
`FSDP needs a non-CPU accelerator device, but no accelerator device is
detected.` The newer `fully_shard` API is present, but a minimal one-rank
CPU/Gloo probe failed during process-group setup with
`makeDeviceForHostname(): unsupported gloo device`. WSL enumeration returned
`E_ACCESSDENIED`, so this host has no Linux runtime for the fallback Gloo test.
These are environment limits; they do not establish FSDP behavior on supported
hardware.

## FSDP environment reproduction

On the available PyTorch 2.9.1+cpu installation, a minimal
`FullyShardedDataParallel(nn.Linear(2, 2))` construction fails immediately with
`FSDP needs a non-CPU accelerator device, but no accelerator device is
detected.` This is direct evidence that this machine cannot perform even a
minimal FSDP forward/backward/checkpoint test; it does not demonstrate FSDP
failure on a supported CUDA system.

The current optimizer and checkpoint ownership are also incompatible with a
simple wrapper toggle. Each model-specific builder creates AdamW and scheduler
before `Trainer`, while PyTorch requires the optimizer after FSDP wrapping.
`CheckpointManager` stores ordinary full model state and keys optimizer state
using parameter object identity/name. FSDP requires an explicit full or
sharded model/optimizer state lifecycle. These API requirements are described
in the [official FSDP documentation](https://docs.pytorch.org/docs/main/distributed/fsdp.html).

## Next action

Run the existing two-process Gloo test on a supported Linux CPU runner. Before
adding FSDP, refactor the training builders so wrapping precedes optimizer
construction and add a distributed checkpoint format with tested save,
restore, inference export and optimizer-state resharding. Official PyTorch
documentation describes the separate state-dict APIs and parameter lifecycle;
the local CPU-only runtime cannot validate an accelerator FSDP run. Until those
checks pass, report DDP as partially verified and FSDP as not implemented.
