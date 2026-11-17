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
- The trainer and stateful samplers now track consumed indices separately from
  worker-dispatched/prefetched indices, validate ordered worker loading, and
  rewind unconsumed indices at the end of a fit call. A real two-worker
  checkpoint/resume equivalence test is configured in Ubuntu CI. It is not
  verified here because this managed Windows host denies multiprocessing pipe
  creation; stochastic worker transforms must be stateless per sample for exact
  replay.
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

These FSDP lifecycle requirements are described in the [official PyTorch FSDP
documentation](https://docs.pytorch.org/docs/main/distributed/fsdp.html).
The observed CPU-only FSDP and Gloo probe errors above show that this host
cannot perform the requested accelerator or Linux runtime verification; they
do not demonstrate failure on supported hardware.

## Next action

Run the existing two-process Gloo test on a supported Linux CPU runner. Before
adding FSDP, refactor the training builders so wrapping precedes optimizer
construction and add a distributed checkpoint format with tested save,
restore, inference export and optimizer-state resharding. Official PyTorch
documentation describes the separate state-dict APIs and parameter lifecycle;
the local CPU-only runtime cannot validate an accelerator FSDP run. Until those
checks pass, report DDP and worker-prefetch recovery as implemented but not
runtime-verified, and FSDP as not implemented.
