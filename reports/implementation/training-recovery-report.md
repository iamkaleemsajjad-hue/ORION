# Training, checkpoint and recovery audit

## Implemented and tested locally

The shared trainer implements AdamW parameter groups, warmup/schedules,
gradient accumulation/clipping, precision controls, validation/metrics,
loss checks, and resumable training state. Checkpoints separate Safetensors
model/optimizer/RNG payloads from JSON state/manifests, verify SHA-256 hashes,
and write through staging/atomic placement. Loading avoids unsafe general
pickle deserialization. CLI inspection matches the actual checkpoint layout.

Tiny synthetic training and checkpoint tests exercise optimizer updates,
state restore, RNG and interruption/resume equivalence for supported
single-process execution. No production checkpoint was created.

## Limits and exactness definition

Verified exact recovery is limited to tested single-process synthetic runs and
their captured model/optimizer/scheduler/scaler/RNG/sampler/config state.
Distributed recovery is partially implemented but lacks a successful
multi-process runtime test here. DataLoader prefetch recovery, FSDP state,
CUDA/NCCL, storage interruption under real concurrent writers, and large
checkpoint throughput remain unverified.

No training was started. The future commands and confirmation gates are in
`docs/cli/reference.md`; recovery behavior and artifact format are in
`docs/training/checkpoints.md`.
