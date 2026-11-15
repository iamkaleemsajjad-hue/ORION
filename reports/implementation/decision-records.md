# Implementation decision records

## ADR-001 - Safe resumable training state

**Status:** Accepted for software implementation, 2026-11-14.

**Context:** Plan Section 25 names `training_state.pt`, while Section 30.3
requires Safetensors and rejects pickle loading. A `.pt` artifact may use
pickle serialization and is not suitable for untrusted release artifacts.

**Decision:** Store model and optimizer tensor state in separate Safetensors
files. Store scheduler/data cursor metadata, optimizer group metadata, RNG
metadata, and hashes in JSON. ORION does not use `torch.load` or pickle to
restore model/checkpoint files.

**Consequence:** The checkpoint is resumable for the supported
single-process paths and for DDP rank state in the implemented format. A
two-process Gloo test exists but was skipped on the current Windows PyTorch
build because it cannot create a Gloo network device. FSDP is not implemented.
Exact recovery with multi-worker DataLoader prefetch is not verified.

## ADR-002 - No trained release artifacts in the source repository

**Status:** Accepted.

**Context:** The task forbids fabricated checkpoints and prohibits training or
downloading a tokenizer corpus during software implementation.

**Decision:** No tokenizer model, model weights, real dataset shards, or
trained-release claims are included. Runtime load APIs require local
hash-verified artifacts and raise typed missing-artifact errors when the
artifacts are absent. Random initialization and synthetic tokenizers are
limited to software tests.

**Consequence:** SDK inference requiring learned behavior remains unavailable
until a separately approved training and release phase.