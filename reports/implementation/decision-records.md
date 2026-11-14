# Implementation decision records

## ADR-001 — Safe resumable training state

**Status:** Accepted for software implementation, 2026-11-14.

**Context:** Plan §25 names `training_state.pt` for checkpointing, but §30.3
requires safetensors only and rejects pickle loading. A `.pt` file commonly
uses pickle serialization; even PyTorch's restricted `weights_only=True` mode
does not make arbitrary files a release-safe model format.

**Decision:** Keep model weights and optimizer tensor state in separate
`.safetensors` files. Store scheduler counters, data cursor, optimizer group
metadata, RNG state, and file hashes in JSON. Never deserialize a Python object
or call `torch.load` in ORION packages.

**Consequence:** The resulting checkpoint is resumable for supported PyTorch
optimizers and deterministic single-process loaders. It is intentionally not
binary-compatible with a hypothetical `training_state.pt` artifact. DDP/FSDP
and multi-worker prefetch exact-resume behavior still require dedicated tests.

## ADR-002 — No trained release artifacts in the source repository

**Status:** Accepted.

**Context:** The task forbids fabricating checkpoints and prohibits running
training or downloading a trained tokenizer corpus.

**Decision:** No tokenizer model, model weights, dataset shards, or release
claims are included. Runtime `load` APIs require local hash-verified artifacts
and raise a typed missing-artifact error when those artifacts are absent.
Small randomly initialized configurations and a tiny synthetic tokenizer may
be used only inside software tests.

**Consequence:** SDK inference is available only after a separately governed
training and release phase.

