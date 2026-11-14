# Training and checkpointing

Training is explicit and separate from package installation. The shared trainer
supports gradient accumulation, finite-loss/gradient checks, clipping, AdamW
parameter groups, warmup/trapezoidal and warmup-stable-decay schedules, and
deterministic seeds. Model packages expose objective-specific entry points.

Checkpoints contain `model.safetensors`, `optimizer.safetensors`, `rng.safetensors`,
`training_state.json`, and `manifest.json`. The manifest hashes all three safe
tensor files and `training_state.json`. The JSON records run identity,
step/tokens seen, scheduler and gradient-scaler state, data cursor, Python and
NumPy RNG metadata, and hashes. Model, optimizer, and framework RNG tensors use
Safetensors. No ORION code calls `torch.load` or loads a pickle artifact. See
ADR-001 for the conflict with the plan's `.pt` filename.

Exact resume is covered for small deterministic single-process tests. Exact
multi-worker prefetch, DDP, and FSDP resume need distributed validation before
those modes are claimed production-ready.

Training resume requires matching model, configuration, and training-data
hashes. Evaluation checks model/configuration identity while allowing a
different held-out data hash; it does not restore the training sampler cursor
into the evaluation dataset.

