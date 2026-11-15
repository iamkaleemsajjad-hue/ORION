# Training and checkpointing

Training is explicit and separate from package installation. The shared trainer
supports gradient accumulation, finite-loss/gradient checks, clipping, AdamW
parameter groups, warmup and decay schedules, and deterministic seeds.
Model-specific packages expose RTD, causal-LM, supervised, contrastive, course
matching and SFT entry points.

Checkpoint directories contain `model.safetensors`,
`model.safetensors.sha256`, `optimizer.safetensors`,
`optimizer.safetensors.sha256`, `rng.safetensors`,
`rng.safetensors.sha256`, `training_state.json`, and `manifest.json`.
The JSON records run identity, step/tokens seen, scheduler and gradient-scaler
state, data cursor, Python and NumPy RNG metadata, and checkpoint hashes.
Model, optimizer, and framework RNG tensors use Safetensors. ORION does not
call `torch.load` or load pickle artifacts. See ADR-001 for the conflict with
the plan's `.pt` filename. The `orion-train info` output is regression-tested
against a checkpoint actually written by `CheckpointManager`.

Exact resume is tested for tiny deterministic single-process runs. The DDP
implementation stores per-rank RNG, sampler and scaler state in the rank-zero
checkpoint and has a two-process Gloo regression test for gradient
synchronization and interruption/resume equivalence. That test was skipped in
this Windows environment because the PyTorch Gloo backend cannot create a
network device; run it on a supported Linux CPU/Gloo runner before relying on
DDP. FSDP and exact recovery with DataLoader worker prefetch are not
implemented/verified.

For DDP, all ranks must see the same checkpoint directory on shared storage and
resume with the same world size. Set `trainer.distributed_strategy` to
`ddp`; launch model-specific training through `torchrun`. The default
`none` strategy is single-process. Call `Trainer.close()` when managing an
ORION-owned process group from a long-lived Python process; CLI processes
release it at exit.