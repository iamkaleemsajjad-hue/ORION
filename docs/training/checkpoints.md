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

Exact resume is tested for tiny deterministic single-process runs. Samplers
can track both dispatched and acknowledged sample positions. In ordered
multi-worker training, checkpoints store only samples returned to the training
loop; unconsumed prefetched indices are replayed after resume. Built-in ORION
datasets and collators are deterministic; exact replay for user-provided
datasets requires stateless per-sample worker transforms. The genuine
multi-worker recovery equivalence test is configured in Ubuntu CI, but could
not run in this managed Windows sandbox because multiprocessing named-pipe
creation returned `WinError 5`.

The DDP implementation stores per-rank RNG, sampler and scaler state in the
rank-zero checkpoint and has a two-process Gloo regression test for gradient
synchronization and interruption/resume equivalence. That test was skipped in
this Windows environment because the PyTorch Gloo backend cannot create a
network device; a dedicated Ubuntu CI job now requires it to execute. CUDA/NCCL
and FSDP checkpoint recovery remain unverified; FSDP is not implemented or
advertised.

For DDP, all ranks must see the same checkpoint directory on shared storage and
resume with the same world size. Set `trainer.distributed_strategy` to
`ddp`; launch model-specific training through `torchrun`. The default
`none` strategy is single-process. Call `Trainer.close()` when managing an
ORION-owned process group from a long-lived Python process; CLI processes
release it at exit.
