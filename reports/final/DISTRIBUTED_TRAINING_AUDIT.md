# Distributed training audit

## DDP implementation and evidence

`orion_train.distributed`, the trainer, stateful samplers and checkpoint layer implement opt-in process-group setup, rank/local device selection, DDP wrapping, deterministic even sharding, epoch propagation, synchronized training, rank-zero checkpoint writes, per-rank RNG/sampler/scaler state, world-size checks and cleanup. The sampler partition unit test passes. The integration test uses two processes and Gloo, but is **skipped** on the present Windows PyTorch build; it is not verified DDP runtime evidence.

## FSDP scope decision

The plan allows FSDP only where memory-bound and recommends a single 80-GB accelerator; it does not mandate FSDP for that target. Thus FSDP is conditionally unnecessary for the recommended single-device configuration, but it is a **real unimplemented feature** for a memory-bound sharded run. Wrapping alone is insufficient: the model builders currently create optimizers before `Trainer`, while FSDP requires wrapping before optimizer construction, and the checkpoint layer needs explicit full/sharded model and optimizer state handling. PyTorch 2.9.1+cpu rejects a minimal FSDP construction due to no accelerator. The FSDP2/CPU-Gloo probe is also blocked by this host’s Gloo device support. No Linux runtime, CUDA or NCCL verification is available.

## Worker prefetch recovery and exact resume limits

The trainer now separates sampler-dispatched indices from batches acknowledged by the training loop, serializes the consumed cursor, rewinds outstanding prefetch at fit boundaries, validates ordered loading, and exposes worker/prefetch settings to each model runner. A two-worker interrupted/resumed run compares model parameters and RNG after continuation. The test is skipped locally because this managed Windows environment denies multiprocessing pipe creation; it is configured as an Ubuntu CI gate, which has not run from this workspace. Exact replay assumes worker transforms are deterministic per sample or stateless. Multi-process rank coordination is implemented but unverified; rank failure, elastic world-size changes, CUDA/NCCL, multi-node execution and FSDP recovery are not claimed. Run the two configured test nodes in `.github/workflows/ci.yml` on Linux/Gloo and the CUDA distributed suite on the target cluster before relying on distributed recovery. See [detailed distributed report](../implementation/distributed-training-report.md) and [official FSDP docs](https://docs.pytorch.org/docs/main/distributed/fsdp.html).
