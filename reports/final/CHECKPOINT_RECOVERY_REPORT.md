# Checkpoint recovery report

Training checkpoints use Safetensors tensors plus JSON metadata rather than pickle. The recovery path records model, optimizer, scheduler, scaler, Python/NumPy/PyTorch RNG, step/epoch, sampler/data state, run configuration and identity metadata; hashes and atomic directory publication detect corrupt or partial artifacts. Inference model weights use the safe tensor loader and checksum verification.

The full synthetic suite passes **89 tests**, including tiny single-process continuation equivalence that compares resumed weights and RNG with uninterrupted execution, model-specific ORION-100M/300M save-resume-evaluate flows, and checkpoint corruption/metadata validation cases. These tests execute the actual trainer and checkpoint code.

Multi-worker prefetch cursor equivalence and two-rank distributed recovery are implemented as real integration tests but both are skipped on this Windows host for OS/backend restrictions. FSDP sharded state, rank failure recovery, elastic world-size changes, disk fault injection, CUDA scaler continuation, and production checkpoint sizes are not verified. See [training recovery report](../implementation/training-recovery-report.md) and [distributed report](DISTRIBUTED_TRAINING_REPORT.md).
