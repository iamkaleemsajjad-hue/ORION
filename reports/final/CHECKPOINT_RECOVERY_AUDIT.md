# Checkpoint and recovery audit

## Implemented

Inference model tensors use Safetensors. Resumable training state is separate and includes optimizer/scheduler/scaler state where used, Python/NumPy/Torch RNG, step/epoch, sampler state, resolved config and manifest metadata. Writes use staging/atomic placement and SHA-256 manifests; verification checks corruption and paths. Loading avoids general pickle deserialization. A regression test created a real tiny checkpoint and compared the CLI's listed files with its on-disk layout. Tiny interrupted/resumed runs verify supported single-process continuation and RNG restoration.

## Recovery guarantees and limits

The tested guarantee is single-process synthetic continuation with the state captured by the trainer. It is not proof of byte-identical CUDA continuation. Worker-aware stateful samplers now save acknowledged/consumed cursor positions separately from prefetched dispatch positions; a real two-worker interrupted/resumed parameter-equivalence test is configured in Ubuntu CI, but could not execute on this managed Windows host because multiprocessing pipe creation is denied. Exact replay for stochastic worker transforms requires stateless per-sample transforms. DDP state fields and rank coordination exist, but two-process integration is skipped here. FSDP/sharded optimizer checkpoints, concurrent-writer crash durability under real storage failure and partial-write recovery at scale remain unverified. No production checkpoint was created.

Evidence: [`training-recovery-report.md`](../implementation/training-recovery-report.md), `tests/test_supervised_training.py`, `tests/test_distributed_training.py`, `tools/verify_checkpoint.py`, `docs/training/checkpoints.md`.
