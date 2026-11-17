# Distributed training report

DDP process setup, rank/local-rank handling, deterministic even sampler sharding, epoch propagation, no-sync accumulation, gradient reduction, rank-zero checkpoint writing, per-rank RNG/sampler state, and cleanup are implemented. The sampler unit test passes. The two-process `test_ddp_gradient_sync_and_exact_rank_state_resume` creates real worker processes and Gloo communication, but is skipped on this Windows PyTorch build because a Gloo network device cannot be created.

`python -m pytest -q` on the current host: **89 passed, 2 skipped, 4 warnings**. One skip is DDP/Gloo; one is the real two-worker DataLoader recovery test. WSL returned `E_ACCESSDENIED`, and Docker could not connect to the Linux engine. The Ubuntu GitHub job exists, but no CI run was available as evidence.

CUDA/NCCL, multi-GPU, multi-node, process failure, and distributed recovery are not runtime-verified. Do not advertise these as proven. Run `.github/workflows/ci.yml`'s `multiprocess-recovery` job on Ubuntu, then run a CUDA/NCCL suite on the target cluster. See [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/distributed.html).
