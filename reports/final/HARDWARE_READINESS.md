# Hardware readiness

## Host tested

Windows x64, Python 3.10.11, PyTorch 2.9.1+cpu; no CUDA device, NCCL, Linux runtime, or WSL access. CPU synthetic forward/backward and tiny trainer tests pass. A CPU run does not establish practical full-model training feasibility.

## Plan estimates versus measured facts

The implementation plan describes a minimum pilot configuration around one 24-GB GPU and recommends one A100/H100 80-GB GPU, with larger multi-GPU plans for larger corpora; these are plan assumptions, not measurements from this host. The 300M plan makes FSDP conditional on memory pressure. No activation/optimizer peak memory, wall-clock, throughput, cost, long-context, batch, CUDA precision, NCCL, or multi-node measurement was collected. The architecture-count tool verifies parameter counts only.

## Readiness categories

- CPU inference/software: synthetic forward behavior verified; usable latency not measured.
- GPU inference/training: unverified; CUDA unavailable.
- CPU full training: not claimed practical; only tiny synthetic steps were run.
- DDP: orchestration present; multiprocess runtime test skipped on Windows.
- FSDP: not implemented; conditional accelerator capability requirement.

Before training, run preflight on the target host, estimate memory from actual sequence/batch/optimizer/checkpoint config, execute bounded pilot steps, measure peak allocation, then select hardware. See `docs/training/overview.md` and [`DISTRIBUTED_TRAINING_AUDIT.md`](DISTRIBUTED_TRAINING_AUDIT.md).
