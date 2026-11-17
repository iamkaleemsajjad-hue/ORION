# Training engine final verification

## Implemented

The shared trainer includes AdamW parameter grouping, supported schedulers and warmup, gradient accumulation, clipping, autocast/FP16 scaling, finite loss checks, validation, logging, deterministic RNG restoration, checkpointing, optional DDP, and non-reentrant Transformer block activation checkpointing with RNG preservation. Model-specific entry points cover ORION-100M RTD and ORION-300M causal LM/SFT plus evaluation.

## Executed evidence

- `python -m pytest -q`: **89 passed, 2 skipped, 4 warnings, 56.39 s**.
- The combined model-specific synthetic trainer regression initializes small real model classes, trains, updates optimizer/scheduler, saves, resumes, and evaluates on synthetic token shards; checkpointing is enabled in that fixture.
- Architecture checkpointing tests compare reference/checkpointed outputs and gradients; decoder cache use is rejected while checkpointed training is active, and eval cache remains usable.
- Strict mypy passes on 124 source files; Ruff, Ruff S for production packages/tools, formatting and compileall pass.

## Readiness limit

This is low-resource synthetic evidence, not a production training run. No authorized tokenizer/data, resolved clean installation, measured target-GPU memory, throughput, CUDA/NCCL execution, or full-scale run exists. Configuration preflight correctly reports target memory as untested and CPU full-model training as requiring external hardware. The package-index lock/install blocker also remains. See [training overview](../../docs/training/overview.md), [checkpoint recovery](CHECKPOINT_RECOVERY_REPORT.md), and [training readiness](TRAINING_READINESS_REPORT.md).
