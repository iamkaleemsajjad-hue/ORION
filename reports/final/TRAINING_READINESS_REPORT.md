# Training readiness report

## Verdict: PARTIALLY READY

Both model-specific training engines execute genuine synthetic training, optimizer/scheduler updates, checkpoints, resume, and evaluation on small CPU models. The final suite is **89 passed, 2 skipped, 4 warnings**; strict mypy, Ruff, Ruff S, formatting, compileall, and parameter counts pass. Activation checkpointing and the truthful doctor/preflight are implemented. Eight wheels/sdists build with validated package descriptions.

## Critical prerequisites that did not pass

1. Reproducible install: no uv/lock; package index blocked; Tokenizers 0.22.2 does not meet `>=0.23.3,<0.24`; no clean install or ORION-only `pip check`.
2. Shared tokenizer: current map digest is approved for experimental research; production corpus and trained artifact are absent.
3. Data: no authorized production corpus or held-out evaluation set.
4. Runtime: Gloo/multi-worker tests skipped on Windows; Linux CI did not execute. CUDA/NCCL and target memory not tested.
5. Security: dependency scan blocked and resolved SBOM absent; monitored offline and external review absent.
6. E-300-1: synthetic framework exists; real-data comparison and final empirical architecture selection remain pending.

The software is not yet a cleanly verified real-training installation. Do not proceed to expensive training until package resolution/install and target-environment preflight pass; production tokenizer training also requires separately authorized corpus sources.
