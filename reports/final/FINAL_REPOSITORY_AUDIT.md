# Final repository audit

## Inventory

The workspace contains **225 tracked files**. Counting the current working tree while excluding `.git`, Python/tool caches, `build/`, bytecode directories, virtual environments, and generated `*.egg-info` directories gives **305 files**, including:

| Measure | Count |
|---|---:|
| Python source/test/tool files | 126 |
| Python lines | 18,997 |
| `test_*.py` files | 14 |
| Markdown files | 136 |
| Independently installable packages | 8 |
| Configuration files under `configs/` | 11 |
| GitHub workflows | 3 |
| Empty Python source files | 0 |
| Model/checkpoint files in `checkpoints/` and `artifacts/` | 0 (only `.gitkeep`) |

The eight distributions remain separated into core, tokenizer, data, training, evaluation, ORION-100M, ORION-300M, and optional MADAD adapter. Independence tests pass in the source suite. No downloaded data, trained model weights, or fake checkpoints are present.

## Live environment and verification

Host: Windows AMD64, CPython 3.10.11, PyTorch 2.9.1+cpu, Tokenizers 0.22.2. Final `python -m pytest -q`: **89 passed, 2 skipped, 4 warnings in 56.39 s**. The skips are the two-process Gloo and multi-worker DataLoader recovery tests; this environment cannot execute them. Strict mypy passes on 124 analyzed files; Ruff, production Ruff S, formatting (262 files), compilation, and `git diff --check` pass.

Parameter counts are **100,291,840** (ORION-100M) and **306,763,200 unique** (ORION-300M with tied embeddings). The new estimator reports persistent FP32 parameter/gradient/AdamW-state lower bounds of 1,604,669,440 bytes and 4,908,211,200 bytes, respectively. It excludes activation/runtime overhead and does not show device fit. Eight wheels and eight sdists were rebuilt; package descriptions and README archive contents were verified. These are build checks, not clean dependency-resolved installs.

## Dependency and release finding

`uv` and `uv.lock` remain absent. Tokenizers 0.22.2 is installed while the package requires `>=0.23.3,<0.24`. The environment sets `PIP_NO_INDEX=1`; when indexing was enabled only in a diagnostic subprocess, the configured proxy refused the connection. No lockfile was fabricated. The global `pip check` reports unrelated conflicts and is not an ORION isolated environment. Clean installation, installed-mode SDK/CLI, resolved SBOM, and a completed dependency vulnerability scan remain unverified.

The current tokenizer map digest is approved for experimental research, while broader production use remains provisional. No production corpus or trained tokenizer exists. CPU smoke tests do not validate recommended training hardware: CUDA/NCCL and target peak memory are unavailable. E-300-1 real-data selection, monitored offline operation, and external security/legal/privacy reviews remain gates.

## Git and preservation

The exact current Git counts, paths, and diff summary are in [FINAL_GIT_STATE.md](FINAL_GIT_STATE.md) and [CHANGED_FILE_MANIFEST.md](CHANGED_FILE_MANIFEST.md); those files are generated from live Git output after this report set is finalized. No commit, push, tag, release, or publication was performed.

## Final assessment

The implementation has real synthetic model/trainer behavior and strong local source checks, but the repository is **not yet verified for authorized real training**. Dependency resolution/installation, production tokenizer/data training, approved datasets, Linux multiprocess execution, target accelerator memory, and security/release validation remain open. See [historical finding reconciliation](HISTORICAL_FINDING_RECONCILIATION.md), [training readiness](TRAINING_READINESS_REPORT.md), and [external approvals](EXTERNAL_APPROVALS_AND_GATES.md).
