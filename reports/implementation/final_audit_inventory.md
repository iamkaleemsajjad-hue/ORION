# Final audit inventory

## Audit baseline

- Audit date: 2026-11-14 (workspace local date).
- Workspace: `C:\Users\pc\Desktop\ORION AI`.
- Git branch/status: `## master...origin/master`.
- Baseline HEAD: `1251ac6372519b4159a8d87f756f1d4a9a6cfdd2 / 2026-11-14 / fix(deps): widen pytest-cov upper bound from <7 to <8 (Dependabot PR #7)`.
- No source, configuration, test, or documentation files were modified before this inventory was recorded. No commit, push, tag, or publication was made during this audit.
- The Git working tree was clean at the start of the audit; existing history and remote tracking were preserved.

## Repository counts

| Measure | Count |
|---|---:|
| Git-tracked files | 195 |
| Installable packages | 8 |
| Python files total | 113 |
| Python files under package `src/` | 95 |
| Package source lines | 12296 |
| Test modules (`test_*.py`) | 9 |
| Python files under `tests/` including `__init__.py` | 10 |
| Test lines | 1622 |
| Python files under `tools/` | 8 |
| Markdown documents | 42 |
| Markdown lines | 1249 |
| Configuration files under `configs/` | 10 |
| Files under `.github/` | 7 |
| Model/checkpoint artifacts (`.safetensors`, `.pt`, `.pth`, `.bin`, `.onnx`, `.ckpt`) | 0 |
| Empty Python files | 0 |
| Project empty directories | 18 |

## Runtime observed at audit start

- Python: 3.10.11; workspace `requires-python` is `>=3.10`; mypy config targets 3.10.
- PyTorch: 2.9.1+cpu (no CUDA runtime available through this interpreter).
- mypy: 2.1.0; workspace dependency range is `>=1.11,<3` (the previous handover's `<2` statement is stale against the current tracked `pyproject.toml`).
- Ruff 0.16.2; Tokenizers 0.22.2; PyArrow 22.0.0; Safetensors 0.7.0; Pydantic 2.12.5; FastAPI 0.128.0; pytest 9.0.2; NumPy 1.26.4; PyYAML 6.0.3.
- `uv` is not installed at the beginning of the audit. No `uv.lock` is tracked.
- There are no model weights or dataset payloads in the tracked repository. Checkpoints/artifacts contain `.gitkeep` only; dataset fixtures are explicitly synthetic metadata.

## Package map and declared dependency boundary

| Package | Role | Runtime dependency family from its `pyproject.toml` |
|---|---|---|
| `orion-core` | Config, schemas, Transformer blocks, safe artifacts | PyTorch, Safetensors, Pydantic, PyYAML |
| `orion-tokenizer` | Byte BPE and explicit local training | `orion-core`, Tokenizers, Pydantic |
| `orion-data` | Provenance/data pipelines | `orion-core`, NumPy, Pydantic, PyYAML; optional Parquet, Hub, dedupe extras |
| `orion-train` | Optimization, training, checkpoints | `orion-core`, PyTorch, Safetensors, NumPy |
| `orion-eval` | Metrics and sealed split tools | `orion-core`, NumPy |
| `orion-100m` | Encoder SDK/CLI/API/training | shared ORION packages, PyTorch, Pydantic; optional API/export extras |
| `orion-300m` | Decoder SDK/CLI/API/training | shared ORION packages, PyTorch, Pydantic; optional API/export extras |
| `orion-madad-adapter` | Optional host payload contracts | `orion-core`; optional model bridges |

All eight package versions are `0.1.0`, and internal runtime dependencies use exact `==0.1.0` constraints. Most external runtime dependencies are bounded ranges, not a resolved lock. The workspace and all eight package `pyproject.toml` files were read during inventory.

## Configuration inventory

- `configs/deployment/local-api.json`
- `configs/evaluation/default.json`
- `configs/evaluation/e-300-1-architecture-pilot.json`
- `configs/finetuning/orion-100m-contrastive.json`
- `configs/finetuning/orion-100m-course-matching.json`
- `configs/finetuning/orion-100m-multitask.json`
- `configs/finetuning/orion-300m-sft.json`
- `configs/pretraining/orion-100m-rtd.json`
- `configs/pretraining/orion-300m-causal.json`
- `configs/tokenizer/byte-bpe-v1.json`

## Findings recorded without code changes

- No tracked Python file is empty. No model/checkpoint artifact exists.
- Empty scaffold directories are listed below for purpose review; they were not deleted.
- The `scripts/` subtree contains no tracked files. Implemented executable tooling currently lives in `tools/` and package CLI modules.
- Candidate same-domain module pairs require reachability/use review before any removal: `orion_data/acquire.py` / `acquisition.py`, `orion_train/schedulers.py` / `schedules.py`, and `orion_tokenizer/train.py` / `training.py`. This inventory does not classify them as dead code.
- The prior handover states the mypy constraint is `<2`; current tracked configuration allows `<3`. The reported diagnostics must be reproduced before they are considered defects.
- The top-level `configs/organizer` and `configs/tutor`, experiment subfolders, `scripts/*`, and `tests/*` subfolders are empty scaffolds. Their intended use must be verified against CI and repository structure before cleanup or relocation.

### Empty project directories

- `.github/PULL_REQUEST_TEMPLATE/`
- `configs/organizer/`
- `configs/tutor/`
- `experiments/organizer/`
- `experiments/tokenizer/`
- `experiments/tutor/`
- `scripts/development/`
- `scripts/release/`
- `scripts/validation/`
- `tests/architecture/`
- `tests/independence/`
- `tests/inference/`
- `tests/integration/`
- `tests/packaging/`
- `tests/security/`
- `tests/tokenizer/`
- `tests/training/`
- `tests/unit/`

## Complete Git-tracked file list at baseline

- `.editorconfig`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- `.github/workflows/security.yml`
- `.gitignore`
- `.pre-commit-config.yaml`
- `CHANGELOG.md`
- `CITATION.cff`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `Dockerfile`
- `LICENSE`
- `Makefile`
- `ORION AI — Technical Implementation Plan.docx`
- `README.md`
- `SECURITY.md`
- `artifacts/.gitkeep`
- `checkpoints/.gitkeep`
- `compose.yaml`
- `configs/deployment/local-api.json`
- `configs/evaluation/default.json`
- `configs/evaluation/e-300-1-architecture-pilot.json`
- `configs/finetuning/orion-100m-contrastive.json`
- `configs/finetuning/orion-100m-course-matching.json`
- `configs/finetuning/orion-100m-multitask.json`
- `configs/finetuning/orion-300m-sft.json`
- `configs/pretraining/orion-100m-rtd.json`
- `configs/pretraining/orion-300m-causal.json`
- `configs/tokenizer/byte-bpe-v1.json`
- `datasets/README.md`
- `datasets/cards/README.md`
- `datasets/manifests/synthetic-example.json`
- `docs/api/reference.md`
- `docs/architecture/decisions.md`
- `docs/architecture/orion-100m.md`
- `docs/architecture/orion-300m.md`
- `docs/architecture/overview.md`
- `docs/cli/reference.md`
- `docs/data/pipeline.md`
- `docs/deployment/hugging-face.md`
- `docs/deployment/offline.md`
- `docs/development/setup.md`
- `docs/development/troubleshooting.md`
- `docs/development/workflows.md`
- `docs/evaluation/metrics.md`
- `docs/governance/data-governance.md`
- `docs/governance/open-decisions.md`
- `docs/governance/release-checklist.md`
- `docs/implementation-status.md`
- `docs/research/research-ledger.md`
- `docs/sdk/overview.md`
- `docs/security/artifacts-and-inference.md`
- `docs/security/security.md`
- `docs/tokenizer/design.md`
- `docs/tokenizer/training.md`
- `docs/training/checkpoints.md`
- `docs/training/overview.md`
- `experiments/README.md`
- `packages/orion-100m/pyproject.toml`
- `packages/orion-100m/src/orion_100m/__init__.py`
- `packages/orion-100m/src/orion_100m/api.py`
- `packages/orion-100m/src/orion_100m/cli.py`
- `packages/orion-100m/src/orion_100m/config.py`
- `packages/orion-100m/src/orion_100m/contrastive.py`
- `packages/orion-100m/src/orion_100m/course_matching.py`
- `packages/orion-100m/src/orion_100m/finetuning.py`
- `packages/orion-100m/src/orion_100m/metadata.py`
- `packages/orion-100m/src/orion_100m/model/__init__.py`
- `packages/orion-100m/src/orion_100m/model/document_model.py`
- `packages/orion-100m/src/orion_100m/model/encoder.py`
- `packages/orion-100m/src/orion_100m/model/heads.py`
- `packages/orion-100m/src/orion_100m/model/losses.py`
- `packages/orion-100m/src/orion_100m/model/metadata.py`
- `packages/orion-100m/src/orion_100m/model/pretraining.py`
- `packages/orion-100m/src/orion_100m/model/rtd.py`
- `packages/orion-100m/src/orion_100m/model/tasks.py`
- `packages/orion-100m/src/orion_100m/sdk.py`
- `packages/orion-100m/src/orion_100m/training.py`
- `packages/orion-300m/pyproject.toml`
- `packages/orion-300m/src/orion_300m/__init__.py`
- `packages/orion-300m/src/orion_300m/api.py`
- `packages/orion-300m/src/orion_300m/architecture_pilot.py`
- `packages/orion-300m/src/orion_300m/cli.py`
- `packages/orion-300m/src/orion_300m/config.py`
- `packages/orion-300m/src/orion_300m/generation.py`
- `packages/orion-300m/src/orion_300m/model/__init__.py`
- `packages/orion-300m/src/orion_300m/model/cache.py`
- `packages/orion-300m/src/orion_300m/model/decoder.py`
- `packages/orion-300m/src/orion_300m/model/encoder_decoder.py`
- `packages/orion-300m/src/orion_300m/model/generation.py`
- `packages/orion-300m/src/orion_300m/model/losses.py`
- `packages/orion-300m/src/orion_300m/sdk.py`
- `packages/orion-300m/src/orion_300m/sft.py`
- `packages/orion-300m/src/orion_300m/training.py`
- `packages/orion-core/pyproject.toml`
- `packages/orion-core/src/orion_core/__init__.py`
- `packages/orion-core/src/orion_core/blocks/__init__.py`
- `packages/orion-core/src/orion_core/blocks/attention.py`
- `packages/orion-core/src/orion_core/blocks/feedforward.py`
- `packages/orion-core/src/orion_core/blocks/norms.py`
- `packages/orion-core/src/orion_core/blocks/rotary.py`
- `packages/orion-core/src/orion_core/config.py`
- `packages/orion-core/src/orion_core/device.py`
- `packages/orion-core/src/orion_core/errors.py`
- `packages/orion-core/src/orion_core/hashing.py`
- `packages/orion-core/src/orion_core/logging.py`
- `packages/orion-core/src/orion_core/runtime.py`
- `packages/orion-core/src/orion_core/schemas.py`
- `packages/orion-core/src/orion_core/serialization.py`
- `packages/orion-core/src/orion_core/tokenizer.py`
- `packages/orion-core/src/orion_core/version.py`
- `packages/orion-data/pyproject.toml`
- `packages/orion-data/src/orion_data/__init__.py`
- `packages/orion-data/src/orion_data/acquire.py`
- `packages/orion-data/src/orion_data/acquisition.py`
- `packages/orion-data/src/orion_data/cli.py`
- `packages/orion-data/src/orion_data/contamination.py`
- `packages/orion-data/src/orion_data/dataset.py`
- `packages/orion-data/src/orion_data/dedup.py`
- `packages/orion-data/src/orion_data/licenses.py`
- `packages/orion-data/src/orion_data/manifest.py`
- `packages/orion-data/src/orion_data/normalize.py`
- `packages/orion-data/src/orion_data/packing.py`
- `packages/orion-data/src/orion_data/pipeline.py`
- `packages/orion-data/src/orion_data/quality.py`
- `packages/orion-data/src/orion_data/segment.py`
- `packages/orion-data/src/orion_data/shards.py`
- `packages/orion-data/src/orion_data/tokenize.py`
- `packages/orion-eval/pyproject.toml`
- `packages/orion-eval/src/orion_eval/__init__.py`
- `packages/orion-eval/src/orion_eval/calibration.py`
- `packages/orion-eval/src/orion_eval/metrics.py`
- `packages/orion-eval/src/orion_eval/sealed.py`
- `packages/orion-madad-adapter/pyproject.toml`
- `packages/orion-madad-adapter/src/orion_madad_adapter/__init__.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/citations.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/contracts.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/organizer.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/packing.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/sdk_bridge.py`
- `packages/orion-madad-adapter/src/orion_madad_adapter/tutor.py`
- `packages/orion-tokenizer/pyproject.toml`
- `packages/orion-tokenizer/releases/v1.0/STATUS.json`
- `packages/orion-tokenizer/src/orion_tokenizer/__init__.py`
- `packages/orion-tokenizer/src/orion_tokenizer/cli.py`
- `packages/orion-tokenizer/src/orion_tokenizer/constants.py`
- `packages/orion-tokenizer/src/orion_tokenizer/spec.py`
- `packages/orion-tokenizer/src/orion_tokenizer/tokenizer.py`
- `packages/orion-tokenizer/src/orion_tokenizer/train.py`
- `packages/orion-tokenizer/src/orion_tokenizer/training.py`
- `packages/orion-train/pyproject.toml`
- `packages/orion-train/src/orion_train/__init__.py`
- `packages/orion-train/src/orion_train/checkpoint.py`
- `packages/orion-train/src/orion_train/cli.py`
- `packages/orion-train/src/orion_train/config.py`
- `packages/orion-train/src/orion_train/corruption.py`
- `packages/orion-train/src/orion_train/losses.py`
- `packages/orion-train/src/orion_train/optim.py`
- `packages/orion-train/src/orion_train/sampler.py`
- `packages/orion-train/src/orion_train/schedulers.py`
- `packages/orion-train/src/orion_train/schedules.py`
- `packages/orion-train/src/orion_train/trainer.py`
- `pyproject.toml`
- `pytest.ini`
- `reports/architecture/README.md`
- `reports/architecture/parameter-counts.json`
- `reports/benchmarks/README.md`
- `reports/implementation/decision-records.md`
- `reports/implementation/handover.md`
- `reports/implementation/repository-audit.md`
- `reports/implementation/requirements-traceability.md`
- `reports/implementation/research-ledger.md`
- `reports/tests/validation-report.md`
- `tests/__init__.py`
- `tests/test_architecture_pilot.py`
- `tests/test_architectures.py`
- `tests/test_evaluation.py`
- `tests/test_independence.py`
- `tests/test_madad_adapter.py`
- `tests/test_sdk_api.py`
- `tests/test_supervised_training.py`
- `tests/test_tokenizer.py`
- `tests/test_training_data.py`
- `tools/__init__.py`
- `tools/count_params.py`
- `tools/export_onnx.py`
- `tools/hf_artifacts.py`
- `tools/inspect_model.py`
- `tools/validate_config.py`
- `tools/verify_checkpoint.py`
- `tools/verify_release.py`

## Post-audit workspace snapshot

- Snapshot date: 2026-11-14.
- Current visible inventory: 225 non-ignored files (195 tracked at baseline,
  78 modified tracked files, and 30 new untracked files including this
  inventory and the changed-file manifest). No files were committed or pushed.
- Current Python files: 123 total (101 package source, 14 under tests
  including `tests/__init__.py`, and 8 under tools); 13 `test_*.py` modules.
- Current package source size: 13,053 lines. Current test Python size: 2,243
  lines. Markdown files: 61. JSON files: 14, of which 11 are under `configs/`.
- Installable packages: 8. Model/checkpoint artifacts: 0. Empty Python files:
  0. The original 18 empty scaffold directories remain preserved.
- Final verification: 77 pytest passed, 1 Windows Gloo test skipped, 1
  dependency deprecation warning; strict mypy (121 source files), Ruff
  check/format, compileall and exact parameter counts passed. See the final
  validation report for blocked gates.
- A scoped source marker scan found no unjustified bare `pass`, TODO/FIXME/XXX,
  `NotImplementedError`, placeholder, fake benchmark, hardcoded prediction or
  `assert True` implementation/test. Synthetic fixtures and ONNX input tensors
  are described in `no_dummy_code_audit.md`.
