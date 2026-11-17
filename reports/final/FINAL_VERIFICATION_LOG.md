# Final verification log

**Run date:** 2026-11-17
**Workspace:** Windows AMD64, CPython 3.10.11, PyTorch 2.9.1+cpu
**Git revision:** `7d3186c7a0378ef0878a49c524068338af945547` (pre-publication audit snapshot; source-only publication is separate)

All final local checks below ran after the last source/test change. No corpus was downloaded, no production tokenizer or ORION model was trained, and no artifact was published.

| Check | Exact command/action | Result | Scope / remaining limit |
|---|---|---|---|
| Complete local suite | `python -m pytest -q` | **89 passed, 2 skipped, 26.72 s** | Skips: Windows PyTorch could not create a Gloo network device; the managed sandbox denied DataLoader multiprocessing named pipes. Linux executions have not been observed. Warnings originate from installed pytest/plugin/PyTorch/FastAPI dependencies. |
| Strict typing | `python -m mypy packages tests tools --show-error-codes` | **Pass; no issues in 124 source files** | Current Python and installed dependency environment; does not establish resolved Tokenizers 0.23.3 compatibility. |
| Lint | `python -m ruff check .` | **Pass** | Configured Ruff rules. |
| Production security rules | `python -m ruff check packages tools --select S` | **Pass** | Ruff S rules only; not a dependency vulnerability audit or external security review. Applying S101 to pytest tests reports expected test assertions and is not the CI scope. |
| Formatting | `python -m ruff format --check .` | **Pass; 262 files already formatted** | Current installed Ruff. |
| Compilation | `python -m compileall -q packages tests tools` | **Pass** | Syntax/bytecode compilation only. |
| Whitespace | `git diff --check` | **Pass** | Git printed LF-to-CRLF notices for files under `core.autocrlf`; no whitespace errors. |
| No-dummy marker scan | `rg -n -i 'TODO|FIXME|XXX|NotImplementedError|placeholder|fake benchmark|hardcoded prediction|assert True|^[[:space:]]*pass([[:space:]]|$)' packages tools` | **No matches** (expected `rg` exit 1) | Marker scan is not a semantic proof; documented deliberate synthetic fixtures remain in tests. |
| Architecture counts | `$env:PYTHONPATH=<all eight package src paths>; python tools/count_params.py` | **Pass:** ORION-100M 100,291,840; ORION-300M 306,763,200 unique, tied embeddings confirmed | PyTorch meta-device count; no full model weights allocated and no learned quality claim. |
| Full package builds | `python -m build --no-isolation --outdir build/closure-distributions <package path>` for each of eight packages | **8 wheels and 8 sdists built** | Build backend worked in the current environment. This is not dependency resolution or installation. |
| Package README metadata | `python -m pytest -q tests/test_package_metadata.py`; archive inspection of all built distributions | **1 test passed; all 8 wheel METADATA files contain Markdown descriptions and all 8 sdists contain README.md** | Fixes the prior setuptools missing-README warning. No packages published. |
| Documentation and registry integrity | Local target scan and JSON registry validation | **136 Markdown files, 107 local links, 0 missing local targets; checked by a repository-wide relative-link scan. Requirements JSON has 54 unique IDs, valid statuses, and no dangling dependency edges.** | Link check covers local targets, not URL availability, Markdown anchors, or rendering. |
| Source training smoke paths | `python -m pytest -q tests/test_training_data.py::test_model_specific_trainers_resume_and_evaluate_synthetic_runs` (also included in full suite) | **Passed in the full suite** | Uses small random-initialized models and synthetic shards; exercises real model-specific trainers, optimizer/scheduler steps, checkpoint resume, and evaluation. Not full production training. |
| Activation checkpointing | `python -m pytest -q tests/test_architectures.py` and full suite | **Passed** | CPU parity tests cover outputs, RNG preservation, gradients, and decoder cache restriction; GPU memory reduction/performance not measured. |
| Doctor/preflight | `run_doctor` and source CLI tests in `tests/test_preflight.py` | **Command behavior tested; live environment result is correctly `blocked`** | No network request. Tokenizers 0.22.2, no uv/lock, host-wide pip conflicts, no approved trained tokenizer, and no CUDA are reported. Source-path invocation is not installed-mode verification. |
| Wheel/sdist install resolution | Fresh environment install with dependencies and `pip check` | **Not completed** | Package-index access is blocked by the configured proxy and `PIP_NO_INDEX=1`; the global environment has unrelated `pip check` conflicts. |
| uv lock | `uv lock --check` | **Not runnable** | `uv` is absent and there is no `uv.lock`; do not fabricate either. |
| Linux multiprocessing | `.github/workflows/ci.yml`, `multiprocess-recovery` job | **Not executed from this workspace** | `wsl --list --quiet` returned `E_ACCESSDENIED`; Docker Linux engine pipe was absent. Workflow configuration is not execution evidence. |
| CUDA/NCCL/FSDP | Device/runtime inspection | **Not tested; CPU-only PyTorch** | No CUDA or NCCL hardware. FSDP is not implemented; the source plan makes it conditional on memory-bound runs. |
| Dependency security scan | Prior `pip-audit --local --format cyclonedx-json ...` attempt | **Blocked before result** | Vulnerability endpoint unreachable through configured proxy; no clean scan conclusion. |
| SBOM | `reports/final/orion-source-declarations.cdx.json` schema validation | **Source declaration inventory only** | This is not a resolved transitive SBOM; the lock-based SBOM is blocked on dependency resolution. |
| Monitored offline run | Proxy-blocked selected smoke tests were run previously | **Not fully verified** | No firewall/network-monitor trace, no clean installed package/artifact. |
| Production training/data | Intentionally not invoked | **Not run** | Experimental token-map approval is recorded; licensed corpus, trained tokenizer, and target compute are absent. |

## Live doctor findings

The doctor reported the workspace and all eight source-package imports as present for the explicit source-path invocation. It correctly classified PyTorch `2.9.1+cpu` as lacking CUDA, Gloo communication as `NOT TESTED`, NCCL as `EXTERNAL HARDWARE REQUIRED`, installed Tokenizers `0.22.2` as failing the declared `>=0.23.3,<0.24` range, experimental token-map approval as recorded and trained artifact as pending, uv/uv.lock as absent, package-index connectivity as `NOT TESTED`, and the global `pip check` as failing on unrelated preinstalled package conflicts. Its overall status was `blocked`; `training_started=false` and `network_access_attempted=false`.

## Reconciliation of historical findings

| Historical finding | Current status | Evidence |
|---|---|---|
| Tokenizers installed version below project constraint | Still true; package-index access prevents installation of 0.23.3. PyPI metadata confirms the 0.23.3 release and a Python 3.10 Windows x64 wheel exist; local download was blocked. | Live doctor; proxy diagnostic in `reports/final/FINAL_DEPENDENCY_AUDIT.md`; [PyPI 0.23.3](https://pypi.org/project/tokenizers/0.23.3/). |
| uv absent / uv.lock absent | Still true. | `Get-Command uv` found none; `uv.lock` absent; doctor reports both. |
| Clean dependency-resolved installation incomplete | Still true. | No network resolution; global `pip check` is not an isolated ORION environment. |
| Linux Gloo and multi-worker recovery unverified | Still true. | Both tests are skipped under this Windows sandbox; WSL access denied and Docker Linux daemon unavailable. |
| FSDP absent | Still true, and conditional in the source plan (`FSDP only if memory-bound`). No target GPU measurement supports adding it here. | `docs/training/overview.md`, `reports/final/FSDP_DECISION_REPORT.md`. |
| Security scan/SBOM incomplete | Still true. Ruff S passes; vulnerability service and resolved SBOM remain blocked. | This log and `reports/final/SBOM_AND_DEPENDENCY_SCAN.md`. |
| Package builds | Improved: rebuilt all eight wheels and sdists; added per-package readmes and verified archive metadata. | `build/closure-distributions` and `tests/test_package_metadata.py`. |
| Production tokenizer/model training absent | Still true by authorization; no fake artifacts exist. | Tokenizer release status, tests, and this log. |



## AdamW memory floor

`tests/test_preflight.py` verifies the lower-bound arithmetic: ORION-100M (100,291,840 parameters) requires 1,604,669,440 bytes (about 1.49 GiB) for FP32 parameters, FP32 gradients, and two FP32 AdamW moments; ORION-300M (306,763,200 unique tied parameters) requires 4,908,211,200 bytes (about 4.57 GiB). The preflight labels this as `WARNING`. It excludes activations, temporary workspaces, allocator reserve, DataLoader memory, and the 100M RTD generator/heads. It is not a peak-memory estimate or device-fit guarantee; target accelerator capacity remains untested.
