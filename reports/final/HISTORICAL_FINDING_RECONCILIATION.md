# Historical audit finding reconciliation

Rechecked on 2026-11-16 against current source, configuration, tests, and this Windows/Python 3.10.11/PyTorch 2.9.1+cpu environment. Historical descriptions below are not treated as proof of present status.

| Prior finding | Current status | Evidence / root cause | Remaining action |
|---|---|---|---|
| Tokenizers 0.22.2 below `>=0.23.3,<0.24` | Still blocked | Live doctor confirms 0.22.2; PyPI lists 0.23.3 and the relevant Windows wheel, but PIP_NO_INDEX plus refused configured proxy prevents retrieval. | Restore approved package access; install/test 0.23.3 in a clean environment. |
| uv unavailable | Still blocked | No `uv` executable was found on PATH; no cached binary/wheel is available in the environment. | Install an approved uv version through the official installation method. |
| Genuine uv.lock absent | Still blocked | Root workspace has no lockfile; resolver not available. | Generate with `uv lock`, then `uv lock --check`. |
| Clean dependency-resolved install of eight packages | Still unverified | Eight wheels/sdists build, but no isolated dependency resolution/installation was possible. | Install each distribution in clean environments with dependencies enabled; run imports, CLI/SDK, and `pip check`. |
| `pip check` absent | Rechecked; only global result exists and fails | Current global `pip check` reports unrelated package conflicts, so it is not an ORION install result. | Run after resolving/installing the project in a clean env. |
| Installed-mode preflight blocked | Still blocked | Source preflight/doctor tests pass; no clean installed environment or installed entrypoint exists. | Execute `uv run orion preflight --mode installed` after resolver/install gate. |
| Token ID owner approval pending | Resolved for experimental research | Release status records approval for digest `733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`; production corpus/training gates remain. | Preserve the approved digest; authorize corpus sources and target environment separately before production training. |
| Two-process DDP Gloo skipped on Windows | Still unverified | Full suite skips `test_ddp_gradient_sync_and_exact_rank_state_resume`; local WSL is access denied and Docker Linux engine absent. | Run actual Ubuntu Gloo test and retain CI evidence. |
| FSDP absent | Still absent; conditional | Plan says DDP up to eight GPUs and FSDP only if memory-bound. No target GPU memory measurement exists; a wrapper-only addition would not cover optimizer/checkpoint lifecycle. | Measure approved target config; implement FSDP only if it is necessary. See [decision report](FSDP_DECISION_REPORT.md). |
| Multi-worker prefetch cursor recovery unverified | Still unverified | Consumed-cursor implementation and equivalence test exist; test is skipped because Windows sandbox denies worker named pipes. | Run the actual two-worker test on Ubuntu. |
| Security scanners not executed | Partly resolved | Final Ruff S checks pass locally; dependency vulnerability scan did not produce a result because its service was unreachable. No external security review. | Run pip-audit/OSV on the resolved lock and obtain independent review. |
| SBOM unverified | Still incomplete | Existing CycloneDX source declaration inventory is schema-valid but unresolved; no transitive lock graph exists. | Generate a resolved SBOM after locking/installing. |
| Monitored network-denial test absent | Still absent | Earlier closed-proxy smoke is not a network monitor/firewall denial. Doctor makes no network request by design. | Run installed-path checks with outbound network blocked and monitored. |
| External security/privacy review absent | Still absent | No qualified legal/privacy/security review occurred. | Obtain review before real data use or public deployment. |
| Installed SDK/CLI/API E2E incomplete | Still unverified in installed mode | Source tests and synthetic endpoint/SDK contracts pass in full suite; clean install is blocked. | Run each installed package outside repository root with local artifacts. |
| E-300-1 real-data comparison pending | Still pending | Candidate runner and synthetic tests exist; no approved real data or experiment result exists. | Run the documented comparable-budget pilot after data/compute approval. |
| GPU/NCCL unverified | Still unverified | PyTorch is CPU-only; doctor reports CUDA unavailable and NCCL external hardware required. | Verify on target CUDA/NCCL hardware. |
| ONNX/quantization parity absent | Still unverified | Export infrastructure exists; no parity run on ORION artifacts or measured quality/latency result exists. | Test after trained weights and supported runtime are available. |
| Production-size memory absent | Still unverified | Doctor only reads current host RAM/disk; it does not claim model peak memory. No target accelerator measurement. | Measure configured sequence/batch/precision on target hardware. |
| Tokenizer/model training absent | Intentionally not run | No corpus or authorization; no fake trained artifacts exist. | Owner authorizes data/tokenizer/model training after external gates. |

## Git manifest discrepancy

The prior 114 raw paths versus 21 whitespace-insensitive paths discrepancy was checked with the current Git data before this final report set was added. At that point, the reported 41 tracked paths and whitespace-insensitive path sets agreed; `tools/validate_config.py` was unchanged and `git diff --check` passed. The final manifest and tracked/untracked counts are regenerated in [final Git state](FINAL_GIT_STATE.md) after report creation. Line-ending notices are due to Git's CRLF policy and are not treated as semantic source changes.
