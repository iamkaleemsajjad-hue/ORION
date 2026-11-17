# Engineering scorecard

Scores are engineering readiness at the audited revision, not learned model quality. A 10 requires applicable requirements to be implemented and independently verified; no category receives 10 while its material verification gates remain open.

| Category | Score / 10 | Evidence and limit |
|---|---:|---|
| Repository architecture | 8 | Eight packages, clear source boundaries, import-independence tests and builds; clean dependency-resolved installation is absent. |
| ORION-100M software | 8 | Target dimensions/count and synthetic math/gradient tests; no real data, trained weights, or task quality. |
| ORION-300M software | 8 | Target GQA/tied architecture/count and cache/causal synthetic tests; real quality, E-300-1 and hardware remain. |
| Shared tokenizer | 6 | Training/verification code and synthetic Unicode/byte tests; experimental map approval is recorded; supported Tokenizers version, production corpus and trained artifact remain blocked. |
| Dataset infrastructure | 7 | Local manifests, checksums, transforms, splits, shards and synthetic verifier tests; no corpus-scale/authorized data validation. |
| Dataset validation | 7 | Synthetic schema, policy, hash, split/leakage and PII-mask cases execute; production licenses, corpus distribution and PII false-negative performance remain unvalidated. |
| Training infrastructure | 7 | Model objectives, small updates, accumulation/clipping, schedules, recovery, and activation checkpointing are exercised; package environment and full real workflow have not passed preflight. |
| Optimizers | 7 | AdamW groups, no-decay handling, optimizer updates and state restoration run on synthetic training paths; target-scale optimizer memory and accelerator behavior are not measured. |
| Learning-rate schedulers | 7 | Warmup/schedule stepping is exercised by tiny synthetic trainers; long-run boundaries and resume behavior across all recipes lack a dedicated full matrix. |
| Training reproducibility | 6 | RNG and sampler restore plus single-process continuation equivalence pass; Linux worker recovery and DDP continuation remain unverified. |
| Checkpoint recovery | 7 | Single-process synthetic resume equivalence, Safetensors checks, and model-specific resume/evaluation pass; worker-prefetch implementation exists but its multi-worker test and distributed runtime are not verified here. |
| Distributed training | 4 | DDP code and sampler logic exist; only sampler/unit behavior ran; two-process test skipped, FSDP absent. |
| Evaluation | 6 | Metrics and sealed split tooling run on fixtures; no trained-model or real benchmark results. |
| Python SDKs | 7 | Contract tests, local loading/missing-weight semantics and validation; installed-mode/real artifact use unverified. |
| CLI | 6 | Command parsers and source tests exist; full installed command workflow cannot pass dependency/approval/data gates. |
| REST API | 6 | FastAPI schemas and synthetic endpoint contracts; concurrency, network denial, trained artifact and service lifecycle not stress-tested. |
| Type safety | 9 | Strict mypy clean on 124 files with pinned version; current unsupported Tokenizers runtime means clean resolved env not checked. |
| Dependency management | 3 | Bounded package metadata and wheels; no uv/lock or isolated resolved install, and global `pip check` is conflicted. |
| Security | 5 | Source safeguards/tests and Ruff security rules pass locally and are configured in CI; dependency scanner failed before result, offline monitor and external review absent. |
| Offline operation | 5 | Local-only design/no implicit weights; no monitored network-denial evidence or clean installed artifact. |
| Documentation | 7 | Broad guides and updated evidence reports; examples not fully installed-executed, historical documents being cross-linked, link targets checked but anchors not. |
| CI/CD | 7 | CI runs strict mypy and the security workflow runs Ruff S rules; workflows are syntax-checked locally, but no locked dependency graph, GitHub execution, installed tests or resolved SBOM gate exists. |
| Packaging | 6 | Eight wheels and sdists built; package-specific README metadata and archive contents verified. Clean dependency-resolving installation and installed entry-point proof remain blocked. |
| Hardware readiness | 3 | Exact parameter counts and CPU smoke tests; no GPU device or real memory/precision/NCCL measurements. |
| Performance | 2 | Functional SDPA/KV-cache/memmap paths; no benchmarks or target hardware results. |
| Maintainability | 8 | Modular packages, strict lint/type checks, no marker matches in current scan; some data paths remain memory-resident and external gates unresolved. |
| Reproducibility | 5 | Deterministic configs, samplers and single-process synthetic recovery; no lock, production dataset manifest or verified distributed replay. |
| Testing quality | 7 | 89 tests pass with two genuine process tests skipped on this restricted Windows host; no isolated install, GPU, production artifacts or real-data evaluation. |

## Separate outcome dimensions

- Software implementation completeness: substantial, with FSDP absent and release installation unresolved.
- Software verification completeness: local synthetic checks pass; installed, Linux distributed, GPU, scanner and monitored-offline gates are open.
- Training readiness: conditional and currently blocked by dependency resolution, corpus authorization and real authorized inputs.
- Actual training completion: **not performed by design**.
- Model quality: **not measured**; no learned capability claim is made.
- Production deployment readiness: **not established**; security, legal/privacy, data, trained artifact, evaluation and hardware review remain.
