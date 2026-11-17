# ORION AI final engineering audit and hardening handover

> This 2026-11-14 handover is historical. The current final audit was refreshed
> against revision `7d3186c7a0378ef0878a49c524068338af945547` on 2026-11-15;
> see [`reports/final/EXECUTIVE_SUMMARY.md`](../final/EXECUTIVE_SUMMARY.md) and
> its verification log before using any environment-specific status below.

Audit date: 2026-11-14. Workspace: `C:\Users\pc\Desktop\ORION AI`.
This is a local software handover. It does not claim trained model capability
or that every acceptance gate passed.

## Executive summary

The existing monorepo was audited before code changes and preserved. Real
defects were reproduced and corrected; tokenizer release gating and atomic
staging, local dataset preparation/verification, source preflight, DDP
orchestration, and exact single-process recovery checks were added or completed.
Strict mypy, Ruff, formatting, compilation, parameter-count checks, and the
synthetic test suite pass.

The repository is **not yet fully cleared for real training**. The final
tokenizer mapping requires owner approval; Tokenizers 0.23.3 is unavailable in
the current environment; there is no `uv.lock` or clean resolved installation;
the DDP runtime test is blocked by this Windows Gloo backend; FSDP is absent;
security scans, GPU benchmarking, corpus-scale validation, and external legal
and privacy review remain outstanding. No real dataset was downloaded or
prepared; no final tokenizer or ORION weights were trained; no artifacts were
published.

## Repository and implemented software

The monorepo has eight separately packaged components: `orion-core`,
`orion-tokenizer`, `orion-data`, `orion-train`, `orion-eval`, `orion-100m`,
`orion-300m`, and the isolated `orion-madad-adapter`. It also contains
configs, docs, tests, tools, reports, CI/container definitions, and release
helpers. The current tracked-plus-untracked, non-ignored inventory is in
`final_audit_inventory.md`; the per-path change list is `changed-files.md`.
No model/checkpoint artifacts are present.

Implemented functions include shared Transformer blocks and schemas; byte-BPE
training and release verification; manifest validation, explicit acquisition,
and local checksum-pinned JSONL preparation; source and model preflight;
encoder/decoder objectives; training/checkpoint code; evaluation metrics;
SDKs, CLIs, REST apps; local artifact verification; and an optional MADAD
payload adapter. The MADAD application itself was not changed.

## Models and tokenizer

- **ORION-100M:** 16 layers, width 640, 10 heads, bidirectional RoPE,
  pre-normalized GeGLU encoder. Executable unique count is **100,291,840**.
  Synthetic CPU forward/backward, masking, padding, and task-loss behavior are
  tested. Document understanding and classifier quality require trained
  weights and real held-out evaluation.
- **ORION-300M:** 28 layers, width 960, 15 query heads/5 KV heads, causal GQA,
  RoPE, RMSNorm, SwiGLU and tied embeddings. Executable unique count is
  **306,763,200**. Synthetic causal behavior, generation, KV cache parity, and
  weight tying are tested. Educational generation quality is unmeasured.
- **E-300-1:** Decoder-only, prefix-LM and encoder-decoder pilot code is
  implemented. Synthetic comparisons run, but architecture selection remains
  pending the approved real-data pilot.
- **Tokenizer:** Shared 32,768-ID byte-level BPE train/verify/export software
  is implemented. The provisional special-token map digest is
  `733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`.
  The checked-in status is pending and the owner must review/approve the map
  before v1 training or freeze. The CLI's flags are operator assertions, not
  authentication. No tokenizer v1 training occurred.

## Data, training and recovery

`orion-data prepare` and `verify-prepared` operate on explicit local,
checksum-pinned JSONL and a verified local production tokenizer; they do not
download sources. Synthetic tests cover staging, provenance/hash manifests,
normalization, filters, course/document splits, deduplication, token/word
alignment, packed memmaps, traversal and license-attribution policy. Real
corpus preparation is blocked on approved sources and trained tokenizer.
Memory-resident records/MinHash and absent corpus-scale benchmarks remain a
scaling limitation.

Single-process trainers implement AdamW, schedules, accumulation, clipping,
precision handling, validation/metrics, and Safetensors-based recovery state.
Tiny synthetic interruption/resume and RNG restoration are tested. DDP
orchestration includes rank-aware sampling, synchronization, rank-zero writes,
and per-rank state; its two-process Gloo integration test is skipped on this
Windows build. Multi-worker prefetch cursor recovery is implemented and its
two-worker resume equivalence test is configured for Ubuntu CI, but it has not
run in this host. FSDP is not implemented. No actual model training command
was invoked.

FSDP cannot be added as a wrapper-only option: model builders create the
optimizer before constructing `Trainer`, whereas PyTorch FSDP changes parameter
ownership and requires wrapping before optimizer creation. The checkpoint
manager also serializes ordinary full model state and maps optimizer entries
by parameter names, so sharded optimizer/model state needs a coordinated format
and restore path. No CUDA/multi-GPU runner is available to validate that
lifecycle. FSDP remains a real software gap pending that refactor and an
accelerator verification run.

A direct `FSDP(nn.Linear(2, 2))` probe on PyTorch 2.9.1+cpu failed before
wrapping with `FSDP needs a non-CPU accelerator device, but no accelerator
device is detected.` This is an environment limitation, not an FSDP runtime
test on supported hardware.

The installed `fully_shard` FSDP2 API is present, but a one-rank CPU/Gloo probe
failed at process-group initialization with `makeDeviceForHostname():
unsupported gloo device`. WSL enumeration returned `E_ACCESSDENIED`, so no
Linux runtime is available here for the fallback Gloo verification.

## SDK, CLI, REST, packaging and releases

Both SDKs expose the documented task methods, validate inputs, report local
model state, and fail with a missing-weights error when trained artifacts are
absent. CLIs and REST schemas are implemented and synthetic contract tests
pass. Unified `orion preflight`, tokenizer validate/train/verify/export, and
local data prepare/verify commands exist. `docs/cli/reference.md` documents
additional model train/evaluate/pilot and checkpoint/release commands. They
were not invoked for real training, datasets or publication.

Source-checkout preflight ran package and configuration checks but exits
blocked because installed Tokenizers is 0.22.2, below the package constraint
`>=0.23.3,<0.24`. All eight distributions have not been installed into a
clean dependency-resolving environment. `uv` is unavailable and no valid
`uv.lock` was generated. Wheel creation or `--no-deps` import checks would not
close this gate.

GitHub and Hub helpers/templates exist, but workflows/scanners were not
executed, no upload was authorized, and no publication occurred. There is no
SBOM or vulnerability scan report. Model artifacts must not be published until
real training/evaluation and release checks.

## Verification evidence

- `python -m pytest -q`: **82 passed, 2 skipped**, four environment/dependency
  deprecation warnings (30.60 s). The skips are two-process Gloo and
  multi-worker DataLoader recovery, which require a supported Linux runtime.
- Strict mypy 2.1.0: **no issues in 121 analyzed source files**.
- Ruff lint and `ruff format --check`: pass; **212 files already formatted**.
- `compileall`: pass.
- Exact parameter-count command: matches both plan targets.
- Source preflight: imports and checked-in configuration contracts pass;
  dependency version check blocks as described above.
- `git diff --check`: no whitespace errors (Git reports line-ending notices).

No CUDA/NCCL, FSDP, isolated install, vulnerability scanner, external security
review, network-denial/offline monitor, real data, model quality, or performance
benchmark was verified. See `reports/tests/validation-report.md` and the
individual audit reports for scope.

## Owner actions before the first real run

1. Review and approve the special-token ID registry. Complete
   `docs/tokenizer/tokenizer_approval.md` and the matching recipe/status fields.
2. Resolve the package-index constraint; install Tokenizers `>=0.23.3,<0.24`
   and `uv`, generate/commit no lockfile yet, then verify a clean install and
   `pip check` in an isolated environment.
3. Prepare an approved local JSONL manifest with immutable revisions, hashes,
   rights and attribution reviewed by the responsible humans.
4. After separately authorizing those inputs, use the exact documented
   tokenizer/data/train/evaluate commands in `docs/tokenizer/training.md`,
   `docs/data/pipeline.md`, and `docs/cli/reference.md`. Run model-specific
   preflight before training.
5. Run DDP Gloo recovery on supported Linux; decide whether FSDP is required
   for target hardware and implement/verify it before selecting FSDP configs.
6. Run security workflows, dependency audit, SBOM, network-denial tests,
   performance/memory measurements, and human privacy/legal review before
   external distribution.
7. Run the E-300-1 pilot on approved data and compute before selecting the
   ORION-300M production architecture.

The currently documented model recipes, including CUDA options, remain
software configurations rather than verified hardware plans. The machine
configuration and environment must pass preflight before a run.

## Reports

- Requirements: `requirements-traceability.md`
- Investigated defects: `issue-investigation.md`
- Type-checking: `type-checking-report.md`
- Dependencies/install: `dependency-install-report.md`
- Distributed execution: `distributed-training-report.md`
- Tokenizer compatibility/approval: `tokenizer-audit.md`
- ORION-100M and ORION-300M: `orion-100m-audit.md`, `orion-300m-audit.md`
- Data pipeline: `data-pipeline-audit.md`
- Training/recovery: `training-recovery-report.md`
- SDK/API: `sdk-api-report.md`
- Security: `security-audit.md`
- Performance: `performance-report.md`
- No dummy code: `no_dummy_code_audit.md`
- Gate checklist: `training-readiness-checklist.md`
- Changed paths: `changed-files.md`

No commit, push, tag, PR, release, upload, remote change or publication was
performed.
