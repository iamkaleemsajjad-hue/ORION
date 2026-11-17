# ORION AI ? final engineering assessment

**Date:** 2026-11-16
**Audited revision:** `7d3186c7a0378ef0878a49c524068338af945547`
**Verdict: PARTIALLY READY.**

## Executive result

The local software contains two independently defined model families, a shared tokenizer implementation, data and training infrastructure, SDK/CLI/API layers, and meaningful synthetic training/recovery tests. The final source run passed **89 tests**, skipped **2 Windows-blocked multiprocess tests**, and emitted **4 warnings** (56.39 seconds) on Windows / CPython 3.10.11 / PyTorch 2.9.1+cpu. Strict mypy is clean across 124 analyzed files; Ruff, Ruff security rules, formatting, compilation, architecture parameter counts, and local package artifact builds pass.

This does **not** establish that production training can start without further software/environment work. Dependency resolution and clean installed-mode verification remain blocked: `uv`/`uv.lock` are absent, Tokenizers 0.22.2 is installed against the declared `>=0.23.3,<0.24` range, and the configured package proxy refuses connections. No resolved clean install or ORION-only `pip check` has been completed.

The registry contains **54 requirements**: 8 VERIFIED, 33 IMPLEMENTED ? NOT VERIFIED, 6 PARTIALLY IMPLEMENTED, and 7 BLOCKED BY EXTERNAL DEPENDENCY. Linux Gloo and actual multi-worker recovery have not executed; CUDA/NCCL, production memory, dependency vulnerability scanning, resolved SBOM, and monitored offline testing remain unverified. The tokenizer ID map is approved for experimental research; approved datasets and target hardware are absent. E-300-1 remains a real-data experiment gate.

A new preflight memory estimate reports only a persistent FP32 parameter/gradient/two-AdamW-moment lower bound: 1,604,669,440 bytes (1.49 GiB) for the 100M encoder and 4,908,211,200 bytes (4.57 GiB) for the 300M tied decoder. It excludes activation and runtime overhead and is not evidence that either workload fits a device.

No real dataset was downloaded, no production tokenizer or model was trained, no trained weights were fabricated, and no model or dataset artifacts were published. The result is **not yet ?only training left?**: clean dependency installation, authorized corpus access, Linux multiprocess verification, and target-hardware preflight still require resolution.

## Current verification

- `python -m pytest -q`: **89 passed, 2 skipped, 4 warnings, 56.39 s**.
- `python -m mypy packages tests tools --show-error-codes`: clean, **124 source files**.
- `python -m ruff check .`, `python -m ruff check packages tools --select S`, `python -m ruff format --check .`, `python -m compileall -q packages tests tools`, and `git diff --check`: pass. Ruff reports **262 files already formatted**.
- Parameter counts: **100,291,840** ORION-100M; **306,763,200 unique** ORION-300M parameters with tied embeddings.
- All eight wheel/sdist pairs rebuilt; per-package Markdown metadata and README archive contents verified. Dependency resolution and installation were not verified.
- Focused preflight tests: **8 passed**; exact memory-floor arithmetic and warning-only semantics are covered.

The two skipped tests are the actual two-process Gloo test and multi-worker DataLoader cursor-recovery test. WSL returned `E_ACCESSDENIED`; no Docker Linux engine was available. The Ubuntu CI job has not produced execution evidence.

## Changes from the final hardening pass

- Added activation checkpointing to model blocks with RNG preservation and tested outputs/gradients/cache constraints.
- Added read-only doctor and preflight status reporting for missing approvals, artifacts, backends, dependencies, and hardware.
- Fixed package full-description metadata in all eight distributions, added regression coverage, and rebuilt the archives.
- Added a transparent AdamW persistent-state memory lower-bound estimate; actual peak and device fit remain untested.
- Corrected package-index diagnosis: after overriding `PIP_NO_INDEX` only in a diagnostic child process, the configured local proxy refused the connection. No untrusted index or certificate bypass was used.

## Remaining external gates

1. Restore approved package-index access on a clean host; install uv, generate the real lock, and complete isolated installs and `pip check`.
2. Preserve the approved map digest; train and verify tokenizer v1 only after separately authorized corpus sources and package prerequisites are available.
3. Supply authorized datasets and run leakage/provenance validation, then execute E-300-1 before final architecture selection where required.
4. Run Linux Gloo and multi-worker recovery tests; provide target accelerator hardware and measure actual training memory.
5. Run dependency/security scanning and generate a resolved SBOM with the locked environment; perform monitored offline and any required external legal/security review.

See [requirements traceability](REQUIREMENTS_TRACEABILITY.md), [training readiness](TRAINING_READINESS_REPORT.md), [external gates](EXTERNAL_APPROVALS_AND_GATES.md), [verification log](FINAL_VERIFICATION_LOG.md), and [next-phase runbook](NEXT_PHASE_RUNBOOK.md).
