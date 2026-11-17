# ORION implementation status

Status date: 2026-11-17. Current final evidence is recorded in
[`reports/final`](../reports/final/EXECUTIVE_SUMMARY.md). This report covers software only. No production
tokenizer, real dataset, or trained ORION weights exist.

## Verified locally

| Area | Evidence | Status |
|---|---|---|
| ORION-100M architecture | Executable count 100,291,840; CPU forward/backward, masking, padding and loss tests | Implemented and verified on synthetic inputs |
| ORION-300M architecture | Executable unique count 306,763,200; tied embeddings, causal masks and KV-cache tests | Implemented and verified on synthetic inputs |
| Tokenizer | Byte-level BPE recipe, train/validate/verify/export CLI, ID and hash checks, golden-vector verification | Current map approved for experimental research; production tokenizer remains untrained |
| Data preparation | `orion-data prepare` and `verify-prepared`; local checksum-pinned JSONL, deterministic splits, PII masking, dedupe, aligned shards and packed memmaps | Implemented; synthetic behavior verified; no approved real corpus was prepared |
| Single-process training | Objective steps, optimizer updates, RNG/sampler restoration, integrity checks, resume equivalence, and activation checkpointing with RNG/output/gradient parity | Implemented and tested on tiny CPU fixtures; accelerator memory/throughput remain unmeasured |
| DDP | Rank-aware sampler test passes; process-group/resume test exists | Not fully verified: Windows Gloo cannot create a network device |
| SDK/API | Missing-weight behavior, schemas, limits, generation and batch contracts | Contract-tested; no trained inference artifact or clean installation |
| Environment doctor and unified preflight | `orion doctor` reports package, lock, runtime, host-resource and backend status without networking; `orion preflight` validates source imports, dependencies, configs and model prerequisites | Implemented; reports installed Tokenizers mismatch, missing lock, data authorization, and unverified target memory without marking them ready |
| Strict typing and code quality | Mypy 2.1.0, Ruff, production Ruff security rules, format and compileall | Passed locally on 123 typed source files |
| Package metadata | Eight wheel/sdist builds, per-package Markdown readmes, package README regression test | Implemented and artifact-checked; dependency-resolved clean installation remains blocked |
| Parameter counts | `tools/count_params.py` | Exact plan counts verified |
| Training memory floor | `orion_core.resource_estimates` and model preflight | FP32 parameters, gradients, and AdamW moments are reported as a lower bound; peak memory and device fit remain untested |

## Remaining gates

- The recorded owner approval covers the exact tokenizer-map digest and
  experimental research only. Corpus authorization and production tokenizer
  training remain separate gates; tokenizer and model training commands remain
  unexecuted.
- `PIP_NO_INDEX=1` blocked pip index queries. Overriding it only in a
  diagnostic subprocess exposed a refused proxy connection at the configured
  local proxy endpoint. PyPI lists Tokenizers 0.23.3 for this Python/platform,
  but the package could not be retrieved. There is no genuine `uv.lock`,
  dependency-resolving clean install, or isolated ORION `pip check` result.
- FSDP is not implemented. The two-process DDP Gloo and multi-worker prefetch
  resume tests are configured for Ubuntu CI, but have not run on this host.
  CUDA/NCCL and FSDP checkpoint recovery remain unverified.
- Data preparation was exercised only with synthetic fixtures. Corpus-scale
  memory/throughput, real leakage, source rights, privacy, PII false negatives,
  and legal sufficiency require approved data and human review.
- Ruff's production-code security rules were executed and passed. The
  dependency vulnerability scan could not reach its service; the checked-in
  CycloneDX file is a source declaration inventory, not a resolved SBOM.
  GitHub workflows and monitored network-denial testing were not executed.
- No GPU benchmarks, production-size memory measurements, ONNX parity,
  quantization parity, or trained-model evaluation were run.
- E-300-1 architecture selection remains pending the approved real-data pilot.

## Validation result

The latest complete test run reports **89 passed and 2 platform-gated tests
skipped** (two-process Gloo and multi-worker DataLoader recovery), with four
environment/dependency deprecation warnings. See
`reports/final/FINAL_VERIFICATION_LOG.md` for current commands and limitations. No real
training, corpus download, final tokenizer training, model artifact publication,
or Hub upload was performed. The source-only GitHub commit does not publish
datasets or trained artifacts.
