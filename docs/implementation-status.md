# ORION implementation status

Status date: 2026-11-14. This report covers software only. No production
tokenizer, real dataset, or trained ORION weights exist.

## Verified locally

| Area | Evidence | Status |
|---|---|---|
| ORION-100M architecture | Executable count 100,291,840; CPU forward/backward, masking, padding and loss tests | Implemented and verified on synthetic inputs |
| ORION-300M architecture | Executable unique count 306,763,200; tied embeddings, causal masks and KV-cache tests | Implemented and verified on synthetic inputs |
| Tokenizer | Byte-level BPE recipe, train/validate/verify/export CLI, ID and hash checks, golden-vector verification | Software implemented; owner mapping approval and production tokenizer remain pending |
| Data preparation | `orion-data prepare` and `verify-prepared`; local checksum-pinned JSONL, deterministic splits, PII masking, dedupe, aligned shards and packed memmaps | Implemented; synthetic behavior verified; no approved real corpus was prepared |
| Single-process training | Objective steps, optimizer updates, RNG/sampler restoration, integrity checks and resume equivalence | Implemented and tested on tiny CPU fixtures |
| DDP | Rank-aware sampler test passes; process-group/resume test exists | Not fully verified: Windows Gloo cannot create a network device |
| SDK/API | Missing-weight behavior, schemas, limits, generation and batch contracts | Contract-tested; no trained inference artifact or clean installation |
| Unified preflight | `orion preflight` validates source imports, dependencies and all configs; model mode checks local prerequisites | Implemented; currently reports Tokenizers 0.22.2 as incompatible with the required 0.23.3+ range |
| Strict typing and code quality | Mypy 2.1.0, Ruff, format and compileall | Passed locally |
| Parameter counts | `tools/count_params.py` | Exact plan counts verified |

## Remaining gates

- The owner must approve the provisional tokenizer ID map and update the
  approval record, machine status and tokenizer recipe consistently. The
  tokenizer and model training commands remain unexecuted.
- The configured package index does not provide Tokenizers 0.23.3 or `uv`.
  There is no genuine `uv.lock`, dependency-resolving clean install, or
  `pip check` result from an isolated ORION environment.
- FSDP is not implemented. The two-process DDP Gloo test must run on a
  supported Linux CPU runner. Multi-worker prefetch cursor recovery, CUDA/NCCL,
  and FSDP checkpoint recovery remain unverified.
- Data preparation was exercised only with synthetic fixtures. Corpus-scale
  memory/throughput, real leakage, source rights, privacy, PII false negatives,
  and legal sufficiency require approved data and human review.
- Security scanners and GitHub workflows were not executed. No SBOM or
  vulnerability report was generated. Offline behavior was not monitored with
  network-denial tooling.
- No GPU benchmarks, production-size memory measurements, ONNX parity,
  quantization parity, or trained-model evaluation were run.
- E-300-1 architecture selection remains pending the approved real-data pilot.

## Validation result

The latest complete test run reports 77 passed and 1 Windows Gloo test skipped,
with one PyTorch `pynvml` deprecation warning. See
`reports/tests/validation-report.md` for commands and all limitations. No real
training, corpus download, final tokenizer training, GitHub publication, or
Hub upload was performed.
