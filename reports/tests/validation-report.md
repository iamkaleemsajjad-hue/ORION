# Software validation report

Validation date: 2026-11-14. Environment: Windows, Python 3.10.11,
PyTorch 2.9.1+cpu. CUDA was unavailable. Tests used synthetic/local fixtures;
no real dataset, production tokenizer, trained model, GPU job or release upload
was produced.

## Executed checks

| Check | Result |
|---|---|
| `python -m pytest -q` | 77 passed, 1 skipped, 1 PyTorch `pynvml` deprecation warning; 31.35 s. The skipped test is the two-process Gloo runtime test on this Windows build. |
| `python -m mypy packages tests tools --show-error-codes` | Passed: no issues in 121 analyzed source files under the pinned strict configuration. |
| `python -m ruff check .` | Passed. |
| `python -m ruff format --check .` | Passed; 184 files already formatted. |
| `python -m compileall -q packages tools tests` | Passed. |
| `python tools/count_params.py` | ORION-100M 100,291,840 parameters; ORION-300M 306,763,200 unique parameters with tied embeddings. |
| `orion preflight` source mode | Imported all eight local package trees and validated all checked-in recipe/schema contracts. Returned `blocked` (exit 1): Tokenizers 0.22.2 is below the >=0.23.3,<0.24 requirement; the tokenizer-owner-approval check is also pending. `training_started` was false. |
| Tokenizer synthetic train/release tests | Tiny 512-token test fixture trains, verifies staging/cleanup behavior, and is refused by production verify/export. Final tokenizer commands were not run. |
| Data preparation/verifier tests | Synthetic local JSONL fixtures exercised prepare/verification, hashes, alignment, split checks, policy and traversal rejection. No production manifest/corpus was used. |
| `git diff --check` | No whitespace errors. Git emitted expected LF-to-CRLF working-copy notices for files under its configured line-ending policy. |

The test suite exercises architecture shapes/counts, causal and bidirectional
masks, GQA/RoPE, gradients, tokenizer Unicode/Urdu and byte fallback, objective
losses, tiny optimizer/recovery behavior, data processing, API contracts,
missing weights, package independence and synthetic DDP sampler partitioning.

## Blocked or not verified

- The two-process DDP Gloo test is skipped because this Windows PyTorch build
  cannot create a Gloo network device. Run it on a supported Linux CPU/Gloo
  runner. CUDA/NCCL and FSDP were not tested; FSDP is not implemented.
- A direct tiny FSDP construction probe on installed PyTorch 2.9.1+cpu failed
  with `FSDP needs a non-CPU accelerator device, but no accelerator device is
  detected.` This confirms local hardware blocks FSDP runtime validation; it is
  not a failure result for a supported accelerator environment.
- A one-rank CPU/Gloo probe of the installed `fully_shard` API failed during
  process-group setup with `makeDeviceForHostname(): unsupported gloo device`.
  WSL enumeration returned `E_ACCESSDENIED`; no Linux runtime is available on
  this host for the required Gloo test.
- `uv` and Tokenizers 0.23.3 were unavailable from the configured package
  index. No genuine `uv.lock` exists and dependency-resolving clean installation
  of all eight packages was not completed. `pip check` from the global
  environment reports conflicts among unrelated pre-installed packages and is
  not evidence about an isolated ORION environment.
- The source preflight is implemented. Installed-mode preflight remains
  unverified until an isolated installation is available.
- Data preparation is tested on synthetic fixtures only. Corpus-scale memory
  and throughput, real-data leakage, provenance rights, language-ID quality,
  and PII false negatives remain unverified or require external review.
- Security scanners and GitHub workflows were not run. No SBOM or vulnerability
  report was generated. Formal security, privacy, and legal reviews remain
  outstanding. Offline behavior was not monitored with network-denial tooling.
- No trained artifact exists, so learned inference, trained-artifact parity,
  ONNX/quantization parity, model-quality evaluation, GPU performance, and
  production-size memory use were not measured.

The final gate status and next actions are in
`reports/implementation/training-readiness-checklist.md` and
`reports/implementation/final_engineering_handover.md`.
