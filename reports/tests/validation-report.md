# Software validation report

Historical validation snapshot: 2026-11-14. Environment: Windows, Python 3.10.11,
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


## Final recheck (2026-11-15)

Against revision `7d3186c7a0378ef0878a49c524068338af945547`, the latest full
`python -m pytest -q` run reported **77 passed, 1 skipped, 4 warnings in 27.69
seconds**. The skipped check remains the Windows two-process Gloo test. Strict
mypy (121 files), Ruff lint, Ruff security rules, Ruff format (210 files), and compileall passed.
Eight package wheels built with dependencies disabled; this is not clean-install
evidence. The initial Ruff security scan reported seven reviewed findings; one redundant assert was removed and six narrow context-specific annotations were added, after which the scan passed. `pip-audit` failed before
completing because the configured proxy refused its vulnerability database
request. The CycloneDX source declaration inventory passed CycloneDX 1.6 schema
validation but is not a resolved release SBOM. A local Markdown target scan
checked 88 Markdown files and 57 local links with no missing targets. No data
download, tokenizer/model training, GPU job, GitHub workflow, commit or push
was performed. See `reports/final/FINAL_VERIFICATION_LOG.md` for exact scope.

## Final engineering continuation (2026-11-16)

After implementing acknowledged sampler cursors for worker-prefetch recovery,
adding a direct Pydantic dependency to `orion-train`, and extending the tiny
model-specific training test to resume through each public model runner, the
full suite reported **82 passed, 2 skipped, 4 warnings**. The skipped tests are
the real two-process Gloo recovery test and the real two-worker prefetch
resume-equivalence test; Windows policy blocks both process paths. The two
tests are configured in the Ubuntu `multiprocess-recovery` CI job, which has
not run from this local workspace.

Strict mypy (121 files), Ruff lint, Ruff S security rules, Ruff formatting
(212 files), compilation and all three workflow YAML parses passed. The model
count tool returned 100,291,840 and 306,763,200 unique parameters and matched
both targets. All eight wheels built with `--no-deps`; `orion-train` wheel
metadata was checked for `Requires-Dist: pydantic<3,>=2.7`. A selected
tokenizer/SDK/independence suite passed 18 tests with HTTP(S)/ALL proxy access
pointed at a closed local port. This is a proxy-blocked smoke check, not a
monitored network-denial test.

`pip-audit --local` failed while connecting to `pypi.org` through the configured
proxy at `127.0.0.1:9`; no vulnerability-clean result is implied. WSL returned
`E_ACCESSDENIED`, and Docker could not connect to its Linux engine. No clean
dependency-resolving install, Linux multiprocessing run, security database
scan, production data workflow, tokenizer training or model training was
completed.
