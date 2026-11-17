# Known limitations

1. The current tokenizer map is approved for experimental research; production scope remains provisional and final-corpus training is absent.
2. Tokenizers 0.23.3 is required but only 0.22.2 is installed; no `uv` or lockfile is available, and the package index/proxy is blocked.
3. No dependency-resolved clean install, installed-mode preflight, all-package `pip check`, or offline wheelhouse test exists.
4. No dataset was acquired or prepared. Corpus-scale memory/throughput, real-data leakage, provenance/legal rights, language-identification quality and PII false negatives are unmeasured.
5. No model weights exist. No learned outputs, model quality, benchmark score, or release readiness is claimed.
6. DDP runtime integration is skipped on Windows; CUDA/NCCL/multi-GPU/multi-node and FSDP are unverified. FSDP is not implemented and remains conditional on memory-bound deployment.
7. Exact recovery under worker prefetch, FSDP state, rank failures and storage fault injection is not established.
8. E-300-1 has synthetic infrastructure only; the 300M production architecture decision is pending approved real data/compute.
9. Security scanner access failed; no complete dependency vulnerability report, release SBOM, monitored network-denial test, or external review exists. Preliminary source declaration inventory is explicitly incomplete.
10. No CUDA, ONNX/quantization parity, production memory, latency, throughput, or target hardware report exists.
11. CI workflows and full OS/Python matrix were inspected but not executed remotely. Documentation/link/API example automation remains partial.
12. Legal, licensing, privacy, PII, safety, and deployment approvals need responsible human review; no compliance certification is made.

All eight package metadata/readme build warnings found during final hardening are fixed: wheel Markdown descriptions and sdist README contents were verified. This does not change limitation 3: artifact construction is not a clean resolved install.
