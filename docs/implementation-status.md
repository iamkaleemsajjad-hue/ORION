# Implementation status

This status describes software behavior in this repository. No production
tokenizer or trained ORION weights exist. The plan's quality, speed, memory,
and capability targets are not results.

## Implemented and verified in this phase

| Area | Status | Evidence |
|---|---|---|
| Package boundaries and shared core | Implemented and tested | CPU unit tests cover configs, runtime helpers, serialization, core blocks, and import independence. |
| ORION-100M model body | Implemented and tested | Encoder forward/backward, attention masking and padding behavior pass; body count is 100,291,840 parameters. |
| ORION-300M model body | Implemented and tested | Decoder forward/backward, causal masking, tied weights, KV-cache equivalence pass; count is 306,763,200 unique parameters. |
| Tokenizer software | Implemented and partially verified | Byte-level BPE interface, local trainer, encoding, decoding, offsets, batching, and integrity checks have synthetic tests. The reserved-ID table is provisional and the final tokenizer has not been trained. |
| Data and shard tools | Implemented and partially verified | Local manifests, license gates, normalizers, filtering, deduplication, packing, acquisition safeguards, token shards, and aligned word-ID shards have synthetic tests. No dataset was acquired. |
| Training engine and checkpoints | Implemented and tested for tiny single-process cases | AdamW updates, RTD, causal-LM, multi-task, Matryoshka contrastive, registry-prototype/open-set and assistant-only SFT one-step runs, safetensors/JSON save and restore, RNG restoration, resume equivalence, and run-identity checks pass on CPU. This is software verification, not production training. |
| Evaluation utilities | Implemented and tested | Classification, span, calibration, AUROC/FPR, bootstrap, and sealed-split utilities have synthetic tests. No held-out evaluation of trained ORION weights was run. |
| SDK, CLI, and REST | Implemented and partially verified | SDK/API contracts, health endpoints, input limits, API-key behavior, streaming, and missing-weight errors have tests; CLI help/info and config validation were exercised. Inference requiring learned behavior cannot run without a trained release. |
| MADAD adapter | Implemented and tested with synthetic payloads | Organizer/Tutor contract tests pass. No MADAD source was changed or imported by standalone packages. |
| Release tooling and docs | Implemented and partially verified | Artifact/checkpoint verifiers, explicit Hub helpers, release gates, GitHub workflows, and handover docs exist. No artifact was uploaded and CI was not run. |

## Remaining work before training

- Review and approve the provisional tokenizer special-token ID map, then train
  the v1 tokenizer on the separately authorized corpus.
- Resolve data permissions and policy decisions D8, D9, D10, D13, and course
  document authorization. No corpus or dataset acquisition is included here.
- Run the implemented decoder-versus-encoder-decoder/prefix-LM E-300-1 pilot
  after the plan's data and compute approvals; synthetic tests do not resolve
  the architecture decision.
- Implement and validate DDP/FSDP orchestration and exact distributed resume.
- Generate `uv.lock` once a compatible `uv` resolver is available. No lockfile
  was fabricated when the environment could not install `uv`.
- The eight package wheels build and import from a workspace-local target. Still
  run an isolated dependency-resolving install, CI, a dependency vulnerability
  scan, and a separate security/legal/privacy review.
- After those gates, train only with authorized data and approved compute, then
  run held-out evaluation, calibration, parity, and hardware benchmarks.

## Training status

No production tokenizer training, dataset download, model pretraining,
fine-tuning, instruction tuning, GPU job, release upload, or publication was
performed. The tests use randomly initialized small models and synthetic local
batches only. Model-facing SDK calls fail clearly when a trained local artifact
is missing.
