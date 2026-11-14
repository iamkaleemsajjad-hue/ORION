# ORION AI engineering handover

## Executive summary

This workspace now contains an independently packaged ORION software monorepo:
shared infrastructure, a randomly initialized ORION-100M encoder and ORION-300M
decoder, local data and training pipelines, SDKs, CLIs, REST applications,
artifact checks, documentation, and synthetic tests. Both architecture parameter
counts match the plan exactly.

This is **not complete against every definition-of-done item in the supplied
plan**. The tokenizer has not been trained, no model weights or data exist, the
ORION-300M architecture pilot is implemented but has not been run on authorized
real data or compute, and distributed training is not implemented. The
requirements traceability report records the status of all
36 plan sections. No real data acquisition, tokenizer training, model training,
GPU run, or publication was performed.

## Package map

| Installable package | Implemented scope |
|---|---|
| `orion-core` | Validated config contracts, typed errors, shared Transformer blocks, runtime/device utilities, safe safetensors/JSON serialization and integrity checks |
| `orion-tokenizer` | Byte-level BPE wrapper, explicit local corpus trainer, special-token map, batch/offset handling, manifests and artifact verification |
| `orion-data` | Dataset manifests/license gates, local normalization and filtering, deduplication, contamination checks, packing, token/word-ID shards and opt-in acquisition |
| `orion-train` | AdamW groups, schedulers, accumulation/clipping, precision handling, deterministic samplers, checkpoint/resume and finite-loss safeguards |
| `orion-eval` | Classification/span metrics, calibration, open-set metrics, bootstrap intervals and sealed-split safeguards |
| `orion-100m` | Encoder, task heads, RTD objective, supervised multi-task, Matryoshka contrastive and registry-prototype/open-set runners, SDK, CLI and FastAPI app |
| `orion-300m` | Causal GQA decoder, prefix-LM mode, pilot encoder-decoder, E-300-1 runner, KV cache, generation/grounding interfaces, pretraining and SFT runners, SDK, CLI and FastAPI app |
| `orion-madad-adapter` | Optional isolated host contract adapters and synthetic contract tests |

See `docs/architecture/` and `reports/implementation/requirements-traceability.md`
for the full module map and requirement-by-requirement evidence.

## Status by major requirement

| Requirement | Status | Evidence / remaining work |
|---|---|---|
| ORION-100M architecture and real forward/backward computation | IMPLEMENTED AND TESTED | 16 layers, width 640, 10 heads; 100,291,840 body parameters; CPU masking, padding and gradient tests |
| ORION-300M architecture and real forward/backward computation | IMPLEMENTED AND TESTED | 28 layers, width 960, 15 query/5 KV heads, tied embeddings; 306,763,200 unique parameters; causal-mask/cache tests |
| Tokenizer software | IMPLEMENTED, TRAINING REQUIRED | Local byte-BPE training and serialization are implemented; the v1 ID assignment is provisional and the final tokenizer is absent |
| ORION-100M RTD training | IMPLEMENTED, TRAINING REQUIRED | Generator/discriminator loss and tiny synthetic optimizer step pass; production data and weights are absent |
| ORION-100M supervised task training | IMPLEMENTED, TRAINING REQUIRED | Classification, multilabel, metadata-span and confidence losses plus resumable runner pass a synthetic CPU step |
| ORION-100M contrastive embeddings | IMPLEMENTED, TRAINING REQUIRED | Matryoshka InfoNCE at configured 640/256/128 dimensions; checksum-bound local JSONL runner and boundary-resume tests pass |
| ORION-100M registry-prototype/open-set training | IMPLEMENTED, TRAINING REQUIRED | Local tokenized registry profiles, sampled negative candidates, known-course classification and explicit unknown-margin loss pass a synthetic CPU step; unseen-course threshold calibration remains required |
| ORION-300M causal pretraining and SFT | IMPLEMENTED, TRAINING REQUIRED | Causal/z-loss and assistant-only SFT runners pass tiny synthetic CPU steps; no authorized corpus or trained model exists |
| ORION-300M E-300-1 architecture pilot | IMPLEMENTED, PARTIALLY VERIFIED | Decoder, prefix-LM and tied-embedding encoder-decoder candidates run on tiny synthetic paired-text fixtures; no approved-data architecture recommendation exists |
| Data pipeline | IMPLEMENTED, PARTIALLY VERIFIED | Local processing, manifests, provenance and acquisition tooling exist; no dataset download or corpus-scale run occurred |
| Training recovery | IMPLEMENTED, PARTIALLY VERIFIED | Single-process model/optimizer/scheduler/scaler/sampler/RNG recovery and resume equivalence pass; DDP/FSDP and multi-worker exact recovery are absent |
| SDK, CLI and REST | IMPLEMENTED, PARTIALLY VERIFIED | Eight wheels build and import in a clean local target; model CLI help/info and API contract tests pass; trained inference cannot run |
| GitHub and Hub release infrastructure | IMPLEMENTED, PARTIALLY VERIFIED | CI/workflows, explicit Hub helpers and release gates exist; CI was not run and no artifact was published |
| DDP/FSDP orchestration | NOT IMPLEMENTED | Single-process trainer only |
| Owner policy/authorization items | BLOCKED — EXPLAIN WHY | Data licensing/privacy, commercial-release, Urdu acceptance, compute budget and course-document authorization require owner decisions; see D8/D9/D10/D13 in `docs/architecture/decisions.md` |

## Verification performed

- `python -m pytest -q`: **51 passed**, with one environment warning for the
  deprecated `pynvml` package.
- `ruff check .`, `ruff format --check .`, and
  `python -m compileall -q packages tools tests`: passed.
- `python tools/count_params.py`: exact counts above; decoder embedding tying
  verified.
- All seven training/fine-tuning/pilot JSON examples validated, including
  contrastive, course-matching, SFT and E-300-1 configs.
- All eight wheels built with `pip wheel --no-deps --no-build-isolation`, then
  installed with `pip install --no-index --no-deps --target` into a clean local
  target. Imports and the generated model CLI entry-point help were exercised.
- Both CLI `info` responses report `trained: false` and
  `pretrained_weights_imported: false`.

These checks use existing workspace dependencies and local synthetic fixtures;
they do not constitute an isolated dependency-resolution test or production
training.

## Tooling and security findings

- `python -m mypy packages tests tools` exited with 171 errors across 48 files.
  The installed mypy is 2.1.0, outside the configured `<2` range; the run also
  reported missing third-party stubs, including `tokenizers` and `pyarrow`, as
  well as source typing errors. No clean strict type-check result is claimed;
  strict settings were retained.
- The global environment has pre-existing `pip check` conflicts across
  protobuf, decorator, httpx, urllib3, NumPy/OpenCV, packaging/cachetools,
  matplotlib/scikit-learn and tornado. No global dependencies were changed.
- `uv` was unavailable and its configured index did not provide it. Therefore
  there is no `uv.lock`; none was fabricated.
- GitHub Actions were not run. Dependency vulnerability scanning, external
  security audit, and formal legal/privacy reviews remain outstanding.
- Checkpoints use safetensors and validated JSON with SHA-256 manifests; unsafe
  pickle deserialization is not used. No Hub credentials or release upload were
  used.

## Before the separate training phase

1. Review and freeze the provisional tokenizer special-token map.
2. Resolve the D8/D9/D10/D13 policy decisions, document permissions, annotation
   plan and compute budget.
3. Prepare authorized registry examples for course-matcher training and
   calibrate it on unseen courses; execute E-300-1 on approved paired data and
   compute before selecting the 300M architecture.
4. Decide whether the target training deployment requires DDP/FSDP now; if so,
   implement and verify distributed checkpoint recovery before scaling up.
5. Produce an isolated, dependency-resolved installation; generate `uv.lock`
   when a compatible resolver is available; run CI and dependency/security
   scans.
6. Only after those gates, prepare authorized local data, train the approved
   tokenizer, run the architecture pilot, and then start separately authorized
   model training.

## Commands for a future authorized phase

These commands were **not executed** in this implementation phase. Replace the
example paths with reviewed local data and checkpoint locations. Dataset
acquisition additionally requires a separately authorized manifest and the
explicit `--execute` flag.

```powershell
python tools/validate_config.py --kind 100m-training configs/pretraining/orion-100m-rtd.json
python tools/validate_config.py --kind 300m-training configs/pretraining/orion-300m-causal.json
python tools/validate_config.py --kind 100m-finetuning configs/finetuning/orion-100m-multitask.json
python tools/validate_config.py --kind 100m-contrastive configs/finetuning/orion-100m-contrastive.json
python tools/validate_config.py --kind 100m-course-matching configs/finetuning/orion-100m-course-matching.json
python tools/validate_config.py --kind 300m-sft configs/finetuning/orion-300m-sft.json
python tools/validate_config.py --kind 300m-architecture-pilot configs/evaluation/e-300-1-architecture-pilot.json

orion-data acquire <approved-manifest.json> --output datasets/raw/<source> --mode commercial-clean --execute
orion-tokenizer train --corpus <approved-local-corpus> --output artifacts/orion-tokenizer-v1 --approve-token-map --confirm-training
orion100m train --tokens <prepared-token-shard> --word-ids <aligned-word-id-shard> --output checkpoints/orion-100m --config-json configs/pretraining/orion-100m-rtd.json --confirm-training
orion100m train-course-matching --registry-json <tokenized-registry.json> --data-jsonl <known-and-unknown-documents.jsonl> --labels-json <matching-task-labels.json> --init-checkpoint checkpoints/orion-100m-finetune/checkpoints/step-000000000100 --output checkpoints/orion-100m-course-matching --config-json configs/finetuning/orion-100m-course-matching.json --confirm-training
orion300m train --objective causal --tokens <prepared-token-shard> --output checkpoints/orion-300m --config-json configs/pretraining/orion-300m-causal.json --confirm-training
orion300m architecture-pilot --training-jsonl <approved-paired-train.jsonl> --validation-jsonl <approved-paired-validation.jsonl> --output experiments/tutor/e-300-1 --config-json configs/evaluation/e-300-1-architecture-pilot.json --confirm-training
```

The user-facing interfaces and remaining exact plan requirements are detailed
in `docs/cli/reference.md`, `docs/training/overview.md`,
`docs/implementation-status.md`, and the requirement traceability matrix.
