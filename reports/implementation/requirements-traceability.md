# Requirements traceability

The section numbers refer to the supplied *ORION AI — Technical Implementation
Plan*. “Tested” below means the described software behavior ran in a synthetic
or local test. It never means that ORION learned the capability or met a model
quality target.

| Plan section | Requirement | Software module(s) | Tests / evidence | Status |
|---:|---|---|---|---|
| 1 | Independent 100M encoder and 300M generator; no pretrained substitution | `orion_100m`, `orion_300m`, release metadata | `tests/test_independence.py`, missing-weight and release checks | IMPLEMENTED AND TESTED |
| 2 | Standalone, offline, provenance-first model family | SDK packages, `orion_data`, docs | import-boundary tests; SDK loads local paths only | IMPLEMENTED AND TESTED |
| 3 | Shared core and separate model weights | `orion_core`, both model packages | architecture and package-boundary tests | IMPLEMENTED AND TESTED |
| 4 | Document classification, matching, extraction, embeddings, unknown detection | `orion_100m` heads/SDK and `orion_eval` | architecture, evaluation and SDK contract tests; learned predictions need weights | IMPLEMENTED, TRAINING REQUIRED |
| 5 | Encoder design comparison and decision | architecture decision docs and config | rationale recorded; planned ablations were not run | IMPLEMENTED, PARTIALLY VERIFIED |
| 6 | 100M encoder dimensions, attention pattern, initialization and parameter count | `orion_100m.config`, `model.encoder` | `tests/test_architectures.py`, `python tools/count_params.py` | IMPLEMENTED AND TESTED |
| 7 | Shared tokenizer contract and stable token IDs | `orion_tokenizer`, core artifact checks | synthetic encode/decode, offsets, Unicode and ID-map tests; production map is provisional | IMPLEMENTED, TRAINING REQUIRED |
| 8 | Dataset choices, annotations, source licensing and provenance | `orion_data`, `datasets/manifests`, dataset cards | synthetic manifest and acquisition-policy tests; no corpus included | IMPLEMENTED, PARTIALLY VERIFIED |
| 9 | Data refinement, filtering, deduplication, segmentation and shards | `orion_data` pipeline/shards | synthetic normalization, quality, dedup, packing, acquisition and aligned-shard tests; large-corpus scale is unmeasured | IMPLEMENTED, PARTIALLY VERIFIED |
| 10 | ORION-100M RTD/GDES objective and pretraining | `orion_100m.training`, `orion_train` | loss/gradient tests and one tiny synthetic CPU step | IMPLEMENTED, TRAINING REQUIRED |
| 11 | ORION-100M multi-task fine-tuning, confidence and unknown detection | `orion_100m.finetuning`, `orion_100m.contrastive`, `orion_100m.course_matching`, task heads and `orion_eval` | synthetic multi-task, Matryoshka contrastive and known/unknown registry-prototype optimizer steps; calibration on unseen real courses remains required | IMPLEMENTED, TRAINING REQUIRED |
| 12 | Sealed evaluation, baselines, metrics and intervals | `orion_eval`, evaluation configs | metric, calibration, bootstrap and sealed-split tests; no held-out evaluation of trained ORION weights | IMPLEMENTED, TRAINING REQUIRED |
| 13 | ONNX export, quantization, cache and offline deployment | `tools/export_onnx.py`, SDK, generation cache | export tooling exists; trained-artifact parity and performance not measured | IMPLEMENTED, PARTIALLY VERIFIED |
| 14 | 300M educational tasks, grounding and abstention interfaces | `orion_300m` SDK/generation | synthetic decoding, no-context, citation and API tests; learned answers need trained weights | IMPLEMENTED, TRAINING REQUIRED |
| 15 | Decoder-only versus encoder-decoder/prefix-LM architecture arbiter | `orion_300m.architecture_pilot`, decoder prefix mode and encoder-decoder pilot | all three candidates run on the same synthetic source-target fixture with separate checkpoints and validation losses; approved-data E-300-1 results remain outstanding | IMPLEMENTED, PARTIALLY VERIFIED |
| 16 | 300M decoder dimensions, GQA, tying, initialization and count | `orion_300m.config`, `model.decoder` | `tests/test_architectures.py`, parameter-count report | IMPLEMENTED AND TESTED |
| 17 | Shared tokenizer and grounded chat/document template | tokenizer and `orion_300m.generation` | control-token and context/citation contract tests; final tokenizer artifact absent | IMPLEMENTED, TRAINING REQUIRED |
| 18 | 300M pretraining and instruction-data provenance | `orion_data`, manifests/cards | license and provenance policy tests; no datasets acquired | IMPLEMENTED, PARTIALLY VERIFIED |
| 19 | Causal corpus stages and structured SFT rendering | `orion_300m.sft`, data/shard tools | chat renderer, assistant-mask and local tokenized JSONL tests; real corpus stages remain unrun | IMPLEMENTED, PARTIALLY VERIFIED |
| 20 | Causal pretraining, z-loss and scaling ladder | `orion_300m.training`, shared losses | loss/gradient tests and one tiny synthetic CPU step; no scaling ladder run | IMPLEMENTED, TRAINING REQUIRED |
| 21 | SFT, abstention and citation shaping | `orion_300m.sft`, generation validators | assistant-only loss and tiny synthetic SFT step pass; no real SFT data or checkpoint | IMPLEMENTED, TRAINING REQUIRED |
| 22 | Grounded/human evaluation metrics and targets | `orion_eval`, evaluation docs | metrics execute on synthetic fixtures; quality targets remain unmeasured | IMPLEMENTED, TRAINING REQUIRED |
| 23 | KV cache, streaming, quantization and speed/quality | `orion_300m` generation, export tool | cached/uncached and streaming CPU tests; quantization and benchmarks not validated | IMPLEMENTED, PARTIALLY VERIFIED |
| 24 | Byte BPE, 32,768 IDs, reserved IDs, golden behavior | `orion_tokenizer` | synthetic byte/Unicode/Urdu/OCR/code tests and serialization checks; final tokenizer not trained | IMPLEMENTED, TRAINING REQUIRED |
| 25 | Core blocks, optimizer, configs, checkpoint/recovery, tracking | core/train packages | CPU math, update, integrity and resume-equivalence tests; external tracking is not integrated | IMPLEMENTED, PARTIALLY VERIFIED |
| 26 | Hardware and memory feasibility | config/docs and architecture counts | exact counts verified; no GPU, memory, cost or throughput benchmark | IMPLEMENTED, PARTIALLY VERIFIED |
| 27 | SDK, CLI, REST schemas, limits and errors | model packages and FastAPI apps | SDK/API tests include ordered generation batches, CLI help/info/config commands; no trained inference artifact | IMPLEMENTED, PARTIALLY VERIFIED |
| 28 | Monorepo and independently installable packages | eight package directories and workspace config | all eight wheels build and import from a workspace-local target; isolated dependency-resolving install remains | IMPLEMENTED, PARTIALLY VERIFIED |
| 29 | Optional isolated MADAD adapter with preserved host responsibilities | `orion_madad_adapter` | synthetic contract and standalone import tests; MADAD repository not present or modified | IMPLEMENTED AND TESTED |
| 30 | Licensing, privacy, safe serialization and local APIs | data policy, core serializer, APIs, security docs | path/license/input/checksum/missing-weight tests; formal legal/privacy review pending | IMPLEMENTED, PARTIALLY VERIFIED |
| 31 | Risk register, governance and truthful release claims | model cards, governance/release docs and gates | release verifier rejects missing/untrained artifacts; owner decisions remain | IMPLEMENTED, PARTIALLY VERIFIED |
| 32 | Automated architecture, tokenizer, training, inference, API and security QA | `tests/`, CI configuration | full local pytest suite and Ruff pass; CI and full security scans not run | IMPLEMENTED, PARTIALLY VERIFIED |
| 33 | Phase release criteria | `tools/verify_release.py`, model cards | verifier checks metadata and artifact integrity; no release satisfies trained-model criteria | IMPLEMENTED, PARTIALLY VERIFIED |
| 34 | Milestones and separation of implementation from training | milestone docs/configs | implementation phase documented; training/data acquisition commands not invoked | IMPLEMENTED, PARTIALLY VERIFIED |
| 35 | Architecture decisions and unresolved owner inputs | `docs/architecture/decisions.md` | D8/D9/D10/D13 and document permissions are listed explicitly | BLOCKED — EXPLAIN WHY: owner policy/authorization decisions are required before corpus preparation and the architecture pilot |
| 36 | Primary research and official API documentation | research ledgers | URLs, access dates, decisions and uncertainties recorded; independent source/API checks are not CI-enforced | IMPLEMENTED, PARTIALLY VERIFIED |

## Cross-cutting remaining requirements

- The v1 special-token assignment is an engineering inference because the plan's
  ID table is damaged. Owner review is required before freezing it.
- The `uv` resolver was unavailable in the environment, so `uv.lock` is absent.
- `python -m mypy packages tests tools` exited with 171 errors across 48 files.
  Installed mypy 2.1.0 is outside the configured `<2` range and reported
  missing third-party stubs, including `tokenizers` and `pyarrow`, plus source
  typing errors; a clean strict type-check result is unavailable.
- DDP/FSDP and multi-worker exact recovery are not implemented or tested.
- No real data, trained tokenizer, trained weights, quality results or hardware
  benchmark exists in this handover.
