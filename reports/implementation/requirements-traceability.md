# Requirements traceability

The section numbers refer to the supplied *ORION AI - Technical Implementation
Plan*. Each row maps the section's principal requirement to source, configuration,
tests, current state, and evidence. Sub-requirements and unresolved conditions
are called out in the cross-cutting notes and linked audit reports.
"Verified" means the described software behavior ran in a local or synthetic
check. It never means ORION learned the capability or met a model quality target.
Detailed remaining limitations are recorded in the final engineering handover.

| Requirement ID | Plan section | Requirement | Software module(s) | Tests / evidence | Status |
|---|---:|---|---|---|---|
| PLAN-01 | 1 | Independent 100M encoder and 300M generator; no pretrained substitution | `orion_100m`, `orion_300m`, release metadata | `tests/test_independence.py`, missing-weight and release checks | IMPLEMENTED AND VERIFIED |
| PLAN-02 | 2 | Standalone, offline, provenance-first model family | SDK packages, `orion_data`, docs | import-boundary tests; SDK loads local paths only | IMPLEMENTED AND VERIFIED |
| PLAN-03 | 3 | Shared core and separate model weights | `orion_core`, both model packages | architecture and package-boundary tests | IMPLEMENTED AND VERIFIED |
| PLAN-04 | 4 | Document classification, matching, extraction, embeddings, unknown detection | `orion_100m` heads/SDK and `orion_eval` | architecture, evaluation and SDK contract tests; learned predictions need weights | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-05 | 5 | Encoder design comparison and decision | architecture decision docs and config | rationale recorded; planned ablations were not run | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-06 | 6 | 100M encoder dimensions, attention pattern, initialization and parameter count | `orion_100m.config`, `model.encoder` | `tests/test_architectures.py`, `python tools/count_params.py` | IMPLEMENTED AND VERIFIED |
| PLAN-07 | 7 | Shared tokenizer contract and stable token IDs | `orion_tokenizer`, core artifact checks | tiny synthetic train/verify/export checks; production map remains provisional and owner-gated | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-08 | 8 | Dataset choices, annotations, source licensing and provenance | `orion_data`, `datasets/manifests`, dataset cards | synthetic manifest and acquisition-policy tests; no corpus included | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-09 | 9 | Data refinement, filtering, deduplication, segmentation and shards | `orion_data` pipeline, `prepare`, shard readers/writers | local checksum-pinned JSONL prepare and verify command exercised with synthetic fixtures; corpus scale and real-data leakage remain unmeasured | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-10 | 10 | ORION-100M RTD/GDES objective and pretraining | `orion_100m.training`, `orion_train` | loss/gradient tests and one tiny synthetic CPU step | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-11 | 11 | ORION-100M multi-task fine-tuning, confidence and unknown detection | `orion_100m.finetuning`, `orion_100m.contrastive`, `orion_100m.course_matching`, task heads and `orion_eval` | synthetic multi-task, Matryoshka contrastive and known/unknown registry-prototype optimizer steps; calibration on unseen real courses remains required | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-12 | 12 | Sealed evaluation, baselines, metrics and intervals | `orion_eval`, evaluation configs | metric, calibration, bootstrap and sealed-split tests; no held-out evaluation of trained ORION weights | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-13 | 13 | ONNX export, quantization, cache and offline deployment | `tools/export_onnx.py`, SDK, generation cache | export tooling exists; trained-artifact parity and performance not measured | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-14 | 14 | 300M educational tasks, grounding and abstention interfaces | `orion_300m` SDK/generation | synthetic decoding, no-context, citation and API tests; learned answers need trained weights | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-15 | 15 | Decoder-only versus encoder-decoder/prefix-LM architecture arbiter | `orion_300m.architecture_pilot`, decoder prefix mode and encoder-decoder pilot | all three candidates run on the same synthetic source-target fixture with separate checkpoints and validation losses; approved-data E-300-1 results remain outstanding | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-16 | 16 | 300M decoder dimensions, GQA, tying, initialization and count | `orion_300m.config`, `model.decoder` | `tests/test_architectures.py`, parameter-count report | IMPLEMENTED AND VERIFIED |
| PLAN-17 | 17 | Shared tokenizer and grounded chat/document template | tokenizer and `orion_300m.generation` | control-token and context/citation contract tests; final tokenizer artifact absent | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-18 | 18 | 300M pretraining and instruction-data provenance | `orion_data`, manifests/cards | license and provenance policy tests; no datasets acquired | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-19 | 19 | Causal corpus stages and structured SFT rendering | `orion_300m.sft`, data/shard tools | chat renderer, assistant-mask and local tokenized JSONL tests; real corpus stages remain unrun | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-20 | 20 | Causal pretraining, z-loss and scaling ladder | `orion_300m.training`, shared losses | loss/gradient tests and one tiny synthetic CPU step; no scaling ladder run | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-21 | 21 | SFT, abstention and citation shaping | `orion_300m.sft`, generation validators | assistant-only loss and tiny synthetic SFT step pass; no real SFT data or checkpoint | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-22 | 22 | Grounded/human evaluation metrics and targets | `orion_eval`, evaluation docs | metrics execute on synthetic fixtures; quality targets remain unmeasured | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-23 | 23 | KV cache, streaming, quantization and speed/quality | `orion_300m` generation, export tool | cached/uncached and streaming CPU tests; quantization and benchmarks not validated | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-24 | 24 | Byte BPE, 32,768 IDs, reserved IDs, golden behavior | `orion_tokenizer` | synthetic byte/Unicode/Urdu/OCR/code tests and serialization checks; final tokenizer not trained | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-25 | 25 | Core blocks, optimizer, configs, checkpoint/recovery, tracking | core/train packages | CPU math, update, integrity and resume-equivalence tests; external tracking is not integrated | PARTIALLY IMPLEMENTED |
| PLAN-25-DIST | 25 | DDP/FSDP, rank-aware training, and exact distributed recovery | orion_train.distributed, trainer.py, stateful samplers, and checkpoint metadata | Sampler sharding unit test passes; two-process exact-resume test is blocked on this Windows PyTorch Gloo backend; FSDP is not implemented | PARTIALLY IMPLEMENTED |
| PLAN-26 | 26 | Hardware and memory feasibility | config/docs and architecture counts | exact counts verified; no GPU, memory, cost or throughput benchmark | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-27 | 27 | SDK, CLI, REST schemas, limits and errors | model packages and FastAPI apps | SDK/API tests include ordered generation batches, CLI help/info/config commands; no trained inference artifact | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-28 | 28 | Monorepo and independently installable packages | eight package directories and workspace config | all eight wheels build and import from a workspace-local target; isolated dependency-resolving install remains | PARTIALLY IMPLEMENTED |
| PLAN-29 | 29 | Optional isolated MADAD adapter with preserved host responsibilities | `orion_madad_adapter` | synthetic contract and standalone import tests; MADAD repository not present or modified | IMPLEMENTED AND VERIFIED |
| PLAN-30 | 30 | Licensing, privacy, safe serialization and local APIs | data policy, core serializer, APIs, security docs | path/license/input/checksum/missing-weight tests; formal legal/privacy review pending | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-31 | 31 | Risk register, governance and truthful release claims | model cards, governance/release docs and gates | release verifier rejects missing/untrained artifacts; owner decisions remain | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-32 | 32 | Automated architecture, tokenizer, training, inference, API and security QA | `tests/`, CI configuration | full local pytest suite and Ruff pass; CI and full security scans not run | PARTIALLY IMPLEMENTED |
| PLAN-33 | 33 | Phase release criteria | `tools/verify_release.py`, model cards | verifier checks metadata and artifact integrity; no release satisfies trained-model criteria | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-34 | 34 | Milestones and separation of implementation from training | milestone docs/configs | implementation phase documented; training/data acquisition commands not invoked | IMPLEMENTED BUT NOT FULLY VERIFIED |
| PLAN-35 | 35 | Architecture decisions and unresolved owner inputs | `docs/architecture/decisions.md` | D8/D9/D10/D13 and document permissions are listed explicitly | BLOCKED BY EXTERNAL RESOURCE |
| PLAN-36 | 36 | Primary research and official API documentation | research ledgers | URLs, access dates, decisions and uncertainties recorded; independent source/API checks are not CI-enforced | IMPLEMENTED BUT NOT FULLY VERIFIED |

## Cross-cutting remaining requirements

- Tokenizer v1 special-token IDs remain provisional. The owner approval record
  must be completed before training or freezing v1.0. The training CLI flag is
  an operator assertion and does not authenticate owner approval.
- No `uv.lock` is present and `uv` is unavailable. Attempts to install `uv`
  and the required Tokenizers version failed because the configured package
  index returned no distributions. Dependency resolution and a clean install
  remain unverified.
- Baseline strict mypy produced 171 diagnostics across 48 files. The actual
  first-party source typing defects were corrected. With pinned mypy 2.1.0 and a
  narrow missing-stub fallback for Tokenizers, strict mypy passes all 121
  analyzed source files.
- The Tokenizers requirement is `>=0.23.3,<0.24` for upstream typing support.
  The current interpreter has Tokenizers 0.22.2; local tokenizer tests ran
  against that installed runtime. The required version has not been installed
  or verified in an isolated environment.
- DDP orchestration, deterministic rank sharding, rank-zero checkpoint writing,
  and per-rank RNG/data-state recovery are implemented. The sampler sharding
  test passes. The two-process Gloo gradient-sync/resume test is skipped on this
  Windows PyTorch build because it cannot create a Gloo network device. FSDP is
  not implemented; multi-worker exact recovery is not verified.
- Latest local checks: pytest 77 passed and 1 platform-specific DDP test was
  skipped; Ruff lint/format, strict mypy, compileall, model parameter counts,
  tokenizer/data/preflight synthetic tests and config validation pass.
  PyTorch emits a `pynvml` deprecation warning.
- The `orion-train info` checkpoint layout mismatch was reproduced, corrected,
  and regression-tested against an actual tiny checkpoint.
- `pip check` fails on conflicts in unrelated globally installed packages. No
  clean dependency-resolving install or vulnerability scanner ran. The GitHub
  security workflow exists but was not executed. There are no real datasets,
  trained tokenizer, trained weights, model quality results, CUDA/NCCL
  verification, FSDP verification, or formal security/legal/privacy review.
- `orion preflight` and `orion-data prepare/verify-prepared` now exist.
  Source preflight checks package imports and configuration contracts, but the
  current environment fails its dependency gate because Tokenizers 0.22.2 is
  installed while the package requires >=0.23.3,<0.24; tokenizer owner
  approval is also reported pending. The data command is
  verified only on synthetic local fixtures. Clean installation, approved
  corpus preparation, and full authorized training workflows remain unverified.
- No commit, push, tag, release, upload, or publication was performed.

## Configuration and discrepancy crosswalk

This second index completes the per-section mapping: source modules, tests and
status are in the primary matrix above; configuration/decision inputs and the
specific gap, correction or unverified claim are recorded here. A dash means
the requirement is a code contract or governance document rather than a
runtime JSON recipe.

| Requirement ID | Configuration / decision input | Defect, resolution, remaining limit |
|---|---|---|
| PLAN-01 | `orion_100m.config`, `orion_300m.config`, release manifests | Independent architectures and local-only loading are present; no weights are claimed. |
| PLAN-02 | `configs/deployment/local-api.json` | SDKs do not fetch weights; network-denial monitoring remains unverified. |
| PLAN-03 | Typed model configs and tokenizer release manifest | Shared code is separated from model weights; package boundaries are covered by import tests. |
| PLAN-04 | 100M fine-tuning recipes; `configs/evaluation/default.json` | Task APIs/heads exist; classification, calibration and unknown detection quality require trained weights and held-out data. |
| PLAN-05 | `reports/implementation/decision-records.md` | Candidate reasoning is documented; real ablation results were not run. |
| PLAN-06 | `configs/pretraining/orion-100m-rtd.json`, `orion_100m.config` | Count and CPU behavior match; initialization quality and planned local/global ablations need training experiments. |
| PLAN-07 | `configs/tokenizer/byte-bpe-v1.json`, `STATUS.json` | Added explicit owner gate and map digest; the map remains provisional and production tokenizer absent. |
| PLAN-08 | `datasets/manifests/`, `datasets/cards/` | Source records must be created from authorized real sources; none were downloaded. |
| PLAN-09 | `configs/data/orion-data-preparation.json` | Added `prepare`/`verify-prepared` and checksum/policy checks; pipeline holds records in memory and is not corpus-scale benchmarked. |
| PLAN-10 | `configs/pretraining/orion-100m-rtd.json` | Synthetic objective/gradient path passes; no RTD pretraining run or scaling evidence exists. |
| PLAN-11 | `configs/finetuning/orion-100m-{multitask,contrastive,course-matching}.json` | Synthetic updates run; unseen-course thresholds and confidence calibration remain data/training dependent. |
| PLAN-12 | `configs/evaluation/default.json` | Evaluation tools execute on fixtures; sealed quality targets/baselines remain unmeasured. |
| PLAN-13 | `configs/deployment/local-api.json`, `tools/export_onnx.py` | Export/deployment tooling exists; ONNX parity, quantization and speed are not measured. |
| PLAN-14 | `configs/deployment/local-api.json`, 300M generation defaults | Grounding/abstention/citation contracts are tested; generated answers require actual trained weights and host verification. |
| PLAN-15 | `configs/evaluation/e-300-1-architecture-pilot.json` | Added/runs synthetic candidate framework; production architecture selection awaits fair approved-data experiment. |
| PLAN-16 | `configs/pretraining/orion-300m-causal.json`, `orion_300m.config` | Exact tied-parameter count and synthetic causal/cache tests pass; quality and target-hardware memory remain unknown. |
| PLAN-17 | Shared tokenizer recipe and v1 registry | Both models validate shared tokenizer metadata; final ID map requires owner decision. |
| PLAN-18 | Dataset manifests/cards and `configs/pretraining/orion-300m-causal.json` | Provenance interfaces exist; no corpus or instruction data acquired. |
| PLAN-19 | `configs/data/orion-data-preparation.json`, SFT writer APIs | Local transforms/rendering are tested synthetically; no real corpus/shards produced. |
| PLAN-20 | `configs/pretraining/orion-300m-causal.json` | Causal loss path executes on tiny synthetic batch; scaling ladder not run. |
| PLAN-21 | `configs/finetuning/orion-300m-sft.json` | Assistant-only target mask and tiny update pass; no real SFT corpus/model. |
| PLAN-22 | `configs/evaluation/default.json` | Metrics run on fixtures; acceptance targets are not results. |
| PLAN-23 | `configs/deployment/local-api.json` and model generation config | KV-cache correctness is tested; GPU, ONNX, quantization and latency claims remain unverified. |
| PLAN-24 | `configs/tokenizer/byte-bpe-v1.json`, registry, approval and release status | Fixed `Regex` API usage and atomic staged release; final trained 32,768-token v1 is pending approval/data. |
| PLAN-25 | `TrainerConfig`; model-specific pretraining/fine-tuning recipes | Single-process resume tests pass; external tracking, FSDP and prefetch-exact recovery are not implemented. |
| PLAN-25-DIST | `TrainerConfig.distributed_*` fields | DDP code and sampler tests exist; Gloo integration is skipped on Windows. FSDP needs optimizer-builder/checkpoint lifecycle redesign and accelerator verification. |
| PLAN-26 | Both pretraining configs and hardware docs | Counts are measured; memory/time/cost entries are planning assumptions without target-GPU benchmark. |
| PLAN-27 | `configs/deployment/local-api.json` and SDK schemas | API/SDK synthetic contracts pass; installed-mode package/entry-point checks await a clean dependency install. |
| PLAN-28 | Root `pyproject.toml` and all eight package `pyproject.toml` files | Package boundaries/wheels exist; no genuine `uv.lock` or dependency-resolving clean install. |
| PLAN-29 | `orion-madad-adapter` contracts | Adapter tests use synthetic payloads; MADAD source is absent and untouched. |
| PLAN-30 | Deployment config, dataset license/provenance manifests, security docs | Added path/hash/input safeguards; scanners and legal/privacy review remain external. |
| PLAN-31 | Governance docs, model cards, release verifier | Release verifier rejects absent/untrained artifacts; no release or benchmark claims exist. |
| PLAN-32 | `pytest.ini`, root quality config and `.github/workflows/` | Local 77/1 suite and static checks pass; GitHub workflows/security scans were not run. |
| PLAN-33 | Release metadata and `tools/verify_release.py` | Artifact validation exists; neither model meets trained release gate. |
| PLAN-34 | Development roadmap and documented CLIs | Milestones/commands are documented; tokenizer/data/model production commands were not invoked. |
| PLAN-35 | Decision records and tokenizer owner approval record | Owner-controlled token mapping remains pending; architecture pilot choice remains pending real evidence. |
| PLAN-36 | `docs/research/research-ledger.md` | Primary sources and checked dates are recorded; source/API drift is not automatically CI-enforced. |

The `PLAN-11` brace expression above denotes the three separate JSON files
listed in the configuration inventory; it is shorthand, not a literal CLI
path. Individual statuses and execution evidence remain in the primary table.
