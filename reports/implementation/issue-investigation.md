# Issue investigation and correction record

All corrections below followed the audit-first procedure: reproduce or inspect
the source path, read related configs/tests/docs, identify the root cause, then
change code and add or run regression verification. No training or data
acquisition commands were invoked.

## 1. Strict typing findings

- **Original finding:** Prior handover reported 171 mypy errors across 48 files
  and attributed them to a mypy version mismatch.
- **Genuine?** Yes, many diagnostics were first-party typing defects. The
  version-mismatch explanation was false: workspace configuration allowed
  mypy 2.1.0 with `>=1.11,<3`.
- **Evidence:** Reproduced with Python 3.10.11, mypy 2.1.0 and the checked-in
  strict configuration. Diagnostics included missing annotations, incorrect
  generic/protocol types, optional handling, and Tensor/NumPy/API contract
  typing; Tokenizers 0.22.2 also lacked stubs.
- **Files read:** Root and eight package `pyproject.toml` files, mypy config,
  source modules reported by mypy, associated schemas and tests, and the prior
  handover.
- **Research:** No architecture change was necessary. The installed compiler
  and declared version range were verified locally; stub availability was
  checked against Tokenizers' official release/source information.
- **Fix:** Corrected first-party annotations and types, pinned mypy 2.1.0, and
  retained strict mode. The only missing-import fallback is limited to
  `tokenizers` and its submodules.
- **Regression/verification:** `python -m mypy packages tests tools
  --show-error-codes` passes with no issues in 121 analyzed source files.
- **Remaining:** The intended Tokenizers >=0.23.3 package could not be installed
  from the configured index, so clean-environment typing is not verified.

## 2. Production tokenizer vocabulary acceptance

- **Original finding:** A reduced vocabulary could be marked production
  `trained` if the token-map approval flag was true.
- **Genuine?** Yes. The trainer validated the maximum but not the production
  exact vocabulary.
- **Files read:** `orion_tokenizer/training.py`, tokenizer ID spec/constants,
  tokenizer config, release `STATUS.json`, tokenizer docs and
  `tests/test_tokenizer.py`.
- **Research:** The 32,768 target and reserved ranges come from the technical
  plan; Tokenizers is used only as the local BPE implementation. Final ID
  spelling remains owner-controlled because the supplied plan table is
  damaged.
- **Fix:** Production status now requires exactly 32,768 IDs. Reduced
  vocabulary is allowed only with the explicit synthetic-test status.
- **Regression/verification:** Added test for reduced production rejection;
  tokenizer tests pass. No tokenizer training was run.

## 3. ORION-300M end-token validation

- **Original finding:** The configured end/EOS token was omitted from the
  vocabulary-bound check.
- **Genuine?** Yes; an end token equal to vocabulary size could pass config
  validation and fail later in embedding/logit use.
- **Files read:** `orion_300m/config.py`, model special-token definitions,
  model configs, architecture tests and 300M docs.
- **Fix:** The configured end token participates in the same special-ID range
  validation as other configured IDs.
- **Regression/verification:** Added a config test rejecting
  `end_token_id == vocab_size`; architecture tests pass.

## 4. Urdu tokenizer fixtures

- **Original finding:** Test and golden-vector input intended to cover Urdu was
  malformed text rather than Urdu code points.
- **Genuine?** Yes; the source display had mojibake. Python UTF-8 inspection
  confirmed the malformed fixture.
- **Files read:** `tests/test_tokenizer.py`, tokenizer golden-vector writer,
  normalization/config paths and tokenizer docs.
- **Fix:** Replaced fixture text with explicit Unicode escapes and kept the
  actual Urdu string in generated golden examples.
- **Regression/verification:** Tests now cover Urdu code points, Unicode
  normalization and language hints; tokenizer suite passes.

## 5. Training CLI checkpoint metadata

- **Original finding:** `orion-train info` listed `state.json` and omitted
  actual checkpoint files.
- **Genuine?** Yes; reproduced the CLI JSON, read the full checkpoint writer
  and Safetensors serializer, and compared a generated checkpoint directory.
  The directory has model/optimizer/RNG Safetensors plus three checksum
  sidecars, `training_state.json`, and `manifest.json`.
- **Files read:** `orion_train/cli.py`, `orion_train/checkpoint.py`,
  `orion_core/serialization.py`, `orion_core/hashing.py`,
  `docs/training/checkpoints.md`, and related tests.
- **Fix:** Updated the CLI's checkpoint file list and accepted an argv list for
  deterministic CLI testing.
- **Regression/verification:** New test writes a real tiny checkpoint and
  compares the CLI list with the directory contents; passes.

## 6. Missing DDP orchestration

- **Original finding:** The previous handover said distributed training was
  absent.
- **Genuine?** Yes at the starting audit state. The final audit added DDP
  orchestration and deterministic per-rank state handling.
- **Files read:** Training config, trainer, sampler, checkpoint implementation,
  model training callers, DDP tests and checkpoint docs.
- **Research:** Reviewed the official
  [PyTorch DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
  [distributed sampler documentation](https://docs.pytorch.org/docs/main/utils.data.html#torch.utils.data.distributed.DistributedSampler),
  [FSDP documentation](https://docs.pytorch.org/docs/main/distributed/fsdp.html),
  and [distributed checkpoint API](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html).
- **Fix:** Added opt-in DDP initialization/wrapping, stateful rank sharding,
  synchronized losses/gradients, rank-zero writes, per-rank RNG/sampler/scaler
  checkpoint state, and recovery world-size checks.
- **Regression/verification:** Sampler rank-sharding test passes. A real
  two-process Gloo gradient-sync/resume test is present but skipped on this
  Windows PyTorch build, which failed to create a Gloo network device.
- **Remaining:** DDP runtime is not verified locally. FSDP and exact
  multi-worker prefetch recovery are not implemented.
  Source inspection also confirmed FSDP cannot be introduced by wrapping the
  current trainer only: every model-specific builder creates AdamW before
  `Trainer`, while FSDP parameter wrapping must precede optimizer creation;
  `CheckpointManager` currently writes full ordinary model state and maps
  optimizer state by parameter identity/name. A distributed full/sharded
  checkpoint lifecycle and accelerator verification are required. A direct
  FSDP1 probe fails because this PyTorch build has no non-CPU accelerator; a
  one-rank FSDP2 CPU/Gloo probe fails during process-group setup with
  `unsupported gloo device`. WSL enumeration is denied on this host.

## 7. Report contradictions

- **Original finding:** Prior status and repository-audit reports contradicted
  the tracked workspace about Git metadata, source files, mypy constraints,
  test totals and distributed status.
- **Genuine?** Yes; verified against Git history/status, pyprojects, source and
  executed test outcomes.
- **Fix:** Reconciled the repository audit, traceability, implementation
  status, validation report and handover. The baseline inventory remains
  separately identified as the pre-edit state.

## 8. Tokenizer CLI and failed-run release integrity

- **Original finding:** The tokenizer CLI exposed training only, the recipe was
  not consumed by the command, and a failed BPE run could leave a partially
  written output directory. The training path also referenced
  `pre_tokenizers.Regex`, which is not the Tokenizers API namespace.
- **Genuine?** Yes. Inspecting the entire CLI, trainer, recipe, package
  dependencies, tokenizer tests and training documentation showed missing
  validate/verify/export commands and direct writes to the final output path.
  Reproducing a tiny synthetic training run raised
  `AttributeError: module 'tokenizers.pre_tokenizers' has no attribute 'Regex'`
  under installed Tokenizers 0.22.2.
- **Research:** The official v0.23.3 pre-tokenizer API specifies the top-level
  `tokenizers.Regex` passed into `pre_tokenizers.Split`.
- **Fix:** Added a validated pending recipe model, explicit
  `validate-config`, `train --config`, `verify`, and local `export` commands.
  Training now stages into a sibling directory and atomically renames only on
  success. Release verification checks the tokenizer and special-token hashes,
  reserved and byte ID ranges, config, and golden encodings. The trainer uses
  the documented `Regex` import.
- **Regression/verification:** Added tiny 512-ID synthetic tokenizer training
  and failed-JSONL cleanup tests. The artifact is explicitly labeled
  `synthetic-test-fixture`, is refused by production verify/export, and no
  final tokenizer was trained.
- **Remaining:** Successful production verify/export cannot be exercised until
  the owner approves and trains the final tokenizer. Tokenizers 0.23.3 and a
  clean package installation remain unavailable in this environment.

## 9. Local data preparation and license attribution

- **Original finding:** Manifest validation did not require attribution text
  even when the repository's reviewed local license policy marked it required;
  no single CLI path connected local manifest validation to preparation and
  integrity verification.
- **Genuine?** Yes. The baseline validator accepted a verified MIT record with
  empty attribution. The dataset CLI exposed validation/acquisition while
  processing primitives existed separately.
- **Files read:** `orion_data/manifest.py`, license registry, acquisition,
  normalization, quality filters, segmentation, deduplication, splitting,
  tokenization, shard readers/writers, data CLI, preparation config,
  documentation and data tests.
- **Research:** Reviewed the primary CC BY 4.0 legal code and OSI MIT license
  to check the attribution/notice rationale. This does not establish dataset
  rights or provide legal advice; URLs and access date are in the research
  ledger.
- **Fix:** Require non-empty attribution where policy says it is required;
  restrict preparation to local JSONL; add `orion-data prepare` and
  `verify-prepared`, with checksums, split/dedupe/contamination validation,
  token/word alignment, packed shards, provenance and path containment.
- **Regression/verification:** Synthetic tests cover missing/present
  attribution, remote-kind rejection, preparation, verifier integrity and
  traversal. No real dataset was prepared.
- **Remaining:** Owner-approved source rights and tokenizer are prerequisites;
  corpus-scale memory/throughput and human legal/privacy review remain.

## 10. Unified preflight and tokenizer approval gates

- **Original finding:** No unified preflight entry point verified all package
  imports and checked-in recipes, while the provisional token map could be
  treated as approved by only a command-line flag.
- **Genuine?** Yes; no `orion` console command or matching tests existed at
  audit start. The token map source plan was damaged and needed an explicit
  owner decision.
- **Fix:** Added read-only `orion preflight` with source/installed modes and
  optional model-data-tokenizer readiness checks. Added machine/doc approval
  record verification and map-digest checks before production tokenizer
  training.
- **Regression/verification:** Source preflight executed package/config
  checks and returned blocked on the actual Tokenizers 0.22.2 vs >=0.23.3
  dependency mismatch. Tests prove pending token approval stops before output
  creation. No production training was invoked.
- **Remaining:** Clean installed mode awaits package resolution; local approval
  fields are auditable assertions, not identity authentication.

## Unresolved audit limitations

No clean dependency-resolving install, full dependency vulnerability scan,
security workflow, CUDA/NCCL test, FSDP run, corpus-scale data validation,
trained-weight inference, performance benchmark, or formal privacy/legal
review was possible or performed. These are recorded as incomplete, not passed.
