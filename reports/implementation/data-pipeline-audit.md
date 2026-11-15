# ORION data pipeline audit

Date checked: 2026-11-14. No dataset acquisition, ingestion of real records,
or tokenizer/model training was performed.

## Scope inspected

The audit read the dataset manifest and license models, acquisition command,
normalization and PII handling, quality filters, segmentation, exact and
near-duplicate logic, contamination checks, token alignment, packing, shard
read/write code, local dataset readers, tokenizer loading contract, data
preparation CLI, configuration, tests, and the data and governance guides.
The local pipeline uses explicit source files and performs no network access.

## Findings and corrections

| Finding | Evidence and root cause | Correction | Verification |
|---|---|---|---|
| The dataset CLI had no complete local preparation command at audit start. | The parser exposed manifest validation and explicit acquisition only; preparation primitives existed but no single checked path connected them. | Added `orion-data prepare` to load a checksum-pinned local JSONL manifest, enforce license policy, normalize/mask/filter/segment, assign deterministic course/document splits, deduplicate and check contamination, tokenize, write aligned shards and packed memmaps, and emit provenance/hash manifests. Added `orion-data verify-prepared`. | `tests/test_data_prepare.py`, `tests/test_training_data.py`; synthetic fixtures exercise the writer/verifier while production preparation rejects synthetic manifests. No real dataset was prepared. |
| License validation did not require attribution text when the project's own policy marked it required. | Executed the pre-change `DatasetManifest.validate_license_policy` from the starting Git tree: an MIT source with `verified=True` and empty attribution passed. The existing license registry marked MIT attribution required. | Validation now rejects blank attribution records for allowlisted entries marked `attribution_required`, in both research and commercial-clean modes. This checks presence only; human review remains necessary. | Regression test covers missing and present attribution. Local test suite rerun is recorded in the validation report. |
| The local `prepare` path accepted a remote-source kind if it happened to carry a local path. | `_load_source_documents` validated path/hash but did not constrain the source kind to its documented JSONL input contract. | Preparation now accepts only `local-jsonl`; acquisition remains a separate explicit command. | Regression test confirms a `huggingface-dataset` source cannot enter local preparation. |
| Prepared-data verification needed an explicit path-traversal regression. | The verifier checks manifest-relative paths for `..`, symlinks, and resolved containment; no direct regression covered a hostile path entry. | Added a fixture manifest containing `../outside.json` and verified rejection. | `test_prepared_verifier_rejects_manifest_path_traversal`. |

The attribution policy is deliberately only a data-quality gate. Creative Commons
BY 4.0 and MIT source terms include attribution or notice conditions; the
manifest cannot establish that a proposed source is authorized or that a text
field contains every required notice. The [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)
and [OSI MIT license](https://opensource.org/license/mit) were checked on the
date above. No legal opinion is provided.

## Implemented behavior

- Source manifests pin an ID, revision, local path, SHA-256, license evidence,
  attribution and authorization context. The local preparation flow checks
  the source checksum before parsing and does not discover or download sources.
- Course groups remain together during deterministic splitting. Conflicting
  explicit split labels fail. Held-out text takes priority during exact and
  near-duplicate handling, and known 13-gram matches are excluded from train.
- PII masking is optional by validated configuration and records counts rather
  than original sensitive values. Language hints are heuristic; no downloaded
  language-identification model is used.
- Prepared token IDs and whole-word IDs are aligned, packed memmaps have
  boundary IDs and masks, file hashes are recorded, and the verifier checks
  file integrity plus document/course/text split invariants.
- Output is staged in a sibling directory and renamed into place after
  artifacts are written. Existing non-empty destinations are refused.
- `orion-data prepare` requires the optional tokenizer extra and an installed,
  complete local production tokenizer. No network fallback exists.

## Verification limits

Only synthetic fixture data and synthetic tokenizer wrappers were used for
tests. The production CLI intentionally refuses synthetic dataset manifests.
The complete command could not be run end-to-end because there is no owner-
approved trained tokenizer and no real approved dataset.

All records, document IDs, segment lists, and MinHash indexes currently reside
in memory. Parquet writing is optional and separate from the JSONL-to-token
path. OCR extraction workers, corpus-scale streaming/deduplication, stratified
course splits, full language identification, legal/privacy review, large-data
leakage validation, and throughput/memory benchmarks remain unverified or
external. These limits block real-corpus readiness but do not require manual
source edits to invoke the implemented local preparation path after approval.
