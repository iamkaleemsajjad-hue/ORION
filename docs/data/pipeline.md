# Data pipeline and provenance

Data commands operate on explicit local files. Acquisition is a separate,
opt-in step; preparing a dataset never downloads. The training preparation
command requires the optional tokenizer dependency and a locally installed,
hash-verified production ORION tokenizer:

```powershell
python -m pip install -e .\packages\orion-tokenizer -e ".\packages\orion-data[prepare]"
orion-data prepare .\datasets\manifests\approved-local.json --tokenizer .\artifacts\orion-tokenizer-v1 --output .\datasets\prepared\course-mix --config .\configs\data\orion-data-preparation.json --mode commercial-clean
orion-data verify-prepared .\datasets\prepared\course-mix
```

The command fails closed unless the manifest is non-synthetic, each source has
a local JSONL path and SHA-256, every source license is verified for the chosen
policy, required attribution metadata is present, and the tokenizer directory
contains a complete trained v1 artifact. Attribution metadata is a provenance
requirement; it does not establish that a source is legally usable.
Relative source paths resolve beside the manifest. No source document is
downloaded or discovered implicitly.

## Input records

Each source JSONL line is an object with:

- `document_id`: globally unique non-empty document ID.
- `text`: non-empty UTF-8 document text.
- `course_id`: optional course grouping key. When supplied, all documents
  for that course stay in one split. Otherwise splitting uses document ID.
- `split`: optional `train`, `validation` or `test`. If explicit
  splits are used for a course/document group, every record in the group must
  provide the same explicit split.
- Optional scalar metadata: `title`, `course_title`, `document_type`,
  `semester`, `date`, `page` and `slide`.

The manifest carries source ID, immutable revision, license, required
attribution where the reviewed policy calls for it, authorization context,
local path and expected SHA-256. The source file's hash is checked before
parsing. Document IDs are not used as output paths.

## Processing behavior

The validated config in `configs/data/orion-data-preparation.json` controls:

1. NFC normalization while retaining Urdu ZWNJ/ZWJ; optional Urdu variant
   folding is recorded as a deliberate transform.
2. PII masking for email, phone, Pakistan ID and credential-bearing URL query
   patterns; counts are reported without logging original text.
3. Quality filtering and structure-preserving document segmentation.
4. Deterministic course-level or document-level train/validation/test splits.
   Cross-split course assignments fail validation.
5. Exact duplicate removal, seeded MinHash near-duplicate checks, and
   13-gram contamination checks against held-out validation/test segments.
   Held-out copies take priority; matching training copies are excluded.
6. Tokenization with EOS separators and the exact local tokenizer hash.
7. Aligned uint16 token and int32 whole-word shards for the current model
   training commands, plus packed memmap sequences with document IDs and
   attention masks.
8. A prepared manifest and report containing counts, source/tokenizer identity,
   shard hashes and the effective configuration.

Output is written to a temporary sibling directory and renamed into place only
after all artifacts have been written. A non-empty output directory is never
overwritten. Run `verify-prepared` after preparation; it checks file hashes,
token/word alignment, record hashes, path safety, and course/document/text
split invariants.

Training inputs are available at:

- `train/tokens/tokens.u16` and `train/tokens/word_ids.i32` for ORION-100M.
- `train/tokens/tokens.u16` for ORION-300M.
- Equivalent paths under `validation` and `test` for evaluation.
- `<SPLIT>/packed/index.json` and its memmaps for boundary-aware readers.

Model training configs control DataLoader worker count, prefetch, and persistent
workers through `trainer.num_workers`, `trainer.prefetch_factor`, and
`trainer.persistent_workers`. Resumable workers require ordered output and the
ORION stateful sampler. Built-in map-style datasets are deterministic; custom
worker-side transforms must be stateless per sample when exact resume is
required.

The pipeline holds records and MinHash signatures in memory. Large-corpus
scaling, OCR workers, Parquet conversion, and licensed statistical language ID
remain infrastructure/data-dependent and have not been benchmarked. No real
dataset was acquired or prepared in this audit. Synthetic manifests are
explicitly rejected by the production preparation command.
