# Data pipeline and provenance

Data tools operate on local files and explicit manifests. The Hub snapshot
helper requires a license-policy mode, a pinned revision, and an explicit
network flag. It records per-file SHA-256 values in a local acquisition
manifest and was not invoked during software implementation. Training data
must be separated from sealed evaluation data by source document and source
course.

Each manifest entry records source ID, revision, license, path, SHA-256,
attribution, authorization where relevant, filters, and optional generator
provenance. The commercial-clean mode fails closed for unknown, non-commercial,
and share-alike licenses. This is a project policy gate, not legal advice.

Implemented primitives include NFC cleanup while preserving Urdu ZWNJ/ZWJ,
optional Urdu character folding as a named data transform, language hints,
quality checks, PII masking, exact deduplication, MinHash near-duplicate
candidates, 13-gram contamination checks, seeded OCR-noise augmentation,
deterministic splits, JSONL shards, and uint16 token memmaps. Large-scale
parquet extraction, OCR workers, and corpus-specific quality models need local
data and infrastructure and are not run here.

