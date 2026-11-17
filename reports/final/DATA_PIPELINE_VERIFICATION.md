# Data pipeline audit

## Implemented software and test evidence

The pipeline has manifest/license/revision/hash schemas, explicit opt-in acquisition, local JSONL preparation, UTF-8/NFC normalization, heuristic language hints, optional PII masking, quality filtering, exact and near-deduplication interfaces, contamination checking, segmentation, token/word alignment, deterministic grouped splits, sequence packing, shard/memmap output, provenance manifests, verification, and data statistics. `orion-data prepare` and `verify-prepared` use explicit local inputs; imports and ordinary inference do not acquire data. Synthetic fixtures cover checksum validation, paths, splits, policies, token alignment and shard verification.

## Limitations

No dataset was downloaded or prepared. No license manifest can establish legal rights; attribution presence still needs human review. Corpus records and MinHash indexes are memory-resident and production-scale streaming/memory behavior is unmeasured. Language detection is heuristic, OCR extraction is not integrated into the standalone model SDK, PII detection has no measured false-negative rate, and no real-data contamination/leakage study exists. A trained, approved shared tokenizer and authorized data are prerequisites to the documented preparation path.

Evidence: [`data-pipeline-audit.md`](../implementation/data-pipeline-audit.md), `tests/test_data_prepare.py`, `tests/test_training_data.py`; no acquisition command was invoked.
