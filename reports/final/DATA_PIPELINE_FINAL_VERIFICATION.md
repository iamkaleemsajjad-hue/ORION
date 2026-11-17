# Data pipeline final verification

The implemented local pipeline covers manifests/revisions/licenses, checksums, strict JSONL validation, normalization, heuristic language hints, quality filters, exact/near deduplication, contamination checks, configurable PII masking, segmentation, token/word alignment, deterministic grouped splits, packing, token shards, memmaps, provenance, verification, and data statistics. Dataset acquisition is explicit and requires an execution flag; no acquisition command was invoked.

Synthetic fixture tests in `tests/test_data_prepare.py` and `tests/test_training_data.py` run as part of the final suite (**89 passed, 2 skipped, 4 warnings**). They verify software behavior, not legal rights, production PII recall, corpus-scale memory, real leakage rates, or the accuracy of heuristic language detection.

Multi-worker consumed-cursor logic distinguishes dispatched/prefetched samples from samples acknowledged by the training loop. Its two-worker interrupted/resumed equivalence test is present but skipped on this Windows host because named-pipe creation is denied. Linux execution is still required to verify no lost/duplicated samples under real workers.

No real dataset was downloaded, licensed, split, or prepared. Production readiness requires approved manifests, verified licenses/attribution, a trained tokenizer v1 artifact, and running the complete validation pipeline on the authorized corpus. See [data audit](DATA_PIPELINE_AUDIT.md) and [data preparation guide](../../docs/data/pipeline.md).
