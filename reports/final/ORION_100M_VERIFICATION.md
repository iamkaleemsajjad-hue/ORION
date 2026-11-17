# ORION-100M audit

## Verified software

The configured encoder is 16 layers, hidden size 640, 10 heads, RoPE, GeGLU and pre-LayerNorm, with configured alternating local/global attention. `python tools/count_params.py` reports **100,291,840 unique parameters**, matching the plan target. The actual model supports bidirectional attention masks, padding, outputs and task heads; synthetic forward/backward and regression tests run in `tests/test_architectures.py`, `tests/test_supervised_training.py`, and related training tests.

Objective modules cover ELECTRA-style generator/discriminator replaced-token detection, MLM comparison, contrastive/Matryoshka representations, multi-task classification, span metadata extraction, course prototypes/open-set thresholds and calibration interfaces. Tiny synthetic updates verify computational and gradient paths; no claim is made about learned task quality.

## Remaining gates

No trained weights, real classification/metadata corpus, calibration split, course registry, or held-out quality results exist. Production tokenizer and dataset preparation are gated by owner-approved token IDs and authorized source manifests. GPU memory, throughput, and production sequence-length behavior have not been benchmarked.

Evidence: [`orion-100m-audit.md`](../implementation/orion-100m-audit.md), `python tools/count_params.py`, and `python -m pytest -q tests/test_architectures.py tests/test_supervised_training.py` (full-suite result in [verification log](FINAL_VERIFICATION_LOG.md)).
