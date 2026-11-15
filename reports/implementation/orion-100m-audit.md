# ORION-100M architecture and objective audit

## Architecture result

The configured model is a bidirectional encoder with 16 layers, width 640,
10 attention heads, RoPE, GeGLU and pre-LayerNorm. The executable parameter
counter reports **100,291,840** unique parameters, matching the target in the
technical plan. Architecture tests exercise dimensions, initialization,
attention and padding masks, gradients, task heads and configured sequence
behavior on synthetic CPU inputs.

## Training/software result

Implemented training modules cover ELECTRA-style generator/discriminator
replacement detection, the MLM comparison objective, contrastive and
Matryoshka representations, task losses, metadata spans, course prototypes,
open-set thresholds and calibration interfaces. Synthetic tests exercise
objective computations and gradient/update paths. They do not validate learned
document classification or calibration quality.

## Limits

No real pretraining/fine-tuning data, trained weights, held-out course registry
or quality evaluation exists. The source/prepared-data path requires an
owner-approved tokenizer and approved datasets. Exact unique-count evidence is
`python tools/count_params.py`; test results are in
`reports/tests/validation-report.md`.
