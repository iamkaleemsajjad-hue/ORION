# Evaluation and benchmark audit

## Implemented evaluation software

`orion-eval` includes classification/multilabel/span metrics, AUROC/FPR-at-TPR, calibration measures, bootstrap intervals, retrieval/ranking support, split/sealed-set safeguards and model-specific evaluation paths. Tiny synthetic fixtures exercise computation. E-300-1 candidate evaluation/report code exists and is configurable.

## No empirical model results

There are no trained weights or production/approved evaluation datasets. Consequently no accuracy, F1, calibration quality, unknown-class performance, perplexity, generation quality, educational safety, real-data architecture comparison, contamination clearance, or benchmark target is reported. E-300-1 must compare candidates with approved identical splits/data budgets and documented objectives; no synthetic result selects the architecture. No CPU/GPU inference latency, tokens/sec, memory, loading time, quantization parity, or ONNX Runtime parity benchmark was executed. Hardware/precision/measurement protocol and results are absent because the required hardware/model artifacts do not exist here.

Run synthetic metric checks with `python -m pytest -q tests/test_evaluation.py tests/test_architecture_pilot.py`. For real evaluation, use `docs/evaluation/metrics.md` and approved configs only after trained artifacts and sealed splits are available. Do not interpret synthetic values as model performance.
