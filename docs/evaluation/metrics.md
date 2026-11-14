# Evaluation metrics

The local evaluation package implements classification precision/recall/F1,
multilabel F1/Hamming/subset accuracy, exact span F1, AUROC, FPR@TPR, expected
calibration error, Brier score, temperature fitting, bootstrap intervals, and a
sealed split guard. Metrics are software functions, not achieved model results.

Run evaluations only after a versioned dataset manifest and trained artifact
exist. Use an untouched test split after calibration and threshold selection.
The plan's numerical targets remain acceptance thresholds and must be recorded
with actual sample counts, uncertainty intervals, and subgroup metrics.
