"""Evaluation metrics, calibration, and sealed-set safeguards."""

from orion_eval.calibration import brier_score, expected_calibration_error, fit_temperature
from orion_eval.metrics import bootstrap_interval, classification_metrics, multilabel_metrics
from orion_eval.sealed import SealedEvaluationGuard

__all__ = [
    "SealedEvaluationGuard",
    "bootstrap_interval",
    "brier_score",
    "classification_metrics",
    "expected_calibration_error",
    "fit_temperature",
    "multilabel_metrics",
]

__version__ = "0.1.0"
