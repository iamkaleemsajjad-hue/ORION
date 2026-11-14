from __future__ import annotations

import pytest
import torch

from orion_eval.calibration import brier_score, expected_calibration_error, fit_temperature
from orion_eval.metrics import (
    binary_auroc,
    bootstrap_interval,
    classification_metrics,
    fpr_at_tpr,
    multilabel_metrics,
    span_f1,
)
from orion_eval.sealed import SealedEvaluationGuard, SealedSetAlreadyEvaluated


def test_classification_multilabel_and_span_metrics() -> None:
    result = classification_metrics(["math", "math", "history"], ["math", "science", "history"])
    assert result["count"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["confusion"]["math"]["science"] == 1
    multilabel = multilabel_metrics(
        [{"algebra", "exam"}, {"lecture"}],
        [{"algebra"}, {"lecture"}],
        labels=["algebra", "exam", "lecture"],
    )
    assert multilabel["micro_f1"] == pytest.approx(0.8)
    assert multilabel["subset_accuracy"] == pytest.approx(0.5)
    assert span_f1([(0, 4, "course")], [(0, 4, "course"), (5, 9, "date")])["f1"] == pytest.approx(
        2 / 3
    )


def test_open_set_metrics_bootstrap_and_sealed_guard(tmp_path) -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert binary_auroc(labels, scores) == pytest.approx(0.75)
    assert fpr_at_tpr(labels, scores, target_tpr=0.5) == pytest.approx(0.0)
    interval = bootstrap_interval(
        [0.0, 1.0, 1.0], lambda values: sum(values) / len(values), resamples=200
    )
    assert 0 <= interval[0] <= interval[1] <= 1

    guard = SealedEvaluationGuard(tmp_path / "sealed.json")
    guard.record(evaluation_id="fixture", split_sha256="a" * 64, release_tag="0.1-preview")
    with pytest.raises(SealedSetAlreadyEvaluated):
        guard.record(evaluation_id="fixture", split_sha256="a" * 64, release_tag="0.1-preview")


def test_calibration_metrics_and_temperature_fit() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    probabilities = torch.softmax(logits, dim=-1)
    ece = expected_calibration_error(probabilities, labels, bins=5)
    brier = brier_score(probabilities, labels)
    temperature = fit_temperature(logits, labels, max_iter=20)
    assert 0 <= ece <= 1
    assert 0 <= brier <= 2
    assert 0.05 <= temperature <= 20
