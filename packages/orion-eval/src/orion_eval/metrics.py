"""Numpy/Python evaluation metrics with no downloaded benchmark dependencies."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return accuracy and per-class, macro, and micro precision/recall/F1."""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("expected and predicted must have the same non-zero length")
    classes = list(labels) if labels is not None else sorted(set(expected) | set(predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    f1_values = []
    for label in classes:
        tp = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        fp = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        fn = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn
        f1_values.append(f1)
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
    return {
        "count": len(expected),
        "accuracy": sum(a == p for a, p in zip(expected, predicted, strict=True)) / len(expected),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "confusion": _confusion(expected, predicted, classes),
    }


def _confusion(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    counts = Counter(zip(expected, predicted, strict=True))
    return {actual: {guess: counts[(actual, guess)] for guess in labels} for actual in labels}


def multilabel_metrics(
    expected: Sequence[set[str]],
    predicted: Sequence[set[str]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float | int]:
    """Compute micro/macro F1, Hamming loss, and exact subset accuracy."""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("expected and predicted must have the same non-zero length")
    classes = (
        list(labels) if labels is not None else sorted(set.union(set(), *expected, *predicted))
    )
    tp = fp = fn = mismatches = 0
    f1s = []
    for label in classes:
        label_tp = sum(label in y and label in p for y, p in zip(expected, predicted, strict=True))
        label_fp = sum(
            label not in y and label in p for y, p in zip(expected, predicted, strict=True)
        )
        label_fn = sum(
            label in y and label not in p for y, p in zip(expected, predicted, strict=True)
        )
        precision = label_tp / max(label_tp + label_fp, 1)
        recall = label_tp / max(label_tp + label_fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        tp += label_tp
        fp += label_fp
        fn += label_fn
        mismatches += label_fp + label_fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "hamming_loss": mismatches / (len(expected) * max(len(classes), 1)),
        "subset_accuracy": sum(y == p for y, p in zip(expected, predicted, strict=True))
        / len(expected),
        "count": len(expected),
    }


def span_f1(
    expected: Sequence[tuple[int, int, str]], predicted: Sequence[tuple[int, int, str]]
) -> dict[str, float | int]:
    """Exact-match span precision/recall/F1 for (start, end, label) tuples."""
    expected_set, predicted_set = set(expected), set(predicted)
    overlap = len(expected_set & predicted_set)
    precision = overlap / max(len(predicted_set), 1)
    recall = overlap / max(len(expected_set), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "support": len(expected_set),
    }


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC by average ranks, including correct tie handling."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    positives = sum(bool(item) for item in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative examples")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and scores[order[end]] == scores[order[offset]]:
            end += 1
        average_rank = (offset + 1 + end) / 2
        rank_sum += average_rank * sum(bool(labels[index]) for index in order[offset:end])
        offset = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def fpr_at_tpr(
    labels: Sequence[int], scores: Sequence[float], *, target_tpr: float = 0.95
) -> float:
    """Return the lowest attainable false-positive rate at or above target TPR."""
    if not 0 < target_tpr <= 1 or len(labels) != len(scores):
        raise ValueError("invalid target_tpr or input lengths")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("FPR@TPR requires positive and negative examples")
    choices = []
    for threshold in sorted(set(scores), reverse=True):
        tp = sum(
            bool(label) and score >= threshold for label, score in zip(labels, scores, strict=True)
        )
        fp = sum(
            not bool(label) and score >= threshold
            for label, score in zip(labels, scores, strict=True)
        )
        if tp / positives >= target_tpr:
            choices.append(fp / negatives)
    return min(choices) if choices else 1.0


def bootstrap_interval(
    values: Sequence[T],
    metric: Callable[[Sequence[T]], float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> tuple[float, float]:
    """Percentile bootstrap interval over already paired observation records."""
    if not values or resamples <= 0 or not 0 < confidence < 1:
        raise ValueError(
            "bootstrap requires non-empty values, positive resamples, and confidence in (0,1)"
        )
    # Bootstrap resampling is statistical, reproducible sampling, not cryptography.
    generator = random.Random(seed)  # noqa: S311
    estimates = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(float(metric(sample)))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )
