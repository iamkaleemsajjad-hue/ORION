"""Calibration metrics and held-out temperature scaling."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    """Top-label ECE with equal-width confidence bins in [0, 1]."""
    if probabilities.ndim != 2 or labels.ndim != 1 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must be [batch, classes] and labels [batch]")
    if bins <= 0 or torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("invalid bin count or probability values")
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels).to(probabilities.dtype)
    total = probabilities.shape[0]
    error = probabilities.new_zeros(())
    edges = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            selected |= confidence == 0
        if selected.any():
            error += (
                selected.sum()
                / total
                * (confidence[selected].mean() - correct[selected].mean()).abs()
            )
    return float(error.detach().cpu())


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass Brier score, averaged across examples."""
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels have incompatible shapes")
    target = F.one_hot(labels.to(torch.long), num_classes=probabilities.shape[-1]).to(
        probabilities.dtype
    )
    return float((probabilities - target).pow(2).sum(dim=-1).mean().detach().cpu())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 100) -> float:
    """Fit one scalar temperature on an explicitly supplied calibration split."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [batch, classes] and labels [batch]")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    log_temperature = torch.zeros((), dtype=torch.float64, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    values = logits.detach().double()
    targets = labels.detach().long()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(values / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())
