"""Configurable task heads and weighted supervised losses for ORION-100M."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import EncoderOutput, Orion100MEncoder
from orion_100m.model.heads import (
    BIOExtractionHead,
    ConfidenceHead,
    MultiLabelHead,
    OpenSetMatcher,
    SequenceClassificationHead,
)


@dataclass(frozen=True)
class TaskOutput:
    """Structured raw task logits and representations; no labels are fabricated."""

    encoder_output: EncoderOutput
    subject_logits: torch.Tensor | None
    document_type_logits: torch.Tensor | None
    topic_logits: torch.Tensor | None
    multilabel_logits: torch.Tensor | None
    metadata_logits: torch.Tensor | None
    confidence_logits: torch.Tensor | None
    course_similarity_logits: torch.Tensor | None


class Orion100MForTasks(nn.Module):
    """Encoder plus task heads configured from the owner's versioned label maps."""

    def __init__(
        self,
        config: Orion100MConfig | None = None,
        *,
        subject_labels: int,
        document_type_labels: int,
        topic_labels: int,
        tag_labels: int,
        metadata_tags: int = 25,
    ) -> None:
        super().__init__()
        self.config = config or Orion100MConfig()
        if min(subject_labels, document_type_labels, topic_labels, tag_labels) <= 0:
            raise ValueError("all configured task head sizes must be positive")
        self.encoder = Orion100MEncoder(self.config)
        pooled_size = self.config.hidden_size * 2
        self.subject_head = SequenceClassificationHead(pooled_size, subject_labels)
        self.document_type_head = SequenceClassificationHead(pooled_size, document_type_labels)
        self.topic_head = SequenceClassificationHead(pooled_size, topic_labels)
        self.multilabel_head = MultiLabelHead(pooled_size, tag_labels)
        self.metadata_head = BIOExtractionHead(self.config.hidden_size, num_tags=metadata_tags)
        self.confidence_head = ConfidenceHead(pooled_size)
        self.course_matcher = OpenSetMatcher(self.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        course_prototypes: torch.Tensor | None = None,
    ) -> TaskOutput:
        encoded = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        prototypes = None
        if course_prototypes is not None:
            prototypes = self.course_matcher(encoded.pooled_output, course_prototypes)
        return TaskOutput(
            encoder_output=encoded,
            subject_logits=self.subject_head(encoded.cls_and_mean),
            document_type_logits=self.document_type_head(encoded.cls_and_mean),
            topic_logits=self.topic_head(encoded.cls_and_mean),
            multilabel_logits=self.multilabel_head(encoded.cls_and_mean),
            metadata_logits=self.metadata_head(encoded.last_hidden_state),
            confidence_logits=self.confidence_head(encoded.cls_and_mean),
            course_similarity_logits=prototypes,
        )


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_log_priors: torch.Tensor | None = None,
    logit_adjustment_tau: float = 1.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Cross entropy with optional class-prior correction and label smoothing."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("classification logits and labels have incompatible shapes")
    adjusted = logits
    if class_log_priors is not None:
        if class_log_priors.shape != (logits.shape[-1],):
            raise ValueError("class_log_priors must have one value per class")
        adjusted = logits + logit_adjustment_tau * class_log_priors.to(logits.device, logits.dtype)
    return F.cross_entropy(adjusted.float(), labels.long(), label_smoothing=label_smoothing)


def multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable sigmoid BCE loss for multi-label document tags."""
    if logits.shape != labels.shape:
        raise ValueError("multi-label logits and labels must have identical shapes")
    weights = (
        None if positive_weights is None else positive_weights.to(logits.device, torch.float32)
    )
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float(), pos_weight=weights)


def metadata_extraction_loss(
    logits: torch.Tensor,
    tag_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """BIO token-classification loss over non-padding positions."""
    if logits.shape[:2] != tag_ids.shape or attention_mask.shape != tag_ids.shape:
        raise ValueError("metadata logits, tags, and attention mask have incompatible shapes")
    targets = tag_ids.masked_fill(~attention_mask.to(torch.bool), ignore_index)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.long().reshape(-1),
        ignore_index=ignore_index,
    )


def multi_task_loss(
    output: TaskOutput,
    targets: dict[str, torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
    attention_mask: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the configured sum of only the tasks with targets in this batch."""
    coefficients = {
        "subject": 1.0,
        "document_type": 1.0,
        "topic": 1.0,
        "multilabel": 0.5,
        "metadata": 0.5,
        "confidence": 0.3,
        **(weights or {}),
    }
    logits_by_task = {
        "subject": output.subject_logits,
        "document_type": output.document_type_logits,
        "topic": output.topic_logits,
    }
    losses: dict[str, torch.Tensor] = {}
    for name, logits in logits_by_task.items():
        if name in targets:
            if logits is None:
                raise ValueError(f"task {name} has labels but no configured head")
            losses[name] = classification_loss(logits, targets[name])
    if "multilabel" in targets:
        if output.multilabel_logits is None:
            raise ValueError("multi-label targets were provided without a configured head")
        losses["multilabel"] = multilabel_loss(
            output.multilabel_logits, targets["multilabel"], positive_weights=positive_weights
        )
    if "metadata" in targets:
        if output.metadata_logits is None or attention_mask is None:
            raise ValueError("metadata loss needs its head and attention_mask")
        losses["metadata"] = metadata_extraction_loss(
            output.metadata_logits, targets["metadata"], attention_mask=attention_mask
        )
    if "confidence" in targets:
        if output.confidence_logits is None:
            raise ValueError("confidence targets were provided without a configured head")
        if output.confidence_logits.shape != targets["confidence"].shape:
            raise ValueError("confidence targets have incompatible shape")
        losses["confidence"] = F.binary_cross_entropy_with_logits(
            output.confidence_logits.float(), targets["confidence"].float()
        )
    if not losses:
        raise ValueError("at least one task target is required")
    total = sum(coefficients[name] * value for name, value in losses.items())
    return total, losses
