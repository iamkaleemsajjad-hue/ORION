"""Label-oriented facade over the canonical ORION-100M task model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import EncoderOutput
from orion_100m.model.tasks import Orion100MForTasks


@dataclass(frozen=True)
class TaskLabelMap:
    """Human-readable task labels used to size task heads at construction time."""

    subjects: tuple[str, ...]
    document_types: tuple[str, ...]
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    bio_tags: tuple[str, ...] = (
        "O",
        "B-title",
        "I-title",
        "B-course_code",
        "I-course_code",
        "B-course_name",
        "I-course_name",
        "B-semester",
        "I-semester",
        "B-instructor",
        "I-instructor",
        "B-date",
        "I-date",
    )

    def validate(self) -> None:
        if len(self.subjects) < 2 or len(self.document_types) < 2:
            raise ValueError("subject and document-type label maps require at least two labels")
        for name, labels in (
            ("subjects", self.subjects),
            ("document_types", self.document_types),
            ("topics", self.topics),
            ("tags", self.tags),
            ("bio_tags", self.bio_tags),
        ):
            if len(labels) != len(set(labels)):
                raise ValueError(f"{name} labels must be unique")
        if len(self.bio_tags) < 3 or self.bio_tags[0] != "O":
            raise ValueError("bio_tags must begin with O and include at least one span type")


@dataclass(frozen=True)
class DocumentTaskOutput:
    encoder: EncoderOutput
    logits: dict[str, torch.Tensor]


class Orion100MDocumentModel(nn.Module):
    """Provide label metadata while delegating all network computation to task heads."""

    def __init__(self, labels: TaskLabelMap, config: Orion100MConfig | None = None) -> None:
        super().__init__()
        labels.validate()
        self.config = config or Orion100MConfig()
        self.labels = labels
        self.tasks = Orion100MForTasks(
            self.config,
            subject_labels=len(labels.subjects),
            document_type_labels=len(labels.document_types),
            topic_labels=max(2, len(labels.topics)),
            tag_labels=max(1, len(labels.tags)),
            metadata_tags=len(labels.bio_tags),
        )

    @property
    def encoder(self) -> nn.Module:
        return self.tasks.encoder

    @property
    def prototype_matcher(self) -> nn.Module:
        return self.tasks.course_matcher

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> DocumentTaskOutput:
        output = self.tasks(input_ids, **kwargs)
        logits = {
            "subject": output.subject_logits,
            "doc_type": output.document_type_logits,
            "topic": output.topic_logits,
            "tags": output.multilabel_logits,
            "metadata": output.metadata_logits,
            "confidence": output.confidence_logits,
        }
        return DocumentTaskOutput(
            output.encoder_output,
            {key: value for key, value in logits.items() if value is not None},
        )

    def get_embeddings(
        self, input_ids: torch.Tensor, *, output_dim: int = 640, **kwargs: object
    ) -> torch.Tensor:
        if output_dim not in {128, 256, 640} or output_dim > self.config.hidden_size:
            raise ValueError("embedding dimension must be a supported prefix of encoder width")
        encoded = self.tasks.encoder(input_ids, **kwargs)
        return torch.nn.functional.normalize(encoded.pooled_output[..., :output_dim], dim=-1)


__all__ = ["DocumentTaskOutput", "Orion100MDocumentModel", "TaskLabelMap"]
