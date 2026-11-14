"""ORION-100M encoder architecture and task-specific modules."""

from orion_100m.model.encoder import EncoderOutput, Orion100MEncoder
from orion_100m.model.heads import (
    BIOExtractionHead,
    MultiLabelHead,
    OpenSetMatcher,
    SequenceClassificationHead,
)
from orion_100m.model.losses import (
    confidence_loss,
    contrastive_info_nce,
    course_open_set_loss,
    masked_language_model_loss,
    matryoshka_info_nce,
    replaced_token_detection_loss,
)

__all__ = [
    "BIOExtractionHead",
    "EncoderOutput",
    "MultiLabelHead",
    "OpenSetMatcher",
    "Orion100MEncoder",
    "SequenceClassificationHead",
    "confidence_loss",
    "contrastive_info_nce",
    "course_open_set_loss",
    "masked_language_model_loss",
    "matryoshka_info_nce",
    "replaced_token_detection_loss",
]
