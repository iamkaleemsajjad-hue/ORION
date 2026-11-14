"""Compatibility names for the canonical RTD pretraining implementation."""

from orion_100m.model.pretraining import (
    GeneratorBlock,
    Orion100MRTDPretrainer,
    RTDDiscriminatorHead,
    RTDOutput,
    SmallMLMGenerator,
    whole_word_mask,
)

Orion100MForRTD = Orion100MRTDPretrainer
RTDGenerator = SmallMLMGenerator
RTDStepOutput = RTDOutput

__all__ = [
    "GeneratorBlock",
    "Orion100MForRTD",
    "Orion100MRTDPretrainer",
    "RTDDiscriminatorHead",
    "RTDGenerator",
    "RTDOutput",
    "RTDStepOutput",
    "SmallMLMGenerator",
    "whole_word_mask",
]
