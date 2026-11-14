"""Validated ORION-100M architecture and task-head configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Orion100MConfig(BaseModel):
    """Architecture contract for the bidirectional 100M encoder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = "orion-100m"
    model_version: str = "0.1.0"
    tokenizer_version: str = "1.0"
    vocab_size: int = Field(default=32768, gt=0)
    num_layers: int = Field(default=16, gt=0)
    hidden_size: int = Field(default=640, gt=0)
    num_heads: int = Field(default=10, gt=0)
    intermediate_size: int = Field(default=1728, gt=0)
    max_sequence_length: int = Field(default=2048, gt=0)
    base_training_length: int = Field(default=1024, gt=0)
    local_window: int = Field(default=128, gt=0)
    global_attention_every: int = Field(default=4, gt=0)
    rope_theta_local: float = Field(default=10000.0, gt=0)
    rope_theta_global: float = Field(default=160000.0, gt=0)
    layer_norm_eps: float = Field(default=1e-5, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    pad_token_id: int = Field(default=0, ge=0)
    cls_token_id: int = Field(default=2, ge=0)
    sep_token_id: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Orion100MConfig:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.max_sequence_length < self.base_training_length:
            raise ValueError("max_sequence_length cannot be shorter than base_training_length")
        if max(self.pad_token_id, self.cls_token_id, self.sep_token_id) >= self.vocab_size:
            raise ValueError("special token IDs must be below vocab_size")
        return self
