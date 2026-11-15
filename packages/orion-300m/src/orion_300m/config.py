"""Validated ORION-300M architecture and generation configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Orion300MConfig(BaseModel):
    """Default decoder-only architecture from the supplied technical plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = "orion-300m"
    model_version: str = "0.1.0"
    tokenizer_version: str = "1.0"
    vocab_size: int = Field(default=32768, gt=0)
    num_layers: int = Field(default=28, gt=0)
    hidden_size: int = Field(default=960, gt=0)
    num_heads: int = Field(default=15, gt=0)
    num_kv_heads: int = Field(default=5, gt=0)
    intermediate_size: int = Field(default=2560, gt=0)
    pretraining_length: int = Field(default=2048, gt=0)
    context_length: int = Field(default=4096, gt=0)
    max_generation_tokens: int = Field(default=512, gt=0)
    rope_theta: float = Field(default=100000.0, gt=0)
    rms_norm_eps: float = Field(default=1e-5, gt=0)
    z_loss_coefficient: float = Field(default=1e-4, ge=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    pad_token_id: int = Field(default=0, ge=0)
    bos_token_id: int = Field(default=5, ge=0)
    eos_token_id: int = Field(default=6, ge=0)
    abstain_token_id: int = Field(default=7, ge=0)
    end_token_id: int = Field(default=19, ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Orion300MConfig:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.context_length < self.pretraining_length:
            raise ValueError("context_length cannot be shorter than pretraining_length")
        if self.max_generation_tokens > self.context_length:
            raise ValueError("max_generation_tokens cannot exceed context_length")
        if (
            max(
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.abstain_token_id,
                self.end_token_id,
            )
            >= self.vocab_size
        ):
            raise ValueError("special token IDs must be below vocab_size")
        return self

    @property
    def expected_parameter_count(self) -> int:
        """Compute unique body parameters, counting the tied vocabulary matrix once."""
        head_dim = self.hidden_size // self.num_heads
        attention = (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.num_kv_heads * head_dim * self.hidden_size
        )
        feed_forward = 3 * self.hidden_size * self.intermediate_size
        norms = (2 * self.hidden_size) * self.num_layers + self.hidden_size
        return (
            self.vocab_size * self.hidden_size
            + self.num_layers * (attention + feed_forward)
            + norms
        )
