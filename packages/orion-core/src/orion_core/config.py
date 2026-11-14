"""Typed configuration loading with explicit validation and safe YAML parsing."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion_core.errors import ConfigurationError

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class RuntimeConfig(BaseModel):
    """Cross-model artifact and runtime contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    vocab_size: int = Field(default=32768, gt=0)
    max_sequence_length: int = Field(gt=0)


def load_yaml_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    """Load a YAML mapping using ``safe_load`` and validate it against a schema."""
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read configuration {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration {source} must contain a YAML mapping")
    try:
        return schema.model_validate(value)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Configuration {source} does not match {schema.__name__}",
            details={"validation_errors": exc.errors(include_input=False)},
        ) from exc
