"""Stable errors shared by the model SDKs, training tools, and APIs."""

from __future__ import annotations


class OrionError(Exception):
    """Base exception carrying a stable machine-readable error code."""

    code = "ORION-E0000"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(OrionError, ValueError):
    code = "ORION-E0002"


class ModelWeightsNotFound(OrionError, FileNotFoundError):
    code = "ORION-E2001"


class ArtifactIntegrityError(OrionError):
    code = "ORION-E2002"


class TokenizerMismatchError(OrionError):
    code = "ORION-E2003"


class DeviceUnavailableError(OrionError):
    code = "ORION-E3002"


class InputLimitError(OrionError, ValueError):
    code = "ORION-E1001"


class EmptyInputError(OrionError, ValueError):
    code = "ORION-E1003"


class NoContextError(OrionError, ValueError):
    code = "ORION-E4001"
