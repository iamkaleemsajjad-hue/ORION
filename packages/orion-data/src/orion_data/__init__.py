"""Offline data provenance, quality, and packing utilities for ORION."""

from orion_data.manifest import DatasetManifest, DatasetSource, LicenseRecord
from orion_data.normalize import normalize_text

__all__ = ["DatasetManifest", "DatasetSource", "LicenseRecord", "normalize_text"]

__version__ = "0.1.0"
