"""Validated dataset and license provenance records."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LicenseRecord(BaseModel):
    """License attribution and restriction flags for one source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spdx_id: str = Field(min_length=1)
    attribution: str = ""
    non_commercial: bool = False
    share_alike: bool = False
    verified: bool = False
    source_url: str | None = None


class DatasetSource(BaseModel):
    """One pinned input source and its provenance metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    revision: str
    license: LicenseRecord
    local_path: str | None = None
    sha256: str | None = None
    synthetic: bool = False

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return value

    @field_validator("revision")
    @classmethod
    def require_pinned_revision(cls, value: str) -> str:
        if value in {"", "main", "latest", "HEAD"}:
            raise ValueError(
                "dataset revisions must be immutable commit hashes or local fixture IDs"
            )
        return value


class DatasetManifest(BaseModel):
    """Versioned manifest for a reproducible offline dataset build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    dataset_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    sources: tuple[DatasetSource, ...]
    normalization: str = "NFC"
    pipeline_version: str = "0.1.0"
    seed: int = 1729
    filters: tuple[str, ...] = ()
    synthetic_fixture: bool = False

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        manifest = cls.model_validate(data)
        if manifest.synthetic_fixture and any(not item.synthetic for item in manifest.sources):
            raise ValueError("A synthetic fixture manifest cannot contain a non-synthetic source")
        return manifest

    def validate_license_policy(self, mode: str) -> None:
        """Enforce the documented research or commercial-clean source policy."""
        if mode not in {"research", "commercial-clean"}:
            raise ValueError("mode must be 'research' or 'commercial-clean'")
        if self.synthetic_fixture:
            return
        for source in self.sources:
            license_record = source.license
            if not license_record.verified or license_record.spdx_id.lower() in {
                "unknown",
                "noassertion",
                "license not specified",
            }:
                raise ValueError(f"Unverified license for source {source.source_id}")
            if mode == "commercial-clean":
                allowed = {"odc-by-1.0", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}
                if license_record.non_commercial or license_record.spdx_id.lower() not in allowed:
                    raise ValueError(
                        f"Source {source.source_id} is not permitted in commercial-clean mode"
                    )
