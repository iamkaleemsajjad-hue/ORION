"""Regression checks for independently distributable package metadata."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[1] / "packages"


def test_each_distribution_declares_an_included_full_description() -> None:
    package_projects = sorted(PACKAGES.glob("*/pyproject.toml"))

    assert len(package_projects) == 8
    for project_file in package_projects:
        project_text = project_file.read_text(encoding="utf-8")
        project_section = re.search(
            r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)",
            project_text,
        )
        assert project_section is not None, project_file
        assert re.search(r'^readme\s*=\s*"README\.md"\s*$', project_section.group(1), re.M), (
            project_file
        )

        readme = project_file.parent / "README.md"
        assert readme.is_file(), readme
        assert len(readme.read_text(encoding="utf-8").strip()) > 100, readme
