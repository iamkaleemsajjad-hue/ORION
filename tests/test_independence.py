from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_standalone_source_has_no_madad_or_pretrained_model_imports() -> None:
    forbidden = {"madad", "transformers", "sentence_transformers"}
    for package in ROOT.glob("packages/orion-*/src"):
        if package.parent.name == "orion-madad-adapter":
            continue
        for source in package.rglob("*.py"):
            assert _imports(source).isdisjoint(forbidden), source


def test_model_packages_do_not_depend_on_each_other_or_madad() -> None:
    for package_name, forbidden in (
        ("orion-100m", {"orion-300m", "orion-madad-adapter"}),
        ("orion-300m", {"orion-100m", "orion-madad-adapter"}),
    ):
        root = ROOT / "packages" / package_name
        dependencies = (root / "pyproject.toml").read_text(encoding="utf-8")
        assert all(name not in dependencies for name in forbidden)
        for source in (root / "src").rglob("*.py"):
            assert _imports(source).isdisjoint({name.replace("-", "_") for name in forbidden})


def test_training_distribution_declares_its_direct_configuration_dependency() -> None:
    metadata = (ROOT / "packages/orion-train/pyproject.toml").read_text(encoding="utf-8")
    runtime_project = metadata.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert '"pydantic>=2.7,<3"' in runtime_project
