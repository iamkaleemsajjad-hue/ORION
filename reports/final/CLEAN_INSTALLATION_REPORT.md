# Clean installation report

## Result: NOT VERIFIED — external package access blocked

The requested clean, dependency-resolving install of each of the eight distributions could not be performed. The environment has no `uv`, no `uv.lock`, sets `PIP_NO_INDEX=1`, and its configured proxy refused the diagnostic request to the official package index. The installed Tokenizers 0.22.2 also fails the package's `>=0.23.3,<0.24` requirement.

## Evidence completed

- Eight wheels and eight source distributions built locally with `python -m build --no-isolation`.
- All eight wheel metadata files contain `Description-Content-Type: text/markdown`; all eight sdists contain `README.md`.
- `tests/test_package_metadata.py`: 1 passed.
- Source test suite: 89 passed, 2 platform-dependent tests skipped, 4 warnings.
- Strict mypy is clean on 124 files; Ruff and Ruff security rules pass. Formatting passes on 262 files.

These checks verify build artifacts and source behavior only. They do not prove that dependency resolution succeeds, that an isolated installation works, or that installed CLI commands resolve outside the repository.

## Blocked checks

`pip check` was executed only in the shared global environment and reports unrelated package conflicts. No ORION-only environment was created or resolved. `wsl --list --quiet` returned `E_ACCESSDENIED`; Docker could not connect to its Linux engine, so Linux package and multiprocess tests could not run here.

## Required clean-host procedure

On a network-enabled, clean host after installing an approved uv version:

1. Run `uv lock` and inspect the generated lock; then `uv lock --check`.
2. Run `uv sync --all-packages --group dev`.
3. Run `uv run python -m pip check` and the complete test/static suite.
4. Build each package wheel and sdist with the package build workflow.
5. In separate clean virtual environments, install each wheel with dependency resolution enabled; run its import, SDK, and entry-point smoke checks from outside the repository root.
6. Repeat for the combined model environment and approved optional extras.

Commands above are configured/documented but remain unexecuted until the external resolver is available. See [dependency audit](FINAL_DEPENDENCY_AUDIT.md) and [setup guide](../../docs/development/setup.md).
