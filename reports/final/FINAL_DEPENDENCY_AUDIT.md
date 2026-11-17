# Final dependency audit

## Current environment

- CPython 3.10.11, PyTorch 2.9.1+cpu, Tokenizers 0.22.2.
- All eight package projects declare Python `>=3.10`; `orion-tokenizer` requires Tokenizers `>=0.23.3,<0.24`.
- `uv` is not on PATH, and `uv.lock` is absent. No lock was fabricated.
- `PIP_NO_INDEX=1` explains why an earlier pip request did not query the index. In a diagnostic child process with package indexing enabled, the configured local proxy (`127.0.0.1:9`) refused connections. Certificate validation was not bypassed and no alternate index was used.
- The supported Tokenizers 0.23.3 release and Python 3.10 Windows x86-64 ABI3 wheel are listed by [official PyPI metadata](https://pypi.org/project/tokenizers/0.23.3/); local retrieval is blocked.

## Corrections and checks

- `orion-train` now declares its directly used Pydantic dependency.
- All eight packages now have package-specific Markdown full descriptions declared through `project.readme`; one regression test covers all eight. The [PyPA metadata specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/) defines this field as the package's full description and resolves string paths relative to the package project root.
- Built eight wheels and eight source distributions using `python -m build --no-isolation --outdir build/closure-distributions <package>`. Inspected wheel `METADATA` and sdist archives: Markdown descriptions are present, each sdist contains `README.md`, and no missing-README warning remained.
- `tests/test_package_metadata.py`: 1 passed. Full final suite: 89 passed, 2 skipped, 4 warnings.
- Root workspace lists eight members and the `dev` dependency group. The lock resolver has not run; optional groups, platform resolution, and transitive constraints therefore remain unverified.

## Not established

No dependency-resolved install, isolated ORION `pip check`, installed-mode entry-point test, or resolved dependency SBOM exists. The current global `pip check` fails on unrelated environment packages (including protobuf, httpx, urllib3, packaging, NumPy, and OpenTelemetry conflicts); it is not an ORION clean-environment result.

**Status:** package build metadata verified; reproducible dependency management blocked on approved package-index access and a real uv resolver. Install uv from [official uv instructions](https://docs.astral.sh/uv/getting-started/installation/), then generate and check the lock using [official project sync guidance](https://docs.astral.sh/uv/concepts/projects/sync/). Do not manually author `uv.lock`.
