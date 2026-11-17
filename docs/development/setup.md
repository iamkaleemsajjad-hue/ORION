# Development setup

## Current installation status

The project requires Python 3.10 or newer. The workspace defines eight
installable packages and their dependency ranges, but this checkout has no
resolved `uv.lock`. The current environment has no `uv`; `PIP_NO_INDEX=1`
prevents pip from querying an index. A diagnostic child process that overrode
that setting reached the configured proxy, which refused its local connection.
PyPI lists a compatible Tokenizers 0.23.3 Windows/Python 3.10 wheel, but this
workspace cannot retrieve it. A clean dependency-resolving installation has
therefore not been verified. See the [Tokenizers 0.23.3 package record](https://pypi.org/project/tokenizers/0.23.3/).

Do not treat the earlier wheel/import check using `--no-deps` as a clean
installation. Reproducible one-command setup is a remaining gate. Once a
compatible resolver is available, the intended workspace setup is:

```powershell
uv sync --all-packages --group dev
```

Generate and review a real lockfile before using that command with
`--locked`. Do not hand-author a lockfile. For standalone package installation,
install only from a clean environment after dependencies have been resolved;
the eight package dependency boundaries are listed in
`reports/implementation/final_audit_inventory.md`.

## Local checks

Run from the repository root after dependencies are installed:

```powershell
python -m pytest -q
python -m mypy packages tests tools --show-error-codes
python -m ruff check .
python -m ruff format --check .
python -m compileall -q packages tools tests
```

The model and data CLI smoke checks run against installed package entry points.
Source-checkout tests use the package paths configured in `pyproject.toml`.
CUDA installation must follow the PyTorch wheel instructions for the target
driver/runtime; no CUDA environment was available in this audit.

Run `orion doctor` after installing the workspace. It checks local packages,
runtime versions, lockfile state, host resources, and backend availability
without network requests. Checks that require an actual network, distributed
runner, or target hardware remain explicitly untested. Run `orion preflight`
to validate package imports and checked-in configurations without downloading
data or starting a training job. Use `orion preflight --model 100m` or
`--model 300m` with local
`--dataset-dir` and `--tokenizer-dir` paths to check model-specific training
inputs. A pending owner-approved tokenizer release is reported as a blocker.

For offline machines, populate and verify a trusted wheelhouse on a connected
build environment. Model inference is designed to load local artifacts without
network access, but network-denial behavior was not monitored in this audit.
