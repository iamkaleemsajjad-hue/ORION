# Development setup

The workspace requires Python 3.10+. Install packages in dependency order:

```powershell
python -m pip install -e .\packages\orion-core
python -m pip install -e .\packages\orion-tokenizer
python -m pip install -e .\packages\orion-data
python -m pip install -e .\packages\orion-train
python -m pip install -e .\packages\orion-eval
python -m pip install -e .\packages\orion-100m
python -m pip install -e .\packages\orion-300m
python -m pip install -e .\packages\orion-madad-adapter
python -m pip install pytest ruff mypy
```

For an offline machine, first populate a trusted wheelhouse from a connected
build environment and install each dependency with `--no-index --find-links`.
The model packages themselves never require a network connection to load local
weights.

Run checks from the repository root:

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q packages tests tools
```

`uv` is optional. A generated `uv.lock` is not checked in because `uv` was not
available in the implementation environment; dependency ranges are declared in
each package's `pyproject.toml` and must be resolved and locked by the release
environment.
