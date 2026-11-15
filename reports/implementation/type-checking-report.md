# Type-checking audit

## Reproduced finding

The prior handover reported 171 mypy diagnostics across 48 files. The
diagnostics were reproduced against Python 3.10.11 and the installed mypy
2.1.0. The claim that mypy 2.1.0 violated the workspace version range was
incorrect: the repository allowed `>=1.11,<3`.

## Correction and verification

Actual first-party annotation defects were corrected across model, core, data,
tokenizer, training, API, adapter and tooling packages. Strict mode remains
enabled. The only missing-import fallback is narrow and limited to Tokenizers,
whose installed 0.22.2 extension did not expose usable type stubs in this
environment. Mypy is pinned at 2.1.0.

Latest command:

```powershell
python -m mypy packages tests tools --show-error-codes
```

Result: **Success: no issues found in 121 source files.** Ruff and compileall
also pass; see `../tests/validation-report.md`.

## Remaining limitation

The package's required Tokenizers version (`>=0.23.3,<0.24`) could not be
installed from the configured package index. Therefore the result is verified
in the current source environment, not a clean install of the declared runtime
dependency set. Re-run mypy after a resolved isolated installation.
