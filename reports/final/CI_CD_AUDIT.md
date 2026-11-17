# CI/CD audit

## Current workflows

Three workflows exist: `ci.yml` covers Python 3.10/3.11/3.12 on Ubuntu and Windows, resolves workspace packages with uv, compiles, runs strict mypy, lints, formats and runs pytest; `security.yml` runs Ruff security rules, audits dependencies and rejects pickle-loading patterns; `release.yml` is manually dispatched and builds wheels/uploads a short-lived workflow artifact. Permissions are read-only. Their branch triggers use `master`. The newly configured mypy and Ruff security commands were executed locally and pass; workflows themselves were inspected, not executed on GitHub.

## Gaps and validation

No `uv.lock` means current `uv sync` resolves compatible latest versions rather than a committed fixed graph. CI still has no package clean-install/installed CLI test. The security workflow was not run and the local vulnerability DB was unreachable. No resolved SBOM artifact, dedicated secret scan, coverage gate, CUDA runner or Linux multiprocessing result was verified. Release automation builds and stores temporary CI artifacts; it does not publish a package/model release.

Evidence: `.github/workflows/{ci,security,release}.yml`; local compilation, Ruff and test results are in [verification log](FINAL_VERIFICATION_LOG.md). No GitHub action was triggered by this audit.
