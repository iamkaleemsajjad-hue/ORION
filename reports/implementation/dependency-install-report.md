# Dependency and installation audit

## Current evidence

- Python 3.10.11; PyTorch 2.9.1+cpu; Tokenizers 0.22.2; mypy 2.1.0.
- Eight separately installable packages declare bounded dependencies and
  internal package version constraints.
- The workspace has no `uv.lock`; `uv` is unavailable.
- Attempts to install `uv` and Tokenizers 0.23.3 failed because the configured
  index returned no distributions.
- A clean, dependency-resolving installation of all eight packages was not
  completed. Existing source/wheel checks that used local paths or `--no-deps`
  do not establish clean-install success.
- The source preflight checks package imports and configuration contracts, and
  correctly exits blocked on the installed Tokenizers version mismatch.
- `pip check` in the global environment reports conflicts among unrelated
  packages. It is not an isolated ORION dependency result.

## Status

**BLOCKED — EXTERNAL PACKAGE INDEX/TOOL AVAILABILITY.** Do not hand-write a
lockfile or claim a reproducible install from wheel creation alone.

## Next verification

On a network-enabled environment with the approved package index:

1. Install the pinned `uv` tool.
2. Resolve the workspace and all requested extras; generate the real lockfile.
3. Review the dependency graph, platform markers and Python constraints.
4. Create a fresh environment from that lock and install all eight packages
   with dependency resolution enabled.
5. Run `pip check`, `orion preflight --mode installed`, package import tests,
   CLI entry-point smoke checks and optional-dependency failure tests.
6. Repeat on the documented CPU platform and separately document CUDA wheels.

No dependency scanner result is implied by this report.
