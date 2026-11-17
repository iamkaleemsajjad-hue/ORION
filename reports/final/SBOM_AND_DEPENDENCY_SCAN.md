# SBOM and dependency scan status

The checked-in `orion-source-declarations.cdx.json` is explicitly a source-declaration inventory. CycloneDX schema validation passed previously, but it lists unresolved direct constraints and is **not** a transitive resolved SBOM.

A lock-derived SBOM cannot be generated honestly yet: `uv` and `uv.lock` are absent, package resolution is blocked by the `PIP_NO_INDEX=1` policy and refused proxy, and the installed Tokenizers version does not satisfy package metadata. No package-level dependency vulnerability conclusion can be inferred from the conflicted global environment.

The earlier `pip-audit --local --format cyclonedx-json ...` attempt failed while reaching the vulnerability service. It produced no clean result. The final local Ruff S scan passed; this does not replace CVE scanning. Once a real lock is created on a network-enabled host, install that exact environment, generate a CycloneDX/SPDX SBOM from resolved packages, validate its schema, run pip-audit/OSV tooling, and record tool/database timestamps and findings.
