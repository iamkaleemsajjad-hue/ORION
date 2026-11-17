# Security, privacy and release audit

## Reviewed software safeguards

The source audit covered safe local artifact serialization and hashes,
checkpoint path handling, tokenizer and prepared-data manifest path checks,
local-only data preparation, explicit acquisition/upload confirmations,
input schemas/limits, missing-weight behavior and credential-by-environment
patterns. Regression tests exercise artifact/path integrity, prepared-manifest
traversal, input validation and local release checks. No automatic data
acquisition or upload is triggered by import or ordinary inference.

## Unverified / external review required

- The current `pip-audit` run failed before completing because the configured
  proxy refused the vulnerability database request. Bandit, Semgrep, Trivy and
  secret scanners were not run.
- No resolved, transitive release SBOM exists. A preliminary CycloneDX 1.6
  source-declaration inventory is schema-validated in
  `reports/final/orion-source-declarations.cdx.json`; it does not contain a
  locked dependency graph or exact installed ORION distributions.
- GitHub workflows were inspected but not executed.
- Offline behavior was not observed with a network-denial monitor.
- No formal threat model, external security review, PII false-negative study,
  privacy review or qualified legal review was completed.
- License and attribution validation checks metadata presence; it cannot prove
  source authorization or legal sufficiency.
- No model artifact has been prepared for public release. Resumable training
  state must stay private unless explicitly reviewed.

This report is not a security certification or legal opinion. Before public
distribution, run the repository's security workflow and dependency/SBOM
scanners in a resolved environment, review findings, test network isolation,
and obtain the required human privacy/legal approvals.


## Final local checks (2026-11-15)

The initial Ruff security-rule pass produced seven findings. Source review
found one redundant assert, which was removed; six context-specific findings
were annotated narrowly with reasons (`HF_TOKEN` is an environment-variable
name, `<|unk|>` is a tokenizer control string, and seeded `random.Random` is
used for reproducible augmentation, bootstrap resampling, and sampler order).
The final `ruff check --select S` passes; both CI workflows now contain the
strict mypy/security-lint gates. `pip-audit` did not complete because the
configured proxy refused the service connection. The source-declaration
CycloneDX 1.6 file passed its local schema validator but omits
resolved/transitive versions. These checks do not replace a network-enabled
dependency scan and external review.
