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

- `pip-audit`, Bandit, Semgrep, Trivy and secret scanners were not run.
- No SBOM was generated; GitHub workflows were inspected but not executed.
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
