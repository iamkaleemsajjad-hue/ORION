# Security overview

See [artifact and inference controls](artifacts-and-inference.md) for hash
verification, safetensors-only loading, API size limits, local-only inference,
and credential handling. Review [data governance](../governance/data-governance.md)
before staging any training corpora.

The current security checks do not constitute a formal audit. External package
vulnerability results depend on the resolver and audit date; the current host
environment has unrelated dependency conflicts recorded in the final report.
