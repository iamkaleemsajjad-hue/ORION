# Security policy

## Supported versions

Security fixes are made on the latest released software version. A future trained model artifact will have a separate model version and provenance record.

## Reporting a vulnerability

Please do not include credentials, private course documents, or student records in an issue. Report suspected vulnerabilities privately to the project maintainers once a project contact is configured. Until then, use the repository host's private vulnerability reporting feature.

## Security design

- Model weights use safetensors only and are SHA-256 verified before loading.
- Training state is represented by JSON metadata and tensor-only safetensors files; the loader does not deserialize pickle.
- Runtime SDKs perform no network access by default and do not log document text.
- REST defaults to loopback, has explicit request-size limits, and accepts text payloads rather than local file paths.
- Dataset manifests pin revisions, licenses, and hashes. Unknown and non-commercial licenses are excluded from the commercial-clean policy.

See [security documentation](docs/security/security.md) for threat boundaries and artifact handling.
