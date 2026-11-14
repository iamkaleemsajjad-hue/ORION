# Security and local inference

- Model weights and training tensor state use safetensors. Hashes are checked
  before state loading; Python pickle deserialization is not used.
- Tokenizer, model, and data assets are loaded from explicit local paths. SDK
  loading performs no network calls and has no pretrained fallback.
- Release verification requires trained status, source/evaluation provenance,
  tokenizer identity, and SHA-256 checks before a release can be accepted.
- User-provided strings are encoded as ordinary bytes. Only trusted SDK prompt
  template code can insert control token IDs.
- FastAPI request bodies have bounded ASGI readers and schema-level text limits.
  Set `ORION_API_TOKEN` for optional local API-key authentication.
- REST binds to loopback by default. Use a separately reviewed reverse proxy if
  exposing an endpoint beyond the local machine.
- Dataset acquisition is explicit, requires pinned revisions, and is not run
  automatically. PII masking and license policy are pipeline controls, not a
  substitute for legal and privacy review.
- Hub upload is an explicit tool command, private by default, and refuses
  artifacts that fail local release verification.

SHA-256 verifies byte identity, not publisher identity. Signed release
provenance and a human review are required before distributing future weights.
