# orion-data

Offline-first dataset manifest validation, provenance, normalization, filtering, split validation, token-shard preparation, and deterministic data readers for ORION.

Dataset acquisition is explicit. Importing this package and running its ordinary validators do not download datasets. Example records and fixtures are synthetic and must not be treated as approved training data.

Use the `orion-data` CLI to inspect and validate local manifests. Production data preparation requires authorized sources, accurate licenses, and a compatible approved tokenizer artifact.
