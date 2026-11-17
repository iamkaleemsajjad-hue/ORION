# orion-tokenizer

The canonical versioned tokenizer interface and explicit byte-level BPE training, export, and verification tools shared by ORION-100M and ORION-300M.

The repository currently contains the tokenizer specification and training software, but no final production-trained vocabulary. Token IDs and the v1 registry remain subject to owner approval. Do not train or publish the production tokenizer until that approval and the corpus authorization are in place.

Install `orion-core` as declared by package metadata. The `orion-tokenizer` command exposes tokenizer configuration and artifact operations; see the ORION tokenizer guide for the validated workflow.
