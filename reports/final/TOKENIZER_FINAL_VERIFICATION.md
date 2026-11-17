# Shared tokenizer final verification

## Software behavior

The single `orion-tokenizer` package owns the byte-level BPE interface, Unicode normalization and reversible byte fallback, token alignment, padding/truncation, training, serialization, release manifest, hashes, golden vectors, and CLI. Both model SDKs validate the shared artifact contract and vocabulary metadata. The final source suite passes (89 passed, 2 skipped, 4 warnings); tokenizer-specific tests are included in that run.

## Token-ID registry and artifact

- Provisional special-token registry digest: `733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`.
- `packages/orion-tokenizer/releases/v1.0/STATUS.json`: status `pending`, owner approval `approved` for experimental research, trained artifact absent, corpus not downloaded.
- IDs 0–255 remain provisional; 256–511 are byte alphabet IDs; 512–32767 are to be learned and frozen from the approved training corpus.
- The approved experimental token assignments and remaining production gates are in [final token ID registry](../../docs/tokenizer/final_token_id_registry.md) and [approval form](../../docs/tokenizer/tokenizer_approval.md).

## External compatibility gate

The declared Tokenizers range is `>=0.23.3,<0.24`, while the current runtime is 0.22.2. Official [PyPI 0.23.3 files](https://pypi.org/project/tokenizers/0.23.3/) confirm the target release exists, but this machine cannot retrieve it through its configured proxy. Tests on 0.22.2 do not establish compatibility with the declared minimum.

## Next authorized action

The current map digest is approved for experimental research; production training needs separately authorized corpus and package access; then run the existing tokenizer validate-config, train, verify, and export workflow. Production tokenizer training was not performed. Do not generate training shards or train model weights until the exact approved artifact is frozen and its hash is recorded for both models.
