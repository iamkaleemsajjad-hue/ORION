# Shared tokenizer audit

## Implemented

One `orion-tokenizer` package supplies byte-level BPE, NFC normalization, reversible byte fallback, batch encode/decode, padding/truncation, token alignment, local-only streaming corpus input, training, serialization, golden vectors, manifests, hashing, staged writes, release verification, and CLI commands. Both model packages consume the shared artifact contract and check version/vocabulary/hash metadata. Tiny synthetic training/release tests exercise behavior; synthetic fixtures are not a final vocabulary.

## Blocking compatibility decision

The provisional special-token map digest is `733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`. The supplied plan’s token table lost entries, so IDs 0–255 and generation controls are explicitly provisional. The current registry and owner approval record are in [`final_token_id_registry.md`](../../docs/tokenizer/final_token_id_registry.md) and [`tokenizer_approval.md`](../../docs/tokenizer/tokenizer_approval.md). **The map is approved for experimental research; production training requires separately authorized corpus sources.** No production tokenizer artifact exists.

The package requires Tokenizers `>=0.23.3,<0.24`; the available runtime is 0.22.2. The existing synthetic tokenizer tests pass on 0.22.2, but the supported version is not installed or verified. No corpus was downloaded and no final tokenizer training was run.

## Next authorized steps

Preserve the approved exact token map and digest; resolve a package index that provides the pinned Tokenizers range; run the documented validate-config, local training, verify, and export commands on an approved corpus. Preserve the resulting tokenizer manifest/hash and use the same artifact for both models. See [`tokenizer-audit.md`](../implementation/tokenizer-audit.md) and [`training.md`](../../docs/tokenizer/training.md).
