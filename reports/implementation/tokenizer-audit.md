# Tokenizer compatibility and approval audit

## Software verification

The shared tokenizer uses byte-level BPE, NFC normalization, a reserved-token
map, a byte alphabet, local corpus input, explicit vocabulary size, corpus
hashes and golden vectors. The CLI now exposes recipe validation, training,
release verification and local export. Production artifacts are staged and
verified before atomic placement; failed writes do not leave a partial release.
Production verification checks manifest/file hashes, token map digest, ID
ranges, tokenizer config and golden encodings. Tiny synthetic training tests
are marked as fixtures and rejected by production verify/export.

A small training reproduction found `pre_tokenizers.Regex` was not a valid
namespace under installed Tokenizers 0.22.2. The code now uses top-level
`tokenizers.Regex` with `pre_tokenizers.Split`, matching the reviewed upstream
0.23.3 API. The test run exercises this path with the installed runtime.

## Owner gate

The special-token digest is
`733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`.
The source plan's original special-token table was damaged, so the checked-in
registry remains provisional. `STATUS.json`, the recipe and approval record
remain pending. CLI confirmation flags are local operator assertions and do
not authenticate the owner.

**Final v1 mapping freeze is blocked pending owner approval.** No production
corpus was downloaded and no final tokenizer was trained/exported.

## Next authorized workflow

Review `docs/tokenizer/final_token_id_registry.md` and
`docs/tokenizer/tokenizer_approval.md`, resolve the token map, record the same
decision and digest in the recipe/status, then follow the exact commands in
`docs/tokenizer/training.md`. Use only an explicitly approved local corpus.
Before release, rerun `orion-tokenizer verify` and export to a new local path.
