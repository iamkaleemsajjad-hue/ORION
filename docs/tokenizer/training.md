# Tokenizer training

The Tokenizers package includes a byte-level BPE trainer that reads explicit
local text or JSONL files, applies NFC normalization, assigns the reserved IDs
and byte alphabet, verifies the requested vocabulary size, writes a corpus
checksum manifest, and stores golden vectors. It performs no download or
implicit corpus search. New artifacts are staged and verified by the trainer
before they are atomically placed at the output path; a failed run leaves no
partial release there.

Before invoking the command, the project owner must complete
`docs/tokenizer/tokenizer_approval.md`, review the ID ranges and map digest in
`docs/tokenizer/final_token_id_registry.md`, and separately authorize the
local corpus. The CLI flag is an operator assertion; it does not authenticate
the owner's identity or prove that the approval record was completed.

The checked-in recipe deliberately remains pending until the owner has reviewed
the mapping. Record that decision in the approval file and set
`special_token_id_table_reviewed` to `true` in the recipe. Copy the approved
decision and digest into the machine release status as described in the approval
record. The CLI flag is a second operator assertion; none of these local values
authenticates identity or substitutes for the completed approval record.

Validate the recipe before training:

```powershell
orion-tokenizer validate-config --config .\configs\tokenizer\byte-bpe-v1.json
orion-tokenizer train --corpus .\approved-local-corpus.jsonl --output .\artifacts\tokenizer-v1 --config .\configs\tokenizer\byte-bpe-v1.json --project-root . --confirm-training --approve-token-map
orion-tokenizer verify --tokenizer-dir .\artifacts\tokenizer-v1
orion-tokenizer export --tokenizer-dir .\artifacts\tokenizer-v1 --output .\artifacts\tokenizer-v1-export
```

`validate-config` checks the recipe but does not imply approval. The production
`train` command requires both explicit confirmation switches and a reviewed
recipe, and must produce exactly 32,768 entries with stable reserved IDs. If an
authorized corpus does not produce the expected number of merge entries,
training fails instead of inventing tokens. `verify` checks manifest hashes,
the special-token map, byte and vocabulary ID ranges, and every golden encoding.
`export` copies only a verified release into a new local directory; it does not
publish or upload. No production tokenizer command was run and release status
remains pending.

The tests train only a tiny 512-entry tokenizer labeled
`synthetic-test-fixture` to exercise local write and failure cleanup. That
artifact is rejected by production verification and is not a v1 release.
