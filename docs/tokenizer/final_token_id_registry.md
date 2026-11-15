# ORION tokenizer v1 token-ID registry

**Status: provisional; owner approval is required before v1 training or release.**
The supplied implementation plan has a damaged special-token table. This file
records the mapping currently implemented in
`packages/orion-tokenizer/src/orion_tokenizer/spec.py`; it does not claim that
the owner has approved the mapping.

| ID range | Current assignment | Review status |
|---:|---|---|
| 0-7 | `<|pad|>`, `<|unk|>`, `<|cls|>`, `<|sep|>`, `<|mask|>`, `<|bos|>`, `<|eos|>`, `<|abstain|>` | Provisional names and order |
| 8-15 | `<meta:filename>`, `<meta:title>`, `<meta:heading>`, `<meta:text>`, `<meta:ocr_low>`, `<meta:ext>`, `<meta:lang>`, `<field_sep>` | Confirm these match the plan |
| 16-23 | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|context|>`, `<|/context|>`, `<|json|>`, `<|/json|>` | Provisional names and order |
| 24-31 | Reserved IDs 24 through 31 | Confirm these remain reserved |
| 32-47 | `<|doc_1|>` through `<|doc_16|>` | Provisional document controls |
| 48-63 | `<|cite_1|>` through `<|cite_16|>` | Provisional citation controls |
| 64-255 | Reserved IDs 64 through 255 | Confirm the reserved block |
| 256-511 | Reversible byte alphabet; ID is 256 plus the byte value | Review the code-defined byte-to-Unicode bijection |
| 512-32767 | Learned BPE merge vocabulary | Assigned by the eventual approved corpus run, then frozen |

The special-token map SHA-256 currently computed by the trainer is
`733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`. The
digest covers the JSON object mapping decimal IDs 0-255 to token strings, with
sorted keys and UTF-8 encoding. It does not cover the byte alphabet or learned
merges. Recompute it after any source change and record the approved value in
`tokenizer_approval.md`.

The current decoder permits EOS, abstention, end-of-turn, and citation markers;
other special IDs are forbidden during ordinary generation. Literal control
spellings in untrusted input are escaped through the byte alphabet. Both model
packages validate tokenizer version, vocabulary size, and artifact hash when
loading a trained release.

## Owner review required

Before freezing v1, the owner must review and approve:

1. The exact names and order for IDs 0-7 and 16-31 whose plan text was lost.
2. The metadata controls at IDs 8-15 and document/citation controls at IDs 32-63.
3. Which controls may be emitted by ORION-300M and which remain input-only.
4. The byte mapping, normalization, and special-token escaping behavior.
5. The final 32,768-entry vocabulary and its checksum after training on an
   separately authorized corpus.

Tokenizer training must remain pending until the owner approval record is
completed. A passing synthetic tokenizer test does not approve these IDs.
