# ORION tokenizer v1 token-ID registry

**Status: owner-approved for experimental research with the recorded digest; still provisional for production release.**
The supplied implementation plan has a damaged special-token table. This file
records the mapping currently implemented in
`packages/orion-tokenizer/src/orion_tokenizer/spec.py`; approval applies only to
the experimental research scope recorded below.

| ID range | Current assignment | Review status |
|---:|---|---|
| 0-7 | `<|pad|>`, `<|unk|>`, `<|cls|>`, `<|sep|>`, `<|mask|>`, `<|bos|>`, `<|eos|>`, `<|abstain|>` | Approved for experimental research |
| 8-15 | `<meta:filename>`, `<meta:title>`, `<meta:heading>`, `<meta:text>`, `<meta:ocr_low>`, `<meta:ext>`, `<meta:lang>`, `<field_sep>` | Approved for experimental research; source plan had a damaged table |
| 16-23 | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|context|>`, `<|/context|>`, `<|json|>`, `<|/json|>` | Approved for experimental research |
| 24-31 | Reserved IDs 24 through 31 | Approved reserved block |
| 32-47 | `<|doc_1|>` through `<|doc_16|>` | Approved for experimental research |
| 48-63 | `<|cite_1|>` through `<|cite_16|>` | Approved for experimental research |
| 64-255 | Reserved IDs 64 through 255 | Approved reserved block |
| 256-511 | Reversible byte alphabet; ID is 256 plus the byte value | Reviewed code-defined byte-to-Unicode bijection |
| 512-32767 | Learned BPE merge vocabulary | Assigned by the eventual approved corpus run, then frozen |

The special-token map SHA-256 currently computed by the trainer is
`733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`. The
digest covers the JSON object mapping decimal IDs 0-255 to token strings, with
sorted keys and UTF-8 encoding. It does not cover the byte alphabet or learned
merges. Recompute it after any source change and record the approved value in
`tokenizer_approval.md`. The current approval covers experimental research only.

The current decoder permits EOS, abstention, end-of-turn, and citation markers;
other special IDs are forbidden during ordinary generation. Literal control
spellings in untrusted input are escaped through the byte alphabet. Both model
packages validate tokenizer version, vocabulary size, and artifact hash when
loading a trained release.

## Approval scope and remaining gates

The owner approved the current special-token map digest for experimental
research on 2026-11-16. The final learned vocabulary (IDs 512-32767) and its
checksum do not exist until tokenizer training completes on a separately
authorized corpus. Broader production or commercial release requires its own
review.

The exact map digest is approved for experimental research in
[`tokenizer_approval.md`](tokenizer_approval.md). Production tokenizer training
still requires separately authorized corpus sources and explicit operator
confirmation. A passing synthetic tokenizer test does not grant data or release
approval.
