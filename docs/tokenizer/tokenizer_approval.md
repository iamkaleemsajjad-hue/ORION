# Tokenizer v1 owner approval record

**Current state: PENDING OWNER REVIEW.** No production tokenizer may be trained
or released using the provisional token map until the owner completes this
record. The implementation team has not supplied owner approval.

## Mapping under review

- Registry: [final token-ID registry](final_token_id_registry.md)
- Tokenizer version: `1.0`
- Current special-token map SHA-256:
  `733da01709c59745290542da2bed85df615873777d35167c61c35bb01f544190`
- Current machine status: `packages/orion-tokenizer/releases/v1.0/STATUS.json`
- Current trained tokenizer artifact: none
- Corpus approval: separate decision; this record does not authorize data use

## Owner decision

Fill these fields only after reviewing every owner-review item in the registry.

| Field | Value |
|---|---|
| Decision | PENDING |
| Owner name or authorized role | PENDING |
| Decision date (ISO 8601) | PENDING |
| Approved special-token map SHA-256 | PENDING |
| Approved byte mapping and normalization | PENDING |
| Approved generation-emittable controls | PENDING |
| Notes or required changes | PENDING |

Approval applies only to the exact digest recorded above. If any token spelling,
ID, normalization, or escaping behavior changes, update the registry, compute a
new digest, and obtain approval again. A corpus license/source approval and an
explicit tokenizer-training authorization are separate requirements.

After completing this record, copy the approved decision and the same digest
into `packages/orion-tokenizer/releases/v1.0/STATUS.json` (`owner_approval` is
`approved`; `status` remains `pending` until a tokenizer is actually trained)
and set `special_token_id_table_reviewed` to `true` in
`configs/tokenizer/byte-bpe-v1.json`. The training CLI checks those records and
the code-defined digest before it starts. These local fields are auditable
assertions; they do not authenticate who edited them.

The CLI `--approve-token-map` switch is an explicit operator assertion. It does
not authenticate the operator or replace this completed owner record. Do not
pass it while this file says PENDING.
