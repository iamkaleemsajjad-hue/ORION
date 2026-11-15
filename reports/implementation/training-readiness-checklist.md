# Training-readiness gate checklist

| Gate | Status | Evidence / blocker |
|---|---|---|
| A. Architecture | IMPLEMENTED AND VERIFIED (synthetic/software scope) | Exact counts 100,291,840 and 306,763,200; CPU architecture tests pass. Learned quality is not part of this gate. |
| B. Tokenizer | IMPLEMENTED BUT NOT FULLY VERIFIED | Tiny synthetic training/release path passes; final ID map needs owner approval and Tokenizers 0.23.3 installation. |
| C. Training | IMPLEMENTED BUT NOT FULLY VERIFIED | Tiny single-process recovery tests pass; DDP integration skipped; FSDP absent; prefetch recovery unverified. |
| D. Data | IMPLEMENTED BUT NOT FULLY VERIFIED | Synthetic local preparation tests pass; no real approved sources or corpus-scale validation. |
| E. SDK/API | IMPLEMENTED BUT NOT FULLY VERIFIED | Synthetic contracts pass; no trained artifacts and no clean installed entry-point test. |
| F. Packaging | BLOCKED BY EXTERNAL RESOURCE | No clean dependency-resolving all-package install; no `uv.lock`; required Tokenizers version unavailable. |
| G. Type safety | IMPLEMENTED AND VERIFIED | Strict mypy passes on 121 source files in the current environment. |
| H. Code quality | IMPLEMENTED AND VERIFIED | Ruff lint/format and compileall pass; see validation report. |
| I. Security | IMPLEMENTED BUT NOT FULLY VERIFIED | Local safeguards reviewed; scanners, SBOM, workflow run, network-denial and external reviews outstanding. |
| J. Documentation | IMPLEMENTED BUT NOT FULLY VERIFIED | Commands and APIs documented; clean installed command execution and future production-data paths remain unverified. |
| K. No dummy code | IMPLEMENTED AND VERIFIED (bounded scan) | Marker scan and synthetic-only use documented; not a substitute for full semantic review. |
| L. Training readiness | BLOCKED | Owner tokenizer approval, resolved clean installation, approved data, Linux DDP verification if used, and hardware preflight remain. |

## Authorized next-phase sequence

1. Resolve package-index access, install the required development/runtime
   dependencies, generate a genuine `uv.lock`, and pass isolated install plus
   preflight.
2. Obtain owner approval for `docs/tokenizer/final_token_id_registry.md`; set
   the approval record, recipe and release status consistently.
3. Run tokenizer `validate-config`, then explicitly authorized local v1
   training, `verify`, and `export` as documented in
   `docs/tokenizer/training.md`.
4. Prepare a rights-reviewed local manifest/data source, then invoke
   `orion-data validate-manifest`, `orion-data prepare`, and
   `orion-data verify-prepared` as documented in `docs/data/pipeline.md`.
5. Run model-specific `orion preflight --model ...` before training. Use the
   exact `orion100m` / `orion300m` train and evaluate command forms in
   `docs/cli/reference.md`; retain explicit confirmation flags and resumable
   checkpoints.
6. Run E-300-1 only on approved data/compute; make the decoder architecture
   decision from the documented experiment output.

Dataset preparation, tokenizer training, model training, and publication were
not executed in this engineering audit. External package availability,
owner approval, data authorization, and target compute are prerequisites, not
software steps to bypass.
