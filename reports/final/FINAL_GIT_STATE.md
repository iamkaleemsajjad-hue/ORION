# Final Git state

Captured 2026-11-16 after local implementation and verification, before the requested source-only GitHub publication.

| Field | Value |
|---|---|
| Branch | `master` |
| Initial/current HEAD | `7d3186c7a0378ef0878a49c524068338af945547` (unchanged; no commit created) |
| `origin/master` local ref | `7d3186c7a0378ef0878a49c524068338af945547` |
| Tracked files | 225 |
| Modified tracked files | 58 |
| Staged files | 0 |
| Untracked files | 80 |
| Porcelain status lines (all untracked paths) | 138 |
| Commit/push/tag/release/publication | None performed |

All changes remain in the local workspace. No commit, push, tag, release, package/model publication, reset, clean, stash, or revert was performed. The starting HEAD and current HEAD are the same.

`core.autocrlf=true`; Git emits LF-to-CRLF notices for some paths. `git diff --check` exits 0 with no whitespace errors. Raw and whitespace-insensitive path sets match.

## Tracked diff summary

```text
.github/workflows/ci.yml                           |  20 ++
 .github/workflows/security.yml                     |   2 +
 docs/cli/reference.md                              |  11 +-
 docs/data/pipeline.md                              |   7 +
 docs/development/setup.md                          |  19 +-
 docs/implementation-status.md                      |  39 ++--
 docs/research/research-ledger.md                   |  13 +-
 docs/training/checkpoints.md                       |  23 ++-
 docs/training/overview.md                          |  42 +++-
 packages/orion-100m/pyproject.toml                 |   1 +
 packages/orion-100m/src/orion_100m/contrastive.py  |   5 +-
 .../orion-100m/src/orion_100m/course_matching.py   |   5 +-
 packages/orion-100m/src/orion_100m/finetuning.py   |   5 +-
 .../orion-100m/src/orion_100m/model/encoder.py     |  39 +++-
 .../orion-100m/src/orion_100m/model/pretraining.py |  21 +-
 packages/orion-100m/src/orion_100m/training.py     |  18 +-
 packages/orion-300m/pyproject.toml                 |   1 +
 .../src/orion_300m/architecture_pilot.py           |   9 +-
 .../orion-300m/src/orion_300m/model/decoder.py     |  60 ++++--
 .../src/orion_300m/model/encoder_decoder.py        |  82 ++++++--
 packages/orion-300m/src/orion_300m/sft.py          |   5 +-
 packages/orion-300m/src/orion_300m/training.py     |  18 +-
 packages/orion-core/pyproject.toml                 |   1 +
 .../orion-core/src/orion_core/blocks/attention.py  |   1 -
 packages/orion-core/src/orion_core/cli.py          |  15 +-
 packages/orion-core/src/orion_core/preflight.py    | 108 +++++++++-
 packages/orion-data/pyproject.toml                 |   1 +
 packages/orion-data/src/orion_data/acquire.py      |   3 +-
 packages/orion-data/src/orion_data/pipeline.py     |   3 +-
 packages/orion-eval/pyproject.toml                 |   1 +
 packages/orion-eval/src/orion_eval/metrics.py      |   3 +-
 packages/orion-madad-adapter/pyproject.toml        |   1 +
 packages/orion-tokenizer/pyproject.toml            |   1 +
 .../src/orion_tokenizer/training.py                |   3 +-
 packages/orion-train/pyproject.toml                |   9 +-
 packages/orion-train/src/orion_train/config.py     |  25 +++
 packages/orion-train/src/orion_train/sampler.py    |  67 ++++++-
 packages/orion-train/src/orion_train/trainer.py    | 169 ++++++++++++++--
 reports/architecture/README.md                     |   3 +-
 reports/implementation/changed-files.md            |  12 +-
 reports/implementation/decision-records.md         |   7 +-
 .../implementation/dependency-install-report.md    |  26 +++
 .../implementation/distributed-training-report.md  |  33 ++--
 reports/implementation/final_audit_inventory.md    |  24 +++
 .../implementation/final_engineering_handover.md   |  18 +-
 reports/implementation/handover.md                 |   7 +-
 reports/implementation/issue-investigation.md      |  35 +++-
 reports/implementation/no_dummy_code_audit.md      |   9 +
 .../implementation/requirements-traceability.md    |  16 +-
 reports/implementation/security-audit.md           |  25 ++-
 .../implementation/training-readiness-checklist.md |   2 +-
 reports/implementation/training-recovery-report.md |   5 +-
 reports/tests/validation-report.md                 |  45 ++++-
 tests/test_architectures.py                        |  75 ++++++-
 tests/test_distributed_training.py                 |   4 +-
 tests/test_independence.py                         |   6 +
 tests/test_preflight.py                            |  57 +++++-
 tests/test_training_data.py                        | 217 ++++++++++++++++++++-
 58 files changed, 1300 insertions(+), 182 deletions(-)
```

The complete list of 58 modified tracked paths and 80 untracked paths is in [CHANGED_FILE_MANIFEST.md](CHANGED_FILE_MANIFEST.md).
