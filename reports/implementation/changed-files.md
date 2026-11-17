# Changed-file manifest

Snapshot date: 2026-11-14. Baseline was clean at the beginning of the audit.
All entries below are local workspace changes; no commit, push, tag, PR, release, remote update, upload or publication was performed.

Modified tracked files: 78

- MODIFIED: `configs/tokenizer/byte-bpe-v1.json`
- MODIFIED: `docs/cli/reference.md`
- MODIFIED: `docs/data/pipeline.md`
- MODIFIED: `docs/development/setup.md`
- MODIFIED: `docs/implementation-status.md`
- MODIFIED: `docs/research/research-ledger.md`
- MODIFIED: `docs/tokenizer/training.md`
- MODIFIED: `docs/training/checkpoints.md`
- MODIFIED: `docs/training/overview.md`
- MODIFIED: `packages/orion-100m/src/orion_100m/api.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/cli.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/contrastive.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/course_matching.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/finetuning.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/model/encoder.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/model/heads.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/model/losses.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/model/pretraining.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/model/tasks.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/sdk.py`
- MODIFIED: `packages/orion-100m/src/orion_100m/training.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/api.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/architecture_pilot.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/cli.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/config.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/model/__init__.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/model/decoder.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/model/encoder_decoder.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/sdk.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/sft.py`
- MODIFIED: `packages/orion-300m/src/orion_300m/training.py`
- MODIFIED: `packages/orion-core/pyproject.toml`
- MODIFIED: `packages/orion-core/src/orion_core/blocks/feedforward.py`
- MODIFIED: `packages/orion-core/src/orion_core/blocks/rotary.py`
- MODIFIED: `packages/orion-core/src/orion_core/runtime.py`
- MODIFIED: `packages/orion-data/pyproject.toml`
- MODIFIED: `packages/orion-data/src/orion_data/cli.py`
- MODIFIED: `packages/orion-data/src/orion_data/manifest.py`
- MODIFIED: `packages/orion-data/src/orion_data/normalize.py`
- MODIFIED: `packages/orion-data/src/orion_data/packing.py`
- MODIFIED: `packages/orion-data/src/orion_data/pipeline.py`
- MODIFIED: `packages/orion-data/src/orion_data/shards.py`
- MODIFIED: `packages/orion-data/src/orion_data/tokenize.py`
- MODIFIED: `packages/orion-eval/src/orion_eval/calibration.py`
- MODIFIED: `packages/orion-madad-adapter/src/orion_madad_adapter/citations.py`
- MODIFIED: `packages/orion-madad-adapter/src/orion_madad_adapter/organizer.py`
- MODIFIED: `packages/orion-madad-adapter/src/orion_madad_adapter/sdk_bridge.py`
- MODIFIED: `packages/orion-tokenizer/pyproject.toml`
- MODIFIED: `packages/orion-tokenizer/releases/v1.0/STATUS.json`
- MODIFIED: `packages/orion-tokenizer/src/orion_tokenizer/cli.py`
- MODIFIED: `packages/orion-tokenizer/src/orion_tokenizer/spec.py`
- MODIFIED: `packages/orion-tokenizer/src/orion_tokenizer/tokenizer.py`
- MODIFIED: `packages/orion-tokenizer/src/orion_tokenizer/training.py`
- MODIFIED: `packages/orion-train/src/orion_train/__init__.py`
- MODIFIED: `packages/orion-train/src/orion_train/checkpoint.py`
- MODIFIED: `packages/orion-train/src/orion_train/cli.py`
- MODIFIED: `packages/orion-train/src/orion_train/config.py`
- MODIFIED: `packages/orion-train/src/orion_train/sampler.py`
- MODIFIED: `packages/orion-train/src/orion_train/schedulers.py`
- MODIFIED: `packages/orion-train/src/orion_train/trainer.py`
- MODIFIED: `pyproject.toml`
- MODIFIED: `reports/implementation/decision-records.md`
- MODIFIED: `reports/implementation/handover.md`
- MODIFIED: `reports/implementation/repository-audit.md`
- MODIFIED: `reports/implementation/requirements-traceability.md`
- MODIFIED: `reports/implementation/research-ledger.md`
- MODIFIED: `reports/tests/validation-report.md`
- MODIFIED: `tests/test_architecture_pilot.py`
- MODIFIED: `tests/test_architectures.py`
- MODIFIED: `tests/test_evaluation.py`
- MODIFIED: `tests/test_madad_adapter.py`
- MODIFIED: `tests/test_sdk_api.py`
- MODIFIED: `tests/test_supervised_training.py`
- MODIFIED: `tests/test_tokenizer.py`
- MODIFIED: `tests/test_training_data.py`
- MODIFIED: `tools/export_onnx.py`
- MODIFIED: `tools/inspect_model.py`
- MODIFIED: `tools/validate_config.py`

New untracked files before this manifest: 29

- NEW: `configs/data/orion-data-preparation.json`
- NEW: `docs/tokenizer/final_token_id_registry.md`
- NEW: `docs/tokenizer/tokenizer_approval.md`
- NEW: `packages/orion-core/src/orion_core/cli.py`
- NEW: `packages/orion-core/src/orion_core/preflight.py`
- NEW: `packages/orion-data/src/orion_data/prepare.py`
- NEW: `packages/orion-tokenizer/src/orion_tokenizer/approval.py`
- NEW: `packages/orion-tokenizer/src/orion_tokenizer/release.py`
- NEW: `packages/orion-train/src/orion_train/distributed.py`
- NEW: `reports/implementation/data-pipeline-audit.md`
- NEW: `reports/implementation/dependency-install-report.md`
- NEW: `reports/implementation/distributed-training-report.md`
- NEW: `reports/implementation/final_audit_inventory.md`
- NEW: `reports/implementation/final_engineering_handover.md`
- NEW: `reports/implementation/issue-investigation.md`
- NEW: `reports/implementation/no_dummy_code_audit.md`
- NEW: `reports/implementation/orion-100m-audit.md`
- NEW: `reports/implementation/orion-300m-audit.md`
- NEW: `reports/implementation/performance-report.md`
- NEW: `reports/implementation/sdk-api-report.md`
- NEW: `reports/implementation/security-audit.md`
- NEW: `reports/implementation/tokenizer-audit.md`
- NEW: `reports/implementation/training-readiness-checklist.md`
- NEW: `reports/implementation/training-recovery-report.md`
- NEW: `reports/implementation/type-checking-report.md`
- NEW: `tests/test_data_prepare.py`
- NEW: `tests/test_distributed_training.py`
- NEW: `tests/test_preflight.py`
- NEW: `tests/test_tokenizer_release.py`
- NEW: ``reports/implementation/changed-files.md`` (this manifest)


## Current audit delta (2026-11-15)

The prior delta above is already included in audited baseline commit
`7d3186c7a0378ef0878a49c524068338af945547`. This audit made documentation and
report changes only; no package/model implementation source, configuration,
tests, Git metadata, or user data were modified. The exact current working-tree
list is generated in `reports/final/CHANGED_FILE_MANIFEST.md`; no commit or push
was performed.
