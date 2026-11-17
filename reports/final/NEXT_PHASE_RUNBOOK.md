# Owner runbook for the authorized data and training phase

This is the exact software workflow documented by the current CLI. It was not executed in this audit. Run it only after package access is restored, token IDs/data are approved, the target device passes preflight, and training is explicitly authorized. Paths below use the documented PowerShell conventions.

## 1. Resolve and install the workspace

In a network-enabled clean checkout/environment, first install an approved pinned `uv` version. Then resolve a real lock and install the workspace:

```powershell
uv lock
uv sync --all-packages --group dev
uv run --with pip python -m pip check
uv run orion preflight --mode installed
```

Do not manually write `uv.lock`. Confirm all eight distributions and their entry points in this isolated environment before proceeding. CUDA-enabled PyTorch must be installed according to the target system’s supported official wheel/index and then revalidated.

## 2. Approve and train shared tokenizer v1

The current map digest is approved for experimental research. Preserve it and separately authorize the local corpus, its license and provenance, before training the tokenizer. Then:

```powershell
uv run orion-tokenizer validate-config --config .\configs\tokenizer\byte-bpe-v1.json
uv run orion-tokenizer train --corpus .\approved-local-corpus.jsonl --output .\artifacts\tokenizer-v1 --config .\configs\tokenizer\byte-bpe-v1.json --project-root . --confirm-training --approve-token-map
uv run orion-tokenizer verify --tokenizer-dir .\artifacts\tokenizer-v1
uv run orion-tokenizer export --tokenizer-dir .\artifacts\tokenizer-v1 --output .\artifacts\tokenizer-v1-export
```

Do not use fixture corpora for a release. Verify and retain the corpus checksum and tokenizer manifest.

## 3. Validate and prepare approved local data

Create an approved immutable manifest whose local JSONL files and license/provenance records have been reviewed. Then:

```powershell
uv run orion-data validate-manifest .\datasets\manifests\approved-local.json --mode commercial-clean
uv run orion-data prepare .\datasets\manifests\approved-local.json --tokenizer .\artifacts\tokenizer-v1-export --output .\datasets\prepared\course-mix --config .\configs\data\orion-data-preparation.json --mode commercial-clean
uv run orion-data verify-prepared .\datasets\prepared\course-mix
```

Preparation rejects synthetic manifests. Check provenance, split, PII, duplicate and quality reports before using shards.

## 4. Preflight the selected device and data

```powershell
uv run orion preflight --model 100m --dataset-dir .\datasets\prepared\course-mix --tokenizer-dir .\artifacts\tokenizer-v1-export --device cuda --precision bf16
uv run orion preflight --model 300m --dataset-dir .\datasets\prepared\course-mix --tokenizer-dir .\artifacts\tokenizer-v1-export --device cuda --precision bf16
```

Proceed only when the relevant preflight exits ready and target memory/compute has been reviewed. A CPU smoke test is not full-scale hardware approval.

## 5. ORION-100M train, resume and evaluate

```powershell
uv run orion100m train --tokens .\datasets\prepared\course-mix\train\tokens\tokens.u16 --word-ids .\datasets\prepared\course-mix\train\tokens\word_ids.i32 --output .\checkpoints\orion-100m --config-json .\configs\pretraining\orion-100m-rtd.json --confirm-training
```

To resume the same run after interruption, use the same output/config/data and append `--resume`:

```powershell
uv run orion100m train --tokens .\datasets\prepared\course-mix\train\tokens\tokens.u16 --word-ids .\datasets\prepared\course-mix\train\tokens\word_ids.i32 --output .\checkpoints\orion-100m --config-json .\configs\pretraining\orion-100m-rtd.json --confirm-training --resume
uv run python tools/verify_checkpoint.py .\checkpoints\orion-100m
uv run orion100m evaluate --tokens .\datasets\prepared\course-mix\validation\tokens\tokens.u16 --word-ids .\datasets\prepared\course-mix\validation\tokens\word_ids.i32 --checkpoint-dir .\checkpoints\orion-100m --config-json .\configs\pretraining\orion-100m-rtd.json
```

## 6. Run the ORION-300M architecture pilot before selection

Prepare approved `source_ids`/`target_ids` JSONL with disjoint, identical splits for each candidate; the general document-preparation command does not invent these task labels. Validate the pilot config, then explicitly run it:

```powershell
uv run python tools/validate_config.py --kind 300m-architecture-pilot .\configs\evaluation\e-300-1-architecture-pilot.json
uv run orion300m architecture-pilot --training-jsonl .\experiments\tutor\e-300-1\train.jsonl --validation-jsonl .\experiments\tutor\e-300-1\validation.jsonl --output .\experiments\tutor\e-300-1\runs --config-json .\configs\evaluation\e-300-1-architecture-pilot.json --confirm-training
```

Resume an interrupted pilot with the same command plus `--resume`. Do not choose the final architecture from synthetic pilot outputs.

## 7. ORION-300M train, resume and evaluate

```powershell
uv run orion300m train --objective causal --tokens .\datasets\prepared\course-mix\train\tokens\tokens.u16 --output .\checkpoints\orion-300m --config-json .\configs\pretraining\orion-300m-causal.json --confirm-training
uv run orion300m train --objective causal --tokens .\datasets\prepared\course-mix\train\tokens\tokens.u16 --output .\checkpoints\orion-300m --config-json .\configs\pretraining\orion-300m-causal.json --confirm-training --resume
uv run python tools/verify_checkpoint.py .\checkpoints\orion-300m
uv run orion300m evaluate --tokens .\datasets\prepared\course-mix\validation\tokens\tokens.u16 --checkpoint-dir .\checkpoints\orion-300m --config-json .\configs\pretraining\orion-300m-causal.json
```

## 8. Verify an explicitly staged release candidate

Only after real training, held-out evaluation and release review:

```powershell
uv run python tools/verify_release.py --model orion-100m .\artifacts\orion-100m-release
uv run python tools/verify_release.py --model orion-300m .\artifacts\orion-300m-release
```

Verification is local. No command here publishes or uploads artifacts. The full API/CLI reference is [`docs/cli/reference.md`](../../docs/cli/reference.md); checkpoint guarantees and limits are in [`docs/training/checkpoints.md`](../../docs/training/checkpoints.md).
