# CLI reference

Run commands after installing the packages. Model commands do not download
weights. `info` reports architecture status without loading weights; inference
commands fail clearly when trained artifacts are missing. API servers bind to
`127.0.0.1` by default.

## Environment preflight

```powershell
orion doctor
orion doctor --format text
orion preflight
orion preflight --mode installed
orion preflight --model 100m --dataset-dir .\datasets\prepared\course-mix --tokenizer-dir .\artifacts\orion-tokenizer-v1 --device cuda --precision bf16
```

`orion doctor` reports local package, Python, PyTorch, Tokenizers, lockfile,
package-index policy, host-memory, disk, CUDA, Gloo, and NCCL checks as JSON.
Use `--format text` for a readable rendering. It does not make network requests,
start training, or reveal proxy values. Missing lock resolution, unavailable
training hardware, or an unexecuted distributed check remain blocked or
explicitly untested; backend availability is not evidence of communication.

`orion preflight` checks the source workspace, installed runtime versions, and
all checked-in run configurations. `--mode installed` checks the eight
installed distributions instead. Adding `--model` also verifies local
prepared data and tokenizer inputs; the command reports owner-approval,
artifact, precision, and minimum-token blockers explicitly. Exit status is 0
when ready, 1 when checks are blocked, and 2 when the preflight itself could
not run. Preflight is read-only and never starts training or downloads data.

## Local dataset preparation

```powershell
orion-data validate-manifest .\datasets\manifests\approved-local.json --mode commercial-clean
orion-data prepare .\datasets\manifests\approved-local.json --tokenizer .\artifacts\orion-tokenizer-v1 --output .\datasets\prepared\course-mix --config .\configs\data\orion-data-preparation.json --mode commercial-clean
orion-data verify-prepared .\datasets\prepared\course-mix
```

Preparation is local-only and refuses synthetic manifests. The manifest must
pin local JSONL file checksums and verified licenses; the tokenizer must be a
complete trained ORION release. No data acquisition or tokenizer/model training
is performed by `prepare` or `verify-prepared`.

## ORION-100M

```powershell
orion100m info
orion100m health
orion100m classify --text "..."
orion100m predict --input-json document.json --registry-json courses.json
orion100m predict-batch documents.json --batch-size 16
orion100m extract --text "..."
orion100m embed --text "..." --dimension 128
orion100m serve --host 127.0.0.1 --port 8761
orion100m train --tokens local.tokens --word-ids word_ids.i32 --output checkpoints/orion-100m --config-json configs/pretraining/orion-100m-rtd.json --confirm-training
orion100m finetune --data-jsonl train.jsonl --labels-json labels.json --init-checkpoint checkpoints/orion-100m/checkpoints/step-000000000100 --output checkpoints/orion-100m-finetune --config-json configs/finetuning/orion-100m-multitask.json --confirm-training
orion100m train-contrastive --data-jsonl pairs.jsonl --init-checkpoint checkpoints/orion-100m/checkpoints/step-000000000100 --output checkpoints/orion-100m-contrastive --config-json configs/finetuning/orion-100m-contrastive.json --confirm-training
orion100m train-course-matching --registry-json courses.json --data-jsonl course-documents.jsonl --labels-json labels.json --init-checkpoint checkpoints/orion-100m-finetune/checkpoints/step-000000000100 --output checkpoints/orion-100m-course-matching --config-json configs/finetuning/orion-100m-course-matching.json --confirm-training
orion100m evaluate --tokens validation.tokens --word-ids validation-word-ids.i32 --checkpoint-dir checkpoints/orion-100m --config-json configs/pretraining/orion-100m-rtd.json
orion100m evaluate-finetune --data-jsonl validation.jsonl --labels-json labels.json --checkpoint-dir checkpoints/orion-100m-finetune --config-json configs/finetuning/orion-100m-multitask.json
orion100m evaluate-contrastive --data-jsonl validation-pairs.jsonl --checkpoint-dir checkpoints/orion-100m-contrastive --config-json configs/finetuning/orion-100m-contrastive.json
orion100m evaluate-course-matching --registry-json courses.json --data-jsonl unseen-course-documents.jsonl --labels-json labels.json --checkpoint-dir checkpoints/orion-100m-course-matching --config-json configs/finetuning/orion-100m-course-matching.json
```

The training command requires the explicit confirmation flag, a pretokenized
local uint16 shard, and a valid JSON run config. The examples are planning
configurations; this repository implementation phase did not execute them.
The fine-tuning data is a local JSONL shard of token IDs and task targets; see
the training guide for its schema and label-map requirements.

## ORION-300M

```powershell
orion300m info
orion300m health
orion300m generate --prompt "..."
orion300m answer --prompt "..." --documents-json context.json
orion300m summarize --documents-json context.json
orion300m questions --documents-json context.json --count 5
orion300m flashcards --prompt "topic" --documents-json context.json --count 8
orion300m mcq --prompt "topic" --documents-json context.json
orion300m serve --host 127.0.0.1 --port 8762
orion300m train --objective causal --tokens local.tokens --output checkpoints/orion-300m --config-json configs/pretraining/orion-300m-causal.json --confirm-training
orion300m train --objective sft --sft-jsonl local-sft.jsonl --init-checkpoint checkpoints/orion-300m/checkpoints/step-000000000100 --output checkpoints/orion-300m-sft --config-json configs/finetuning/orion-300m-sft.json --confirm-training
orion300m architecture-pilot --training-jsonl pilot-train.jsonl --validation-jsonl pilot-validation.jsonl --output experiments/tutor/e-300-1 --config-json configs/evaluation/e-300-1-architecture-pilot.json --confirm-training
orion300m evaluate --tokens validation.tokens --checkpoint-dir checkpoints/orion-300m --config-json configs/pretraining/orion-300m-causal.json
orion300m evaluate --objective sft --sft-jsonl validation-sft.jsonl --checkpoint-dir checkpoints/orion-300m-sft --config-json configs/finetuning/orion-300m-sft.json
```

For SFT, prepare tokenized JSONL with the Python `write_sft_jsonl` helper. It
renders local chat records and masks loss to assistant content and end markers.

`orion-tokenizer validate-config`, `train`, `verify`, and `export` are also
available. Production tokenizer training requires the explicit confirmation
flags and a reviewed recipe. Verification and export require a real trained
release; the current pending recipe is not one. `orion-data acquire` requires
explicit opt-in, license-policy selection, and pinned source manifests. No
production tokenizer training, acquisition, or export was run.
