# Training guide

No real ORION training on a production configuration or real dataset was run.
Tiny synthetic CPU tests exercised one optimizer step for each implemented
pretraining objective. No trained tokenizer or model weights are in the
repository. Prepare and review licensed data, tokenize it with the approved
frozen v1 tokenizer, write local uint16 shards, then validate the JSON configs
before intentionally invoking a training command.

```powershell
python tools/validate_config.py --kind 100m-training configs/pretraining/orion-100m-rtd.json
python tools/validate_config.py --kind 300m-training configs/pretraining/orion-300m-causal.json
python tools/validate_config.py --kind 100m-finetuning configs/finetuning/orion-100m-multitask.json
python tools/validate_config.py --kind 100m-contrastive configs/finetuning/orion-100m-contrastive.json
python tools/validate_config.py --kind 100m-course-matching configs/finetuning/orion-100m-course-matching.json
python tools/validate_config.py --kind 300m-sft configs/finetuning/orion-300m-sft.json
python tools/validate_config.py --kind 300m-architecture-pilot configs/evaluation/e-300-1-architecture-pilot.json
```

For a later run, include `--confirm-training` on the corresponding model CLI.
No training command acquires data. Pretraining consumes local token shards;
supervised fine-tuning consumes local tokenized JSONL. ORION-100M pretraining
uses an RTD generator/discriminator loss; ORION-300M pretraining uses causal
next-token loss with the configured z-loss. ORION-100M multi-task fine-tuning
reads validated local tokenized JSONL, and ORION-300M assistant-only SFT includes a message renderer,
assistant target masks, local JSONL loading, and the same resumable trainer.
Paired-text contrastive training supports the configured 640/256/128
Matryoshka dimensions. Course matching has a separate registry-prototype and
open-set runner. The shared trainer has an opt-in DDP strategy with deterministic
rank sharding, rank-zero checkpoints, and per-rank RNG/data-cursor state. Set
`trainer.distributed_strategy` to `ddp` and launch the model CLI with `torchrun`.
The process-group integration and exact-resume test are implemented but could
not execute in this Windows PyTorch build because Gloo cannot create a network
device. Run that test on Linux before treating DDP as verified. FSDP is not
implemented; do not set an FSDP strategy.

Contrastive JSONL contains one positive pair per row; in-batch non-matching
pairs act as negatives. Each side is a non-empty list of tokenizer IDs within
the configured sequence length:

```json
{"anchor_ids": [2, 301, 302, 3], "positive_ids": [2, 410, 411, 3]}
```

At least two records are required so the contrastive objective has in-batch
negatives. Rows are validated while building a byte-offset index, and the
training checkpoint binds to the input file checksum and the verified RTD
initial checkpoint.

Course-matching training reads a local registry JSON object whose courses have
unique IDs and tokenized profiles, plus tokenized JSONL documents. A null
`course_id` explicitly identifies an unknown-course example. For each batch,
known course profiles are paired with sampled registry negatives; unknown
documents are trained with an unscaled cosine rejection margin. Both known and
unknown examples must occur in the input shard:

```json
{"courses": [
  {"course_id": "MATH-101", "input_ids": [2, 301, 302, 3]},
  {"course_id": "PHYS-101", "input_ids": [2, 410, 411, 3]}
]}
```

```json
{"document_ids": [2, 501, 502, 3], "course_id": "MATH-101"}
{"document_ids": [2, 601, 602, 3], "course_id": null}
```

Use `orion100m train-course-matching` with a checksum-verified ORION-100M
multi-task checkpoint and its matching `--labels-json`. The complete task model
is restored so previously trained classification heads remain usable. Thresholds
still require calibration on the planned unseen-course split; a synthetic
training step does not establish a valid operating point.

For ORION-100M, `orion100m finetune` accepts a label-map JSON object, a verified
ORION-100M RTD checkpoint via `--init-checkpoint`, and a
tokenized JSONL shard. Every row in one shard must have the same target fields;
supported targets are `subject`, `document_type`, `topic`, `multilabel`,
`metadata`, and `confidence`. `input_ids` are integer token IDs and `metadata`
contains one BIO tag ID per input token. The example optimizer settings are
unverified defaults, not a recommendation based on a pilot.

For ORION-300M, use `write_sft_jsonl(conversations, tokenizer, path,
max_length=...)` to render local `system`/`user`/`assistant` message lists, then
pass a verified ORION-300M causal pretraining checkpoint through
`--init-checkpoint`. The
output contains `input_ids` and `assistant_mask`; only assistant content and
turn-ending markers contribute to loss. It refuses to truncate an example that
exceeds the requested sequence length.

`Trainer` supports AdamW, no-decay groups, gradient accumulation/clipping,
autocast precision, schedulers, finite-value checks, validation-loss calls, and
resumable checkpoints. Checkpoints use safetensors plus JSON; they do not use
pickle. DDP checkpoints add every rank's Python, NumPy, CPU/CUDA RNG and sampler
cursor to the verified JSON metadata while rank zero writes the shared
checkpoint. The exact-resume test is enabled on supported Linux runners but was
skipped in this Windows run after Gloo initialization failed with `unsupported
gloo device`; DDP is implemented but not locally verified. Multi-worker
data-loader recovery and FSDP recovery are not implemented.

The large default configurations require substantial accelerator memory and
compute. Only CPU smoke tests with small synthetic configurations were run; no
hardware throughput or memory estimates were verified here. Do not begin full
training until the plan's architecture, corpus licensing, tokenizer-ID review,
compute budget, and experimental gates have been approved.

The E-300-1 pilot compares decoder-only, prefix-LM, and tied-embedding
encoder-decoder candidates on matching tokenized source-target JSONL splits.
Each row contains `source_ids` and `target_ids`; the target must end with the
configured EOS ID. The example config uses a reduced pilot architecture and is
not the production model configuration. Validate the splits and budget before
running `orion300m architecture-pilot`; it requires `--confirm-training`, saves
independent resumable checkpoints per candidate, and reports measured
parameter counts, train/validation losses, data hashes, and elapsed time.
Only synthetic unit fixtures exercised the runner here; no real pilot results
exist.
