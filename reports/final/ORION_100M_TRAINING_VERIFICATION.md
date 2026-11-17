# ORION-100M training software verification

## Architecture

The default encoder has 16 layers, width 640, 10 heads, pre-LayerNorm, RoPE, GeGLU, and alternating local/global attention. `tools/count_params.py` instantiated the model on the PyTorch meta device and measured **100,291,840 body parameters**, matching the plan.

## Training and recovery path

The actual `orion_100m.training` RTD path constructs the configured generator/discriminator, optimizer, scheduler, stateful data loader, and checkpoint manager. `test_model_specific_trainers_resume_and_evaluate_synthetic_runs` executes a small one-layer encoder/generator fixture with whole-word alignment, gradient checkpointing, training, resume to step 2, and held-out synthetic evaluation; losses are required finite. Architecture and objective tests are in `tests/test_architectures.py`, `tests/test_supervised_training.py`, and `tests/test_training_data.py`.

The final full suite reports **89 passed, 2 skipped, 4 warnings**. This proves software paths on small random-initialized CPU models. It does not establish document classification quality, confidence calibration, unknown-course detection, or production throughput.

## Remaining gates

Approved corpus/labels and licenses, approved shared tokenizer ID registry and trained artifact, clean dependency-resolved installation, target hardware preflight/memory measurement, production training, and held-out task evaluation. Do not claim learned document intelligence before those steps.
