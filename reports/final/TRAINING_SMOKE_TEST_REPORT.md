# Synthetic training smoke-test report

Date: 2026-11-16. Environment: Windows, Python 3.10.11, PyTorch 2.9.1+cpu.
All model inputs are temporary synthetic token shards. This report concerns
software execution only; no production tokenizer, corpus, model training run,
trained checkpoint, or model-quality result was produced.

## Executed evidence

The final full command `python -m pytest -q` passed **89 tests**, skipped **2**
platform-dependent tests, and emitted four environment/dependency warnings. It includes:

- `tests/test_training_data.py::test_model_specific_trainers_resume_and_evaluate_synthetic_runs`
  creates small actual ORION-100M RTD and ORION-300M causal models, runs one
  training step, writes a checkpoint, resumes each run through its public
  model-specific training function, reaches step 2, checks finite losses, and
  evaluates each resulting local checkpoint on a separate synthetic shard.
- `tests/test_training_data.py::test_interrupted_training_resumes_with_equivalent_parameters_and_rng`
  compares an uninterrupted run with a checkpointed and resumed run, including
  model parameters and Python/NumPy/PyTorch RNG continuation.
- `tests/test_training_data.py::test_safe_checkpoint_restores_parameters_optimizer_and_rng`
  verifies restored model parameters, optimizer configuration, and random
  streams.
- `tests/test_supervised_training.py` exercises small real multi-task,
  contrastive, course-matching, and assistant-only SFT optimizer/checkpoint
  paths.
- `tests/test_sdk_api.py::test_decoder_sdk_streams_real_cpu_generation_events`
  exercises CPU generation events from a locally constructed small random
  decoder; separate tests require a clear missing-weights error for absent
  trained releases.
- `tests/test_architectures.py` compares checkpointed and ordinary encoder and
  decoder outputs/gradients while preserving dropout RNG; it also verifies the
  decoder cache restriction during checkpointed training and cache use in eval.

The end-to-end model-specific test was also run directly under its final node
name and passed: **1 passed, 3 warnings in 21.81 seconds**. Its resume path uses
a one-step interruption under the same serialized two-step configuration,
then reloads the checkpoint through `train_rtd(..., resume=True)` and
`train_causal_lm(..., resume=True)`.

## Limits

The synthetic models use one layer and very small hidden sizes to keep CPU
checks bounded. Full target-size training, approved datasets, production
tokenizer training, learning-quality validation, CUDA/NCCL, and Linux
multi-process recovery remain unperformed. The two skipped tests are:

- `tests/test_distributed_training.py::test_ddp_gradient_sync_and_exact_rank_state_resume`
- `tests/test_training_data.py::test_multiworker_prefetch_checkpoint_tracks_consumed_samples`

They are real two-process tests configured in the Ubuntu `multiprocess-recovery`
CI job. The current managed Windows host denies Gloo device and multiprocessing
pipe creation; no remote CI result is available.

## Reproduction commands

```powershell
python -m pytest -q
python -m pytest -q tests/test_training_data.py::test_model_specific_trainers_resume_and_evaluate_synthetic_runs
python -m pytest -q tests/test_training_data.py::test_interrupted_training_resumes_with_equivalent_parameters_and_rng
```

The commands above were executed in this workspace; the final full suite is the
authoritative final verification entry in `FINAL_VERIFICATION_LOG.md`.
