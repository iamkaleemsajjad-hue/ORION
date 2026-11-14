# Software validation report

Validation was run on Windows with Python 3.10.11, CPU synthetic fixtures, and
the dependencies already available in the workspace. No dataset acquisition,
tokenizer training, production model training, GPU job, or release upload was
run. Tiny one-step objective tests are software tests and do not constitute
ORION training.

## Passing checks

| Check | Result |
|---|---|
| `python -m pytest -q` | 51 tests passed; one environment warning reports deprecated `pynvml`. |
| `ruff check .` | Passed. |
| `ruff format --check .` | Passed after formatting; all checked files were formatted. |
| `python tools/count_params.py` | Both parameter counts match their targets; tied decoder embeddings are verified. |
| `python tools/validate_config.py --kind 100m-training configs/pretraining/orion-100m-rtd.json` | Passed. |
| `python tools/validate_config.py --kind 300m-training configs/pretraining/orion-300m-causal.json` | Passed. |
| Seven pretraining, fine-tuning, SFT, and architecture-pilot config validators | Passed. |
| `python -m compileall -q packages tools tests` | Passed. |
| All eight package wheel builds and local-target import check | Passed with `pip wheel --no-deps --no-build-isolation` and a clean `--no-index --no-deps --target` install; dependencies came from the existing environment. |
| Model CLI help/info smoke checks | Passed from source and generated wheel entry points with `PYTHONPATH` set to the local install; `info` reports no trained weights and no imported pretrained weights. |

The suite includes forward/backward and parameter-count checks; tokenizer byte,
Unicode, Urdu, and alignment tests; tiny synthetic RTD, causal, multi-task,
assistant-only SFT, Matryoshka contrastive, registry-prototype/open-set, and
three-candidate architecture-pilot optimizer steps; prefix-LM mask and
encoder-decoder cross-attention tests; checkpoint integrity, RNG, run identity,
and interrupted-run equivalence checks; held-out-shard evaluation; CPU
generation/cache/streaming and ordered batch API; API body limits and error
sanitization; data guards; evaluation metrics; adapter contracts; and static
package-independence checks. The batch-sampler tests cover contrastive resume
at an epoch boundary. Checkpoint tests detect JSON-state tampering and reject
fine-tuning from a mismatched model/config checkpoint.

## Checks that did not pass or were not completed

- `python -m mypy packages tests tools` exited with 171 errors across 48 files.
  The environment has mypy 2.1.0, while `pyproject.toml` requires mypy `<2`, so
  the configured version was not available. The run also reported missing
  third-party stubs, including `tokenizers` and `pyarrow`, along with source
  typing errors. No clean strict type-check result is claimed; strict settings
  were not weakened to hide the findings.
- `pip check` reports conflicts among unrelated packages in the pre-existing
  global Python environment (including protobuf, decorator, httpx, urllib3,
  NumPy/OpenCV, packaging, cachetools, matplotlib/scikit-learn, and tornado).
  No global packages were changed to resolve them. A clean isolated install was
  not performed.
- `uv.lock` generation and an isolated dependency-resolving install remain
  incomplete: `uv` was unavailable and the configured package index did not
  provide it. Wheel builds and no-dependency target imports did pass.
- GitHub Actions workflows were added but not executed in this workspace.
- A full dependency vulnerability scan, external security audit, formal privacy
  review, and legal review were not performed.
- No release payload was built or uploaded because no trained model artifact is
  present.

The command log in this report describes this implementation run; rerun all
commands under the final, intended package-install environment before a release
or production training phase.
