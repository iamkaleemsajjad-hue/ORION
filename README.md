# ORION AI

ORION AI is an independent model family for educational document understanding and context-grounded educational text generation. This repository contains software and architecture only: **no trained ORION weights and no frozen production tokenizer are present**. SDK model loading therefore fails clearly until valid artifacts are supplied. No dataset download, production tokenizer training, or model training on real data was run. Tiny synthetic CPU tests did execute one-step objective updates on randomly initialized small models.

The monorepo is organized as separately installable packages. ORION-100M is a bidirectional document encoder; ORION-300M is a causal text generator. Both use the same versioned tokenizer contract, but own their model weights and training objectives. MADAD-specific behavior is isolated in `orion-madad-adapter`.

## Current project status

- Model definitions, data/training software, package interfaces, and synthetic verification fixtures are present. See the implementation status for features that remain partial or require future training.
- The supplied plan's special-token ID table has missing token text in several cells. The ID ranges are preserved, and the currently inferred entries are documented as provisional in `docs/architecture/decisions.md`.
- Target parameter counts and quality figures are design targets, never achieved results.
- Dataset manifests in this repository are synthetic examples only. Dataset acquisition commands are opt-in and have not been invoked.

See the [implementation status](docs/implementation-status.md), [requirements traceability](reports/implementation/requirements-traceability.md), [engineering handover](reports/implementation/handover.md), and [technical decision record](docs/architecture/decisions.md).

## Repository map

| Path | Responsibility |
|---|---|
| `packages/orion-core` | Config contracts, safe serialization, hashes, shared Transformer blocks and runtime utilities |
| `packages/orion-tokenizer` | Versioned byte-level BPE interface and explicit tokenizer-training command |
| `packages/orion-data` | Offline data validation, normalization, provenance, filtering, deduplication, packing and manifests |
| `packages/orion-train` | Reproducible training loop, optimization, checkpoints and recovery |
| `packages/orion-eval` | Metrics, calibration, bootstrap intervals and sealed-set safeguards |
| `packages/orion-100m` | ORION-100M encoder, task heads, objectives, SDK, CLI and API |
| `packages/orion-300m` | ORION-300M decoder, cache, generation, objectives, SDK, CLI and API |
| `packages/orion-madad-adapter` | Optional host-side contracts; no standalone model package imports it |
| `configs` | Validated tokenizer, model, training, evaluation and deployment configuration examples |
| `datasets` | Manifests and cards only; no downloaded corpora |
| `docs` | Architecture, data, training, interfaces, security and governance documentation |
| `tools` | Parameter, config, checkpoint, release and export validation utilities |
| `tests` | Synthetic unit, architecture, tokenizer, training, inference, security and boundary tests |

## Development setup

Python 3.10 or newer is required. Install each package from the repository root in dependency order, or use the workspace resolver documented in `docs/development/setup.md` when `uv` is available. The model packages do not fetch weights or access the network at runtime.

```powershell
python -m pip install -e .\packages\orion-core
python -m pip install -e .\packages\orion-tokenizer
python -m pip install -e .\packages\orion-data
python -m pip install -e .\packages\orion-train
python -m pip install -e .\packages\orion-eval
python -m pip install -e .\packages\orion-100m
python -m pip install -e .\packages\orion-300m
python -m pip install -e .\packages\orion-madad-adapter
python -m pytest
```

This installs software dependencies only. It does not train anything or download model artifacts or datasets.

## Verify architecture counts without allocating full weights

```powershell
python tools/count_params.py
```

For future data preparation and training commands, see [training guide](docs/training/overview.md) and [dataset pipeline guide](docs/data/pipeline.md). Those commands are documented but have not been run on real data.

## Security and provenance

Weights load only from local safetensors files after SHA-256 verification. Training recovery state is stored as JSON plus safetensors tensors; pickle loading is not used. Dataset provenance records revision, license, filters and checksums. See [security](docs/security/security.md) and [governance](docs/governance/data-governance.md).

## License

The source code is licensed under MIT. Model weights and datasets, once they exist, require their own provenance and license records; this source license does not grant rights to any future training data or model artifacts.
