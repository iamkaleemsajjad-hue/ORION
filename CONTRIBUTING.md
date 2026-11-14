# Contributing

Contributions should preserve the project's independence, reproducibility, and honest artifact status.

## Development

1. Use Python 3.10 or newer and install the packages from `docs/development/setup.md`.
2. Keep model-specific code inside its own package. The standalone model packages must not import MADAD or each other.
3. Add tests for behavior that changes. Tests must use synthetic fixtures and exercise real core logic.
4. Run `python -m pytest`, `ruff check packages tests tools`, and `ruff format --check packages tests tools` before submitting.
5. Do not train the production tokenizer or either model as part of routine development. Do not download training data unless the work item explicitly authorizes that acquisition.

## Architecture and data

Changes to frozen model dimensions, parameter counts, special-token IDs, tokenizer normalization, schemas, or dataset licensing require a decision record and corresponding traceability updates. Dataset fixtures must be labeled synthetic. Never commit checkpoints, credentials, private course materials, or unreviewed dataset shards.

## Pull requests

Describe the behavior, validation performed, and any unverified hardware or training assumptions. A green tiny synthetic test is evidence of software behavior only; it is not evidence of model quality.
