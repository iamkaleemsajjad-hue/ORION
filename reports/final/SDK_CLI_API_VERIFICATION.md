# SDK, CLI and REST audit

## Interfaces exercised

The 100M SDK provides local loading, classify/predict/batch, extract, embeddings, info and health contracts. The 300M SDK provides local load, generation, grounded answer, summarize, question, flashcard and MCQ interfaces, info and health. Both validate inputs, expose typed outputs/errors and fail with a missing-weights error when no trained artifact exists. REST apps provide Pydantic request/response schemas, health, local-first behavior, consistent errors, batch/grounding contracts and generation streaming. Package CLIs and the shared `orion`/data/tokenizer tools exist. Synthetic API/SDK tests cover invalid input, missing weights, citations as unverified markers, no-context behavior, batches and package independence.

## Limitations

No trained-artifact inference is possible. Clean installed-package SDK/CLI/preflight behavior is not established because dependency resolution failed. No concurrent-load, cancellation, streaming-client disconnect, monitored network-denial, authentication deployment, request-timeout saturation, or API process shutdown stress test was run. An untrained random initialization is never surfaced as a successful learned prediction.

Evidence: [`sdk-api-report.md`](../implementation/sdk-api-report.md), `tests/test_sdk_api.py`, `tests/test_independence.py`, `docs/cli/reference.md`, `docs/api/reference.md`. FastAPI TestClient behavior in the current global environment emits compatibility deprecation warnings; test run still passes.
