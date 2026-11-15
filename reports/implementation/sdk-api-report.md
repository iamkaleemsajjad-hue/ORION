# SDK, CLI and REST verification report

Both model SDKs validate inputs, expose task methods and model/health metadata,
load only explicit local artifacts, and report a useful missing-weights error
instead of fabricating predictions. Model-specific CLIs and FastAPI apps
implement task requests, schemas, local default binding, consistent errors,
and generation/context contracts. The shared `orion preflight`, tokenizer
commands and data prepare/verify commands are also present.

The synthetic test suite covers SDK/API contracts, batches, invalid inputs,
missing weights, no-context behavior, citations as unverified markers, and
package independence. These checks do not verify successful inference with
trained weights. Clean installed-package entry points and installed-mode
preflight remain unverified because dependency resolution was blocked.

No concurrency/load or network-monitor security test was run. See
`docs/cli/reference.md`, package API modules, and
`reports/tests/validation-report.md` for exact command/test evidence.
