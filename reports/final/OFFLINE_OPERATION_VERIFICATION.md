# Offline readiness audit

## Implemented properties

Model and tokenizer loaders use explicit local artifact paths; the packages do not silently substitute or fetch pretrained weights. Data acquisition and Hugging Face artifact operations are separate, opt-in commands. Ordinary import, source tests, tokenization, and local prepared-data workflows contain no required model download step. SDK contracts fail clearly when local trained weights are absent.

## Not verified

No monitored network-denial test ran, and therefore full offline behavior is **implemented by design but not independently verified**. The `pip-audit` attempt confirms this environment cannot reach its configured package index; that does not prove runtime has no network side effects. Offline installation additionally requires wheelhouses/dependencies prepared beforehand. No trained local artifacts exist, so real inference cannot be tested offline.

As a partial check, `tests/test_tokenizer.py`, `tests/test_sdk_api.py`, and
`tests/test_independence.py` passed **18 tests** while `HTTP_PROXY`,
`HTTPS_PROXY`, and `ALL_PROXY` pointed at a closed local port and `NO_PROXY` was
empty. These tests cover local tokenizer, SDK contracts, generated CPU events,
and absent-weight errors. Proxy settings are not equivalent to a monitored
firewall-denied environment and do not establish offline behavior for all
runtime paths.

Next: use a firewall-denied/network-monitor runner after clean installation; exercise imports, tokenizer/model loading, SDK, CLI and API against local fixtures/artifacts, fail on DNS/socket attempts, and archive the trace. See [`offline.md`](../../docs/deployment/offline.md).
