# Final requirements traceability

The machine-readable registry is [REQUIREMENTS_TRACEABILITY.json](REQUIREMENTS_TRACEABILITY.json). It maps PLAN-01 through PLAN-36 and the separate distributed-training requirement PLAN-25-DIST (37 plan rows), plus 13 hardening gates, to implementation evidence, tests, commands, known deficiencies, dependencies, and one of the requested final statuses. The current implementation plan is the supplied `ORION AI — Technical Implementation Plan.docx`; existing section crosswalk and source/test pointers are in [`reports/implementation/requirements-traceability.md`](../implementation/requirements-traceability.md).

## Status interpretation

`VERIFIED` is restricted to the software behavior actually exercised in the recorded synthetic/local checks. It does not mean learned capability, production data quality, or release fitness. `IMPLEMENTED — NOT VERIFIED` means the code path exists but its specified environment or artifact could not be exercised. `PARTIALLY IMPLEMENTED` means a material requested subfeature is absent. `BLOCKED BY EXTERNAL DEPENDENCY` identifies owner authorization, package-index, platform, hardware, data, or trained-artifact prerequisites. `NOT APPLICABLE — JUSTIFIED` is used only when the technical plan makes a feature conditional and the documented target does not require it.

## Aggregate at this revision

At this revision (2026-11-16), the 54 rows comprise **8 VERIFIED, 33 IMPLEMENTED — NOT VERIFIED, 6 PARTIALLY IMPLEMENTED, 7 BLOCKED BY EXTERNAL DEPENDENCY, 0 NOT IMPLEMENTED, and 0 NOT APPLICABLE — JUSTIFIED**. They are not a percent-complete score. Requirement rows have explicit verification commands and results; cross-cutting gates are separated from plan-section requirements so unavailable hardware or authorization is visible rather than hidden in a broad “implemented” status. `HARD-CI-01` records strict-mypy, Ruff security rules, and the Linux multiprocessing recovery job; GitHub execution remains unverified.

## Requirements added during final closure

| ID | Status | Requirement | Verification |
|---|---|---|---|
| `HARD-TRAIN-02` | VERIFIED | Model-block activation checkpointing with preserved RNG, gradients, and decoder cache restrictions | Full `python -m pytest -q`: 89 passed, 2 platform skips; focused architecture parity and model-specific trainer tests are included. GPU memory/throughput are unmeasured. |
| `HARD-DOC-01` | IMPLEMENTED — NOT VERIFIED | Read-only doctor/preflight with truthful statuses for untested backends, missing artifacts/approval, package state, and hardware | Doctor/preflight behavior tests pass. Current source-path invocation reports blocked without network or training; installed-mode invocation awaits clean installation. |
| `HARD-PKG-01` | VERIFIED | Package-specific Markdown full descriptions included in all eight distributions | New regression test passes; all eight wheels and sdists built and archive metadata inspected. Dependency resolution/install remains a separate blocked gate. |
| `HARD-HW-02` | VERIFIED | Preflight reports a persistent AdamW state lower bound without claiming peak fit | Exact 100M/300M byte arithmetic and WARNING semantics pass tests; CUDA peak remains `NOT TESTED`. |

The legacy table is retained to preserve its detailed evidence. Its historic labels (“IMPLEMENTED AND VERIFIED” etc.) should be read with the limits described above; this current registry is the authoritative status view for 2026-11-16.

Two hardening rows intentionally have no unit-test node: `HARD-SEC-01` is a
vulnerability database scan and `HARD-SEC-02` requires a lock-resolved
transitive dependency graph. Both remain blocked and explicitly identify the
required external scanner/lock action; no source test is presented as a
substitute.

## Dependency graph

The principal prerequisites are: approved tokenizer ID registry → frozen shared tokenizer → data tokenization/shards → model objectives and evaluation; resolved package environment → installed-mode preflight → runnable future training commands; approved dataset manifests → leakage/provenance checks → E-300-1 and model training; target hardware → distributed/precision/memory validation → production configuration. The JSON registry records these edges per requirement.
