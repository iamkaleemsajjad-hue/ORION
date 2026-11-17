# E-300-1 architecture experiment readiness

The ORION-300M package includes decoder-only, prefix-LM, and pilot encoder-decoder candidates plus a runner that records candidate configs, parameters, training/validation losses, data hashes, elapsed time, and resumable checkpoints. The command requires an explicit training confirmation. Synthetic tests exercise the framework; no authorized real-data comparison was run.

The source plan keeps decoder-only as a provisional default and describes an empirical architecture arbiter. A winning architecture is **not established**. Before execution, approve the same source-target training data and splits, a comparable pilot parameter/compute budget, objective and optimizer settings, evaluation suite, and decision threshold. The plan's criteria and caveats are in [architecture decisions](../../docs/architecture/decisions.md) and [ORION-300M guide](../../docs/architecture/orion-300m.md).

**Result:** infrastructure is available; experiment and selection remain pending authorized data/compute. The exact implemented command and required configuration are documented in `docs/training/overview.md`; confirm the CLI help and validate the config on the resolved installed environment before launching. No real experiment was performed, and no empirical winner is claimed.

The runbook's command form is:

```powershell
uv run orion300m architecture-pilot --training-jsonl .\experiments\tutor\e-300-1\train.jsonl --validation-jsonl .\experiments\tutor\e-300-1\validation.jsonl --output .\experiments\tutor\e-300-1\runs --config-json .\configs\evaluation\e-300-1-architecture-pilot.json --confirm-training
```

This command was **not** executed on real data; its CLI implementation exists and synthetic candidate tests run in the full source suite.
