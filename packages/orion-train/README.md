# orion-train

Reusable PyTorch training loops, optimizer and scheduler construction, deterministic state handling, checkpoint recovery, and distributed-training interfaces for ORION models.

The package supplies training software, not trained weights or production metrics. Run the model-specific preflight and a synthetic smoke test before any authorized full training run. CPU smoke tests do not establish full-model training capacity.

The package's resume behavior depends on the selected sampler/data-loader execution mode; consult the recovery documentation for verified guarantees and platform-specific validation gates.
