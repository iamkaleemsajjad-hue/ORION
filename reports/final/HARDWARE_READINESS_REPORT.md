# Hardware readiness

## Observed host

Windows AMD64, 12 logical CPUs, CPython 3.10.11, PyTorch 2.9.1+cpu. Live doctor observed about 0.38 GiB currently available RAM and 111.54 GiB free workspace disk at the time of that check. These are transient host readings, not training capacity estimates. CUDA is unavailable; Gloo backend is compiled but communication is untested; NCCL is unavailable.

## Plan targets

The supplied plan lists a 24-GB GPU as minimum experimental hardware for reduced pilots and ORION-100M experiments, and a single 80-GB A100/H100 as recommended development hardware for planned token budgets. These are plan assumptions, not measured ORION performance. The plan describes 32-64 GB system RAM and NVMe storage assumptions; actual workload fit remains to be measured.

## Verification limits

CPU synthetic training tests pass, but no full-model CPU training readiness is claimed. No GPU inference/training, BF16/FP16 backend, CUDA/NCCL DDP, peak memory, throughput, tokens/s, or target-device preflight was measured. Current preflight correctly labels target peak memory `NOT TESTED` and CPU full-scale training as requiring external hardware. Provide exact target GPU/driver/CUDA details and run model-specific preflight plus a measured memory smoke before production training.


## Persistent-state lower bound

Preflight computes a transparent FP32 floor for parameters, gradients, and two AdamW moment tensors: ORION-100M is 1,604,669,440 bytes (about 1.49 GiB), and ORION-300M is 4,908,211,200 bytes (about 4.57 GiB). This excludes activation memory, temporary workspaces, allocator reserve, DataLoader memory, and the 100M RTD generator/heads. It is an arithmetic lower bound, not peak memory or a fit guarantee. No accelerator was present to measure full-step peak memory.
