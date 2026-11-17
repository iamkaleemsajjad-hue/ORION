# Performance audit

No performance numbers are claimed. The host has CPU-only PyTorch and no trained model artifacts. PyTorch SDPA, deterministic data transforms, packed memmapped shards and a preallocated decoder KV cache are implemented; tests establish functional behavior and cached/uncached parity on small inputs, not acceleration or memory savings.

Not measured: throughput, tokens/second, latency, peak memory, model load time, target 4,096-token decoder behavior, CPU/GPU batch scaling, FlashAttention, quantization, ONNX Runtime parity, or production-scale data loading. No before/after benchmark or hardware-specific optimization is asserted.

Next benchmark protocol: record device/driver, PyTorch/CUDA, precision, batch, context, warm-up and repetitions; compare each optimized path to reference output within declared tolerances; measure synchronization-aware latency, throughput and peak memory; retain raw run metadata. Run only after a target system and approved test artifacts exist. See [`performance-report.md`](../implementation/performance-report.md).
