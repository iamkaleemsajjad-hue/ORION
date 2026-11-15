# Performance and resource audit

No hardware performance numbers are claimed. The environment was Windows with
PyTorch 2.9.1+cpu and no CUDA device. No GPU, NCCL, FlashAttention, FSDP,
quantization, ONNX Runtime, throughput, peak-memory, latency, or model-size
benchmark was run.

Software paths include PyTorch SDPA-compatible attention, CPU execution,
memmapped prepared shards and decoder KV caching. Synthetic tests verify basic
functional behavior and cached/uncached output parity, but not speed or memory
improvement. ONNX export tooling exists; exported-model parity is unverified.

Before choosing training hardware or enabling optimized kernels, run the
synthetic benchmark harness on target CPU/GPU hardware and compare to reference
outputs using documented tolerances. Measure peak memory, throughput, latency
and numerical drift for actual sequence/batch settings. Do not extrapolate
these results to trained-model quality.
