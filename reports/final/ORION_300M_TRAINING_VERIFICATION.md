# ORION-300M training software verification

## Architecture

The default decoder has 28 layers, width 960, 15 query heads and 5 KV heads, RoPE, RMSNorm, SwiGLU, causal masking, and tied input/output embeddings. The meta-device utility measured **306,763,200 unique parameters**, matching the plan and counting tied embeddings once.

## Training and recovery path

The actual `orion_300m.training` causal-LM path builds the model, AdamW groups, scheduler, stateful token-block loader, and checkpoint manager. The model-specific synthetic regression runs a small real decoder, trains with gradient checkpointing, saves and resumes to the configured step, and evaluates on a separate synthetic token shard. Causal loss, cache, and mask tests run in the final suite (**89 passed, 2 skipped, 4 warnings**).

No full ORION-300M training occurred. No generated educational answer or model quality claim is made from random initialization. GQA, cached decoding, generation controls, stop behavior, and context grounding have software tests, but no real learned response quality exists.

## Remaining gates

Owner-approved tokenizer v1 and corpora, resolved installation, real-data E-300-1 architecture comparison, target GPU memory/throughput validation, full training, and held-out educational/factual/safety evaluation. CUDA/NCCL, ONNX/quantization parity, and production-size memory are unverified.
