# ORION-300M architecture and generation audit

## Architecture result

The decoder configuration is 28 layers, width 960, 15 query heads and 5 KV
heads, with GQA, RoPE, RMSNorm, SwiGLU, causal attention and tied embeddings.
Executable counting reports **306,763,200 unique parameters** after tying,
matching the plan. Synthetic tests cover causality, cache parity, generation
termination, special IDs, and weight tying.

## Objectives and pilot

Causal LM and assistant-masked SFT software plus decoder-only, prefix-LM and
encoder-decoder pilot paths are implemented. Synthetic pilot comparisons run;
they are not evidence for selecting an architecture. E-300-1 remains pending a
fair experiment on approved data/compute with its specified objective and
metrics. Do not select the final production architecture from synthetic
results.

## Limits

No trained weights or quality results exist. GPU cache performance, quantized
parity, ONNX parity and 4,096-token production memory behavior are unverified.
Run the architecture-count tool and synthetic test suite as listed in
`reports/tests/validation-report.md`.
