# ORION-300M audit

## Verified software

The configured decoder uses 28 layers, hidden size 960, 15 query heads and 5 KV heads, GQA, RoPE, RMSNorm, SwiGLU, causal attention, tied token embeddings and a 4,096-token context limit. Executable unique parameter count is **306,763,200** with the tied matrix counted once. Synthetic tests exercise causal masking, future-token isolation, logits, cache behavior, generation stopping, deterministic sampling controls, and tied weights.

Causal LM loss, assistant-masked SFT, cached and uncached decoding, greedy/sampling controls, streaming interfaces, context/citation validation, and abstention contracts are implemented. The SDK fails honestly when trained weights are missing and does not return canned model answers.

## E-300-1 and remaining gates

Decoder-only, prefix-LM, and encoder-decoder pilot paths exist and run on synthetic source/target inputs. Synthetic validation losses are not evidence for architecture selection. Run the documented E-300-1 command only after data, token IDs, compute budget, and comparable configurations are approved; retain architecture decision as pending until real-data outcomes are reviewed. No model training or quality evaluation occurred. CUDA cache performance, quantized/ONNX parity, and 4,096-token target-memory use are unverified.

Evidence: [`orion-300m-audit.md`](../implementation/orion-300m-audit.md), [`E-300-1 config`](../../configs/evaluation/e-300-1-architecture-pilot.json), `python tools/count_params.py`, and `tests/test_architecture_pilot.py`.
