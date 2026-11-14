# ORION-300M architecture

The implemented default is a decoder-only causal Transformer: 28 layers, width
960, 15 query heads, 5 key/value heads, 2,560 SwiGLU intermediate width, RoPE,
RMSNorm, tied embeddings, and a 4,096-token context. Its exact body count is
306,763,200 unique parameters. The model returns causal CE plus the configured
z-loss and supports a static KV cache.

The SDK implements local offline generation, stop IDs/sequences, temperature,
top-k/top-p, repetition penalty, deterministic seeds, batch generation, and
token-level `stream_ids`. Task prompts include explicitly supplied context only.
No-context answer tasks return an abstention status without calling the model.
Citation markers map to supplied document IDs and optional page/slide metadata;
they are never described as verified evidence.

MCQ, quiz, question, and flashcard paths ask the model for JSON and parse the
result. Invalid JSON is reported as `invalid_structured_output`; there is no
constrained JSON grammar decoder or schema-specific validation yet.

The plan requires an architecture arbiter between decoder-only and
encoder-decoder/prefix-LM alternatives. The pilot software now implements all
three candidates: decoder-only, a bidirectional-prefix/causal-target mode, and
a tied-embedding GQA encoder-decoder. `orion300m architecture-pilot` trains the
configured candidates on the same local paired-token splits and writes
parameter counts, training/validation losses, elapsed time, hashes, and
checkpoints to an E-300-1 report. This runner was exercised only on tiny
synthetic CPU fixtures. The real architecture decision remains provisional
until the owner approves the data and compute and the planned pilot is run.
