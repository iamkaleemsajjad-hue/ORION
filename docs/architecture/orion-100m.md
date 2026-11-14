# ORION-100M architecture

The model is a bidirectional, pre-LayerNorm encoder. Its default body has 16
blocks, width 640, 10 heads, 1,728 GeGLU intermediate channels, and no linear
biases. Every fourth block is global; other blocks use a 128-token bidirectional
window. RoPE uses theta 10,000 in local blocks and 160,000 in global blocks.
Embedding and normalization layers follow the initialization in code.

The body count is 100,291,840 parameters. This excludes task heads. `tests/test_architectures.py`
constructs the default model on PyTorch's meta device and asserts the count
without allocating weight storage.

The software supplies sequence, multi-label, metadata BIO, confidence, and
registry-prototype heads. The separate RTD pretrainer uses an 8-layer 384-wide
MLM generator, shared token embedding, token discriminator, whole-word masking
when aligned word IDs are provided, and a 50x RTD loss multiplier. Current
production training config requires a word-ID shard by default.

The SDK accepts text already extracted by a host. It has no file parser. Course
matching uses only a caller-provided course registry, and unknown thresholds
are read from trained release metadata. Embeddings expose normalized leading
dimensions 640/256/128; the software supports this shape, but quality at each
prefix depends on future Matryoshka training.

No ORION-100M weights have been trained. The tokenizer-ID spelling gaps in the
provided plan are documented in the decision record and must be approved before
the tokenizer is frozen.
