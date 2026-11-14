# Architecture overview

ORION contains two separate models. They share tokenizer IDs and infrastructure,
not weights or task behavior.

## ORION-100M

The encoder is 16 pre-normalized Transformer blocks at width 640, 10 attention
heads, GeGLU width 1,728, and a 32,768-entry embedding table. Every fourth block
uses global attention; the others use a 128-token bidirectional window. Local
and global layers use RoPE with the planned theta values. The encoder body has
100,291,840 trainable parameters. Classification and extraction heads are
additional and depend on the released label maps.

The inference contract takes structured text that a host application already
parsed. The SDK never opens PDFs or moves files. Course/topic matching accepts a
caller-supplied registry and can abstain. A fresh random instance is not a
classifier; public prediction methods require an artifact marked trained.

## ORION-300M

The default generator is a 28-layer causal Transformer at width 960, with 15
query heads, 5 key/value heads, a 2,560-wide SwiGLU, RoPE at theta 100,000,
RMSNorm, and tied input/output embeddings. Its planned body contains
306,763,200 trainable parameters and supports up to 4,096 prompt tokens. The
software implements a causal LM and cached generation; retrieval grounding,
abstention, citations, and structured output still require training plus host
validation.

## Shared boundaries

`orion-core` owns low-level blocks, configuration, hashing, and safe tensor IO.
`orion-tokenizer`, data, training, and evaluation tools can be installed
separately. The models do not import each other or MADAD. The adapter converts
typed host payloads but does not touch files or a database.

