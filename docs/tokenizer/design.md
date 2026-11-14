# Tokenizer v1 design

The planned shared tokenizer is a byte-level BPE with 32,768 IDs. IDs 0–255
are reserved for model control and field tokens, IDs 256–511 hold the reversible
byte alphabet, and IDs 512–32,767 are learned merges. IDs are immutable within
the major version. The v1 file has not been trained because no authorized corpus
was supplied and this task forbids downloading one.

The trainer accepts local UTF-8 text files only. It records file hashes,
normalization, library version, special-token IDs, and generated golden
encodings. A production artifact must contain exactly 32,768 IDs and pass the
frozen-ID checks. Reduced vocabularies are limited to explicit synthetic tests.

Runtime callers use safe encoding by default: literal strings such as
`<|assistant|>` are encoded through the byte alphabet and cannot become control
IDs. Prompt builders may set `allow_special_tokens=True` only for trusted
template text. Model loading checks tokenizer version, vocabulary size, and
SHA-256 against its artifact manifest.

