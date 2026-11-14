# Tokenizer training

The Tokenizers package includes a byte-level BPE trainer and local CLI. It reads
explicit text or JSONL files, applies NFC normalization, fixes the 256 special
IDs and byte alphabet, checks the requested vocabulary size, writes a local
manifest with corpus checksums, and stores golden vectors. It performs no
download or implicit corpus search.

```powershell
orion-tokenizer train --corpus .\local-approved.jsonl --output .\artifacts\tokenizer-v1 --confirm-training --approve-token-map
```

Both `--confirm-training` and the owner-reviewed `--approve-token-map` flag are
required. That command was not run here. The
repository's production tokenizer status remains `pending`, because plan table
spellings were lost in the supplied DOCX and the corpus is not approved.

The trainer must produce exactly 32,768 vocabulary entries and stable special
IDs. If a permitted corpus cannot produce the target number of merge entries,
training fails instead of padding with invented tokens. Validate Unicode,
Urdu, math, code, OCR and byte round-trips before freezing v1.
