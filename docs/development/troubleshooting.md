# Troubleshooting

- **`ModelWeightsNotFound`**: install a local release and point the relevant
  environment variable to it. The repository has no trained weights.
- **Tokenizer mismatch**: check tokenizer version, 32,768 vocabulary size,
  special-token IDs, and the SHA-256 in the model artifact manifest.
- **CUDA unavailable**: use `device="cpu"` for small software checks or install a
  PyTorch build compatible with the target driver. Full-model CPU inference is
  memory-heavy and was not benchmarked here.
- **Invalid RTD batch**: the shard must contain at least one ordinary token ID
  greater than or equal to 256 in each overall batch; control-only batches are
  rejected instead of producing NaN loss.
- **Checkpoint verification fails**: preserve the complete checkpoint folder,
  including model, optimizer, RNG safetensors and their checksum files, JSON
  metadata, and manifest. Do not edit a checkpoint in place.
- **Tokenizer training refuses a corpus**: it uses explicit local files only,
  requires permission confirmation, exact target vocabulary size, and may fail
  if the corpus is too small or does not support the reserved byte IDs.
- **Package import fails from source**: install workspace packages in dependency
  order or set `PYTHONPATH` to each package's `src` directory for development.
