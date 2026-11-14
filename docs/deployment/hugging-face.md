# Hugging Face Hub artifacts

`tools/hf_artifacts.py` provides explicit revision-pinned download and
verification-gated upload helpers. It imports `huggingface_hub` only when the
command runs. Install the optional `orion-data[hub]` extra and use the Hub's
secure CLI login or `HF_TOKEN`; credentials are never printed or written to a
manifest.

```powershell
python tools/hf_artifacts.py upload --model orion-300m --release-dir .\release --repo-id organization/orion-300m --confirm-upload
python tools/hf_artifacts.py download --model orion-300m --repo-id organization/orion-300m --revision FULL_40_CHARACTER_COMMIT_HASH --destination .\models\orion-300m
```

ORION-300M `artifact.json` records `tokenizer_version` alongside its hashes.
Uploads create private repositories and require `--confirm-upload`. They first
run the local release verifier and allowlist files. Downloads require an
immutable commit and are verified after transfer. Neither operation was run.
There is no trained artifact eligible for upload today.
