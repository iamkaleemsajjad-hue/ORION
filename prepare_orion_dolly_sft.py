from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orion_tokenizer.tokenizer import OrionTokenizer
from orion_300m.sft import render_chat_messages, write_sft_jsonl

ROOT = Path.cwd()
SOURCE = ROOT / "datasets" / "raw" / "dolly-15k" / "databricks-dolly-15k.jsonl"
TOKENIZER_DIR = ROOT / "artifacts" / "tokenizers" / "orion-tokenizer-v1"
OUT = ROOT / "datasets" / "prepared" / "dolly"
MAX_LENGTH = 2048

tokenizer = OrionTokenizer.load(TOKENIZER_DIR)
conversations = {"train": [], "validation": [], "test": []}
skipped = 0

for line in SOURCE.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip()
    context = str(row.get("context", "")).strip()
    if not instruction or not response:
        skipped += 1
        continue

    user_content = instruction
    if context:
        user_content += "\n\nContext:\n" + context
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    try:
        render_chat_messages(messages, tokenizer, max_length=MAX_LENGTH)
    except ValueError:
        skipped += 1
        continue

    key = hashlib.sha256((instruction + "\n" + context + "\n" + response).encode("utf-8")).digest()
    bucket = int.from_bytes(key[:8], "big") / 2**64
    split = "train" if bucket < 0.98 else ("validation" if bucket < 0.99 else "test")
    conversations[split].append(messages)

OUT.mkdir(parents=True, exist_ok=True)
for split, items in conversations.items():
    if not items:
        raise RuntimeError(f"No valid Dolly conversations for {split}")
    target = OUT / f"sft-{split}.jsonl"
    if target.exists():
        target.unlink()
    write_sft_jsonl(items, tokenizer, target, max_length=MAX_LENGTH)
    print(split, "records:", len(items), "file:", target)

print("Skipped invalid/overlength rows:", skipped)
print("Tokenizer SHA256:", tokenizer.sha256)
