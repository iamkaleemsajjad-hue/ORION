from __future__ import annotations

import json
from pathlib import Path

source = Path("packages/orion-tokenizer/src/orion_tokenizer/training.py")
text = source.read_text(encoding="utf-8")

needle = '''    backend.train_from_iterator(
        iter_local_text(corpus_files, field=jsonl_text_field),
        trainer=trainer,
        length=None,
    )
'''
insert = needle + '''
    # Freeze ORION's byte alphabet at IDs 256..511. The Tokenizers BPE trainer
    # sorts its initial alphabet internally, so remap model vocabulary IDs after
    # training. BPE merges are token-string pairs and remain valid after remapping.
    serialized = json.loads(backend.to_str())
    model_vocab = serialized["model"]["vocab"]
    ordered_specials = list(SPECIAL_TOKENS)
    ordered_bytes = [byte_map[byte] for byte in range(256)]

    missing = [
        token for token in ordered_specials + ordered_bytes
        if token not in model_vocab
    ]
    if missing:
        raise ValueError(
            f"Cannot freeze ORION token IDs; missing {len(missing)} required tokens"
        )

    special_set = set(ordered_specials)
    byte_set = set(ordered_bytes)
    remaining = [
        token
        for token, _old_id in sorted(model_vocab.items(), key=lambda item: item[1])
        if token not in special_set and token not in byte_set
    ]
    if len(remaining) + 512 != training_config.vocab_size:
        raise ValueError(
            "Vocabulary allocation mismatch after reserving special and byte IDs: "
            f"{len(remaining)} learned tokens"
        )

    frozen_vocab = {
        token: index
        for index, token in enumerate(ordered_specials + ordered_bytes + remaining)
    }
    serialized["model"]["vocab"] = frozen_vocab

    # Keep added-token records consistent with the remapped model vocabulary.
    for added in serialized.get("added_tokens", []):
        content = added.get("content")
        if content in frozen_vocab:
            added["id"] = frozen_vocab[content]

    backend = Tokenizer.from_str(json.dumps(serialized, ensure_ascii=False))
'''
if text.count(needle) != 1:
    raise SystemExit(
        "Patch stopped safely: expected exactly one insertion point; "
        f"found {text.count(needle)}"
    )
if "Freeze ORION's byte alphabet at IDs 256..511" in text:
    raise SystemExit("Patch already appears to be installed; no changes made.")

source.write_text(text.replace(needle, insert), encoding="utf-8")
print("Patched tokenizer trainer: special IDs 0-255 and byte IDs 256-511 are now explicitly frozen.")
