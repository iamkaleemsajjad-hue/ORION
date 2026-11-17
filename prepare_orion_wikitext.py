from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path.cwd()
RAW = ROOT / "datasets" / "raw" / "wikitext-2-raw-v1"
INTERIM = ROOT / "datasets" / "interim"
MANIFESTS = ROOT / "datasets" / "manifests"
SOURCE_JSONL = INTERIM / "wikitext2_documents.jsonl"
MANIFEST = MANIFESTS / "wikitext2-local.json"

INTERIM.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

splits = {
    "train": RAW / "train-00000-of-00001.parquet",
    "validation": RAW / "validation-00000-of-00001.parquet",
    "test": RAW / "test-00000-of-00001.parquet",
}

records = []
for split, path in splits.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    texts = pq.read_table(path).column("text").to_pylist()
    text = "\n\n".join(value.strip() for value in texts if isinstance(value, str) and value.strip())
    if len(text) < 80:
        raise ValueError(f"{split} split contains too little text: {len(text)} chars")
    records.append({
        "document_id": f"wikitext2-{split}-document",
        "source_id": "wikitext-2-raw-v1",
        "split": split,
        "language": "en",
        "text": text,
    })

with SOURCE_JSONL.open("w", encoding="utf-8", newline="\n") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

digest = hashlib.sha256(SOURCE_JSONL.read_bytes()).hexdigest()
manifest = {
    "schema_version": "1.0",
    "dataset_id": "orion-wikitext2-experiment",
    "created_at": date.today().isoformat(),
    "normalization": "NFC",
    "pipeline_version": "0.1.0",
    "seed": 1729,
    "filters": ["normalize", "quality_filter", "deduplicate", "segment"],
    "synthetic_fixture": False,
    "sources": [{
        "source_id": "wikitext-2-raw-v1",
        "kind": "local-jsonl",
        "revision": "wikitext-2-raw-v1-local-2026-11-16",
        "local_path": "../interim/wikitext2_documents.jsonl",
        "sha256": digest,
        "synthetic": False,
        "license": {
            "spdx_id": "CC-BY-SA-3.0",
            "attribution": "WikiText-2 dataset; source text derived from Wikipedia articles. Retain source attribution.",
            "non_commercial": False,
            "share_alike": True,
            "verified": True,
            "source_url": "https://huggingface.co/datasets/Salesforce/wikitext"
        }
    }]
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Created:", SOURCE_JSONL)
print("Source records:", len(records), "SHA256:", digest)
print("Created manifest:", MANIFEST)
for record in records:
    print(record["split"], "characters:", len(record["text"]))
