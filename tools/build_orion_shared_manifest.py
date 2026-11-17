
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path.cwd()
RAW = ROOT / "datasets" / "raw" / "tokenizer-large"
OUT = ROOT / "datasets" / "processed" / "orion-shared-source"
OUT.mkdir(parents=True, exist_ok=True)

sources = [
    {
        "source_id": "fineweb-edu",
        "input": RAW / "fineweb_edu.jsonl",
        "output": OUT / "fineweb_edu.jsonl",
        "revision": "local-snapshot-2026-11-16",
        "license": {
            "spdx_id": "ODC-By-1.0",
            "attribution": "HuggingFaceFW/fineweb-edu, sample-10BT",
            "non_commercial": False,
            "share_alike": False,
            "verified": True,
            "source_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        },
        "language": "en",
    },
    {
        "source_id": "wikipedia-ur",
        "input": RAW / "wikipedia_ur.jsonl",
        "output": OUT / "wikipedia_ur.jsonl",
        "revision": "local-snapshot-2026-11-16",
        "license": {
            "spdx_id": "CC-BY-SA-3.0",
            "attribution": "Wikimedia Foundation, Urdu Wikipedia; CC BY-SA 3.0",
            "non_commercial": False,
            "share_alike": True,
            "verified": True,
            "source_url": "https://dumps.wikimedia.org/",
        },
        "language": "ur",
    },
]

manifest_sources = []

for source in sources:
    count = 0
    digest = hashlib.sha256()

    with source["input"].open("r", encoding="utf-8") as src, \
         source["output"].open("w", encoding="utf-8", newline="\n") as dst:
        for line_number, line in enumerate(src, 1):
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            record = {
                "document_id": f'{source["source_id"]}-{line_number:09d}',
                "text": text,
                "source_id": source["source_id"],
                "language": source["language"],
            }
            encoded = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            dst.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1

    manifest_sources.append({
        "source_id": source["source_id"],
        "kind": "local-jsonl",
        "revision": source["revision"],
        "license": source["license"],
        "local_path": str(source["output"].resolve()),
        "sha256": digest.hexdigest(),
        "synthetic": False,
    })
    print(f'{source["source_id"]}: {count:,} records')

manifest = {
    "schema_version": "1.0",
    "dataset_id": "orion-fineweb-edu-urdu-wikipedia-v1",
    "created_at": date.today().isoformat(),
    "sources": manifest_sources,
    "normalization": "NFC",
    "pipeline_version": "0.1.0",
    "seed": 1729,
    "filters": [],
    "synthetic_fixture": False,
}

manifest_path = ROOT / "datasets" / "manifests" / "orion-shared-v1.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"\nManifest created: {manifest_path}")
