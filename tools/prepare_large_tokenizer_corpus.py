"""
Resume-safe preparation of a larger ORION tokenizer corpus.

Run from the ORION AI project root:
    python tools\prepare_large_tokenizer_corpus.py

Optional Hugging Face authentication:
    hf auth login
or set HF_TOKEN in the current CMD session before running.

Outputs:
    datasets\raw\tokenizer-large\fineweb_edu.jsonl
    datasets\raw\tokenizer-large\wikipedia_ur.jsonl
    datasets\raw\tokenizer-large\corpus_manifest.json

The script preserves and resumes `.partial` files by replaying the source stream
and skipping already-written JSONL records. It does not train the tokenizer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path.cwd()
OUT = ROOT / "datasets" / "raw" / "tokenizer-large"
FINEWEB_LIMIT_BYTES = int(1.5 * 1024**3)
MIN_DOCUMENT_CHARS = 200
MAX_DOCUMENT_CHARS = 200_000


def inspect_partial(path: Path) -> tuple[int, int]:
    """Count complete JSONL records and discard only a torn final line."""
    if not path.exists():
        return 0, 0

    with path.open("rb+") as handle:
        data_end = handle.seek(0, os.SEEK_END)
        if data_end == 0:
            return 0, 0
        handle.seek(data_end - 1)
        if handle.read(1) != b"\n":
            handle.seek(0)
            content = handle.read()
            last_newline = content.rfind(b"\n")
            keep = last_newline + 1
            handle.truncate(keep)
        handle.seek(0)
        count = sum(1 for line in handle if line.strip())
        size = handle.seek(0, os.SEEK_END)
    return count, size


def write_stream(
    *,
    dataset_id: str,
    config: str,
    split: str,
    output_name: str,
    max_bytes: int | None,
) -> dict[str, Any]:
    destination = OUT / output_name
    partial = destination.with_suffix(destination.suffix + ".partial")

    if destination.exists():
        with destination.open("r", encoding="utf-8") as existing:
            documents = sum(1 for _ in existing)
        bytes_written = destination.stat().st_size
        print(
            f"Reusing completed {destination}: {documents:,} docs, "
            f"{bytes_written / 1024**3:.3f} GiB"
        )
        return {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "output": str(destination.relative_to(ROOT)),
            "documents": documents,
            "utf8_bytes": bytes_written,
            "reused_existing_file": True,
        }

    documents, bytes_written = inspect_partial(partial)
    if documents:
        print(
            f"Resuming {output_name} after {documents:,} existing documents "
            f"({bytes_written / 1024**2:.1f} MiB)."
        )
    else:
        print(f"\nStreaming {dataset_id} / {config} ({split})")

    stream = load_dataset(
        dataset_id,
        name=config,
        split=split,
        streaming=True,
        token=True,
    )

    skipped = 0
    mode = "a" if partial.exists() else "w"
    with partial.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in stream:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < MIN_DOCUMENT_CHARS or len(text) > MAX_DOCUMENT_CHARS:
                continue

            # Replaying the deterministic stream lets us resume without
            # downloading/storing source shards ourselves.
            if skipped < documents:
                skipped += 1
                continue

            record = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            encoded_size = len(record.encode("utf-8"))
            if max_bytes is not None and bytes_written + encoded_size > max_bytes:
                break

            handle.write(record)
            documents += 1
            bytes_written += encoded_size

            if documents % 5000 == 0:
                handle.flush()
                print(
                    f"  {output_name}: {documents:,} docs, "
                    f"{bytes_written / 1024**2:,.1f} MiB"
                )

    if documents == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"No usable documents were written for {dataset_id}/{config}")

    os.replace(partial, destination)
    print(
        f"Finished {destination}: {documents:,} docs, "
        f"{bytes_written / 1024**3:.3f} GiB"
    )
    return {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "output": str(destination.relative_to(ROOT)),
        "documents": documents,
        "utf8_bytes": bytes_written,
        "resumed": skipped > 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
            "output": "fineweb_edu.jsonl",
            "max_bytes": FINEWEB_LIMIT_BYTES,
            "license": "ODC-By 1.0; retain attribution and check dataset/Common Crawl terms",
            "purpose": "English educational web text",
        },
        {
            "dataset": "wikimedia/wikipedia",
            "config": "20231101.ur",
            "split": "train",
            "output": "wikipedia_ur.jsonl",
            "max_bytes": None,
            "license": "CC BY-SA 3.0 / GFDL; comply with attribution and share-alike terms",
            "purpose": "Urdu encyclopedia text",
        },
    ]

    results = []
    for source in sources:
        result = write_stream(
            dataset_id=source["dataset"],
            config=source["config"],
            split=source["split"],
            output_name=source["output"],
            max_bytes=source["max_bytes"],
        )
        result["license"] = source["license"]
        result["purpose"] = source["purpose"]
        results.append(result)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "JSONL, one object per line, UTF-8, field=text",
        "preprocessing": {
            "minimum_document_characters": MIN_DOCUMENT_CHARS,
            "maximum_document_characters": MAX_DOCUMENT_CHARS,
            "fineweb_edu_text_limit_bytes": FINEWEB_LIMIT_BYTES,
            "wikipedia_urdu_limit": "all streamed rows",
        },
        "sources": results,
        "note": (
            "This is a bounded tokenizer-training sample, not a complete download "
            "of the multi-terabyte source datasets. Review and retain each source's "
            "license/attribution requirements."
        ),
    }
    manifest_path = OUT / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nCorpus preparation complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
