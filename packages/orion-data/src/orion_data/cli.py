"""Dataset status and explicit acquisition commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion_data.acquire import acquire_manifest_sources
from orion_data.manifest import DatasetManifest
from orion_data.prepare import PreparationConfig, prepare_dataset, verify_prepared_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("manifest")
    validate.add_argument("--mode", choices=("research", "commercial-clean"), default="research")
    prepare = subparsers.add_parser(
        "prepare", help="prepare local, checksum-pinned JSONL into training shards"
    )
    prepare.add_argument("manifest", help="dataset manifest with local JSONL sources")
    prepare.add_argument("--tokenizer", required=True, help="local trained ORION tokenizer")
    prepare.add_argument("--output", required=True, help="new or empty output directory")
    prepare.add_argument(
        "--config",
        default="configs/data/orion-data-preparation.json",
        help="validated data preparation JSON config",
    )
    prepare.add_argument("--mode", choices=("research", "commercial-clean"), default="research")
    verify = subparsers.add_parser("verify-prepared", help="verify hashes and split invariants")
    verify.add_argument("dataset_dir")
    acquire = subparsers.add_parser("acquire", help="download explicitly pinned sources")
    acquire.add_argument("manifest")
    acquire.add_argument("--output", required=True)
    acquire.add_argument("--mode", choices=("research", "commercial-clean"), default="research")
    acquire.add_argument(
        "--execute", action="store_true", help="explicitly authorize data acquisition"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-prepared":
        print(json.dumps(verify_prepared_dataset(args.dataset_dir), indent=2))
        return 0
    if args.command == "prepare":
        raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = PreparationConfig.model_validate(raw_config)
        result = prepare_dataset(
            args.manifest,
            args.tokenizer,
            args.output,
            config=config,
            license_mode=args.mode,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    manifest = DatasetManifest.load(args.manifest)
    if args.command == "validate-manifest":
        manifest.validate_license_policy(args.mode)
        print(f"Manifest {manifest.dataset_id}: valid ({args.mode})")
        return 0
    paths = acquire_manifest_sources(
        manifest, args.output, execute=args.execute, license_mode=args.mode
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
