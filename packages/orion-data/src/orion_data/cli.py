"""Dataset status and explicit acquisition commands."""

from __future__ import annotations

import argparse

from orion_data.acquire import acquire_manifest_sources
from orion_data.manifest import DatasetManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("manifest")
    validate.add_argument("--mode", choices=("research", "commercial-clean"), default="research")
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
