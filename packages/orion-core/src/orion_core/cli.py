"""Unified ORION environment preflight command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orion_core.preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the read-only ``orion preflight`` command."""
    parser = argparse.ArgumentParser(prog="orion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="check software and optional training inputs"
    )
    preflight.add_argument("--root", type=Path, default=Path.cwd(), help="ORION checkout root")
    preflight.add_argument("--mode", choices=("source", "installed"), default="source")
    preflight.add_argument("--model", choices=("100m", "300m"))
    preflight.add_argument("--dataset-dir", type=Path)
    preflight.add_argument("--tokenizer-dir", type=Path)
    preflight.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    preflight.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight, printing JSON and returning 0 ready, 1 blocked, or 2 error."""
    args = build_parser().parse_args(argv)
    if args.command != "preflight":
        raise AssertionError("argparse accepted an unknown command")
    if args.model is None and any(
        value is not None for value in (args.dataset_dir, args.tokenizer_dir, args.precision)
    ):
        build_parser().error("--dataset-dir, --tokenizer-dir, and --precision require --model")
    try:
        result = run_preflight(
            args.root,
            mode=args.mode,
            model=args.model,
            dataset_dir=args.dataset_dir,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
            precision=args.precision,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "training_started": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
