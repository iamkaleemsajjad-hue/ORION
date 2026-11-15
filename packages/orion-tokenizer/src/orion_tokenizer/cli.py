"""Explicit tokenizer training, validation, verification, and local export CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from orion_core.errors import OrionError
from orion_tokenizer.approval import require_tokenizer_owner_approval
from orion_tokenizer.release import export_tokenizer_release, verify_tokenizer_release
from orion_tokenizer.training import (
    TokenizerReleaseConfig,
    train_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion-tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate the tokenizer recipe")
    validate.add_argument("--config", required=True, help="tokenizer recipe JSON")

    train = subparsers.add_parser("train", help="train BPE from explicit local corpus files")
    train.add_argument("--corpus", action="append", required=True, help="local text or JSONL file")
    train.add_argument("--output", required=True, help="new or empty tokenizer release directory")
    train.add_argument("--config", required=True, help="validated tokenizer recipe JSON")
    train.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="checkout containing owner approval records",
    )
    train.add_argument("--text-field", default="text", help="JSONL field containing text")
    train.add_argument(
        "--confirm-training",
        action="store_true",
        help="explicitly authorize this tokenizer training run",
    )
    train.add_argument(
        "--approve-token-map",
        action="store_true",
        help="assert that the owner-reviewed v1 special-token map is approved",
    )

    verify = subparsers.add_parser("verify", help="verify a complete trained local release")
    verify.add_argument("--tokenizer-dir", required=True)

    export = subparsers.add_parser("export", help="copy a verified release to a local directory")
    export.add_argument("--tokenizer-dir", required=True, help="verified source release")
    export.add_argument("--output", required=True, help="new or empty export directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = _load_recipe(args.config)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "tokenizer_version": config.tokenizer_version,
                        "vocab_size": config.vocab_size,
                        "special_token_map_sha256": config.special_token_map_sha256,
                        "owner_reviewed": config.special_token_id_table_reviewed,
                        "production_tokenizer_trained": config.production_tokenizer_trained,
                        "training_started": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "train":
            if not args.confirm_training:
                print(
                    "Tokenizer training not started. Re-run with --confirm-training when authorized.",
                    file=sys.stderr,
                )
                return 2
            recipe = _load_recipe(args.config)
            training_config = recipe.training_config(approve_token_map=args.approve_token_map)
            require_tokenizer_owner_approval(
                args.project_root,
                recipe.special_token_map_sha256,
            )
            result = train_tokenizer(
                args.corpus,
                args.output,
                config=training_config,
                jsonl_text_field=args.text_field,
            )
            print(json.dumps({"status": "trained", "directory": str(result)}, indent=2))
            return 0
        if args.command == "verify":
            print(
                json.dumps(verify_tokenizer_release(args.tokenizer_dir), indent=2, sort_keys=True)
            )
            return 0
        if args.command == "export":
            export_result = export_tokenizer_release(args.tokenizer_dir, args.output)
            print(json.dumps(export_result, indent=2, sort_keys=True))
            return 0
    except (OSError, ValueError, OrionError, ValidationError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "training_started": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    return 2


def _load_recipe(path: str | Path) -> TokenizerReleaseConfig:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    return TokenizerReleaseConfig.model_validate(raw)


if __name__ == "__main__":
    raise SystemExit(main())
