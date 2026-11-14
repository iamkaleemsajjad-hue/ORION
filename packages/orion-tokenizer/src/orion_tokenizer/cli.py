"""Explicit tokenizer training CLI; all corpus inputs must be local files."""

from __future__ import annotations

import argparse
import sys

from orion_tokenizer.training import TokenizerTrainingConfig, train_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion-tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train BPE from explicit local corpus files")
    train.add_argument("--corpus", action="append", required=True, help="local text or JSONL file")
    train.add_argument("--output", required=True, help="new or empty tokenizer release directory")
    train.add_argument("--text-field", default="text", help="JSONL field containing text")
    train.add_argument("--vocab-size", type=int, default=32768)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument(
        "--confirm-training",
        action="store_true",
        help="explicitly authorize this tokenizer training run",
    )
    train.add_argument(
        "--approve-token-map",
        action="store_true",
        help="confirm the owner-reviewed v1 special-token ID map",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if not args.confirm_training:
            print(
                "Tokenizer training not started. Re-run with --confirm-training when authorized.",
                file=sys.stderr,
            )
            return 2
        result = train_tokenizer(
            args.corpus,
            args.output,
            config=TokenizerTrainingConfig(
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                seed=args.seed,
                token_map_approved=args.approve_token_map,
            ),
            jsonl_text_field=args.text_field,
        )
        print(f"Tokenizer artifact written to {result}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
