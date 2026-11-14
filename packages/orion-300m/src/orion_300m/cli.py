"""Command-line interface for local ORION-300M inference and explicit training runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from orion_300m.config import Orion300MConfig
from orion_300m.sdk import Orion300M


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion300m")
    parser.add_argument("--version", action="version", version="orion-300m 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "answer", "summarize", "questions", "flashcards", "mcq"):
        command = commands.add_parser(name)
        command.add_argument("--prompt", default="", help="task prompt or question")
        command.add_argument("--documents-json", help="local JSON array of context documents")
        command.add_argument("--max-new-tokens", type=int, default=256)
        command.add_argument("--temperature", type=float, default=0.2)
        command.add_argument("--top-p", type=float, default=0.9)
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--seed", type=int)
        command.add_argument("--count", type=int, default=5)
        command.add_argument("--request-id")
    for name in ("info", "health"):
        commands.add_parser(name)
    train = commands.add_parser("train")
    train.add_argument("--objective", choices=("causal", "sft"), default="causal")
    train.add_argument("--tokens")
    train.add_argument("--sft-jsonl", help="tokenized assistant-mask JSONL for SFT")
    train.add_argument("--init-checkpoint", help="verified causal pretraining checkpoint for SFT")
    train.add_argument("--output", required=True)
    train.add_argument("--config-json", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this training run"
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--objective", choices=("causal", "sft"), default="causal")
    evaluate.add_argument("--tokens")
    evaluate.add_argument("--sft-jsonl", help="tokenized held-out assistant-mask JSONL")
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--config-json", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--max-batches", type=int)
    architecture = commands.add_parser("architecture-pilot")
    architecture.add_argument("--training-jsonl", required=True)
    architecture.add_argument("--validation-jsonl", required=True)
    architecture.add_argument("--output", required=True)
    architecture.add_argument("--config-json", required=True)
    architecture.add_argument("--device", default="auto")
    architecture.add_argument("--resume", action="store_true")
    architecture.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this pilot run"
    )
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8762)
    return parser


def _emit(value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _static_info() -> dict[str, Any]:
    import torch

    from orion_300m.model.decoder import Orion300MDecoder

    config = Orion300MConfig()
    with torch.device("meta"):
        model = Orion300MDecoder(config)
    return {
        "name": config.model_name,
        "version": config.model_version,
        "status": "software_implemented_weights_not_installed",
        "trained": False,
        "pretrained_weights_imported": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "context_length": config.context_length,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            _emit(_static_info())
            return 0
        if args.command == "train":
            if not args.confirm_training:
                raise ValueError(
                    "Training was not started; pass --confirm-training when you intend to run it."
                )
            if args.objective == "sft":
                if not args.sft_jsonl:
                    raise ValueError("--sft-jsonl is required for the sft objective")
                if not args.init_checkpoint:
                    raise ValueError("--init-checkpoint is required for the sft objective")
                from orion_300m.sft import SFTTrainingConfig, train_sft_jsonl

                config = SFTTrainingConfig.model_validate(
                    json.loads(Path(args.config_json).read_text(encoding="utf-8"))
                )
                result = train_sft_jsonl(
                    args.sft_jsonl,
                    args.output,
                    config,
                    initial_checkpoint=args.init_checkpoint,
                    device=args.device,
                    resume=args.resume,
                )
            else:
                if not args.tokens:
                    raise ValueError("--tokens is required for the causal objective")
                from orion_300m.training import CausalTrainingConfig, train_causal_lm

                config = CausalTrainingConfig.model_validate(
                    json.loads(Path(args.config_json).read_text(encoding="utf-8"))
                )
                result = train_causal_lm(
                    args.tokens, args.output, config, device=args.device, resume=args.resume
                )
            _emit(result)
            return 0
        if args.command == "architecture-pilot":
            if not args.confirm_training:
                raise ValueError(
                    "Architecture pilot was not started; pass --confirm-training when authorized."
                )
            from orion_300m.architecture_pilot import (
                ArchitecturePilotConfig,
                run_architecture_pilot,
            )

            result = run_architecture_pilot(
                args.training_jsonl,
                args.validation_jsonl,
                args.output,
                ArchitecturePilotConfig.model_validate(_read_json(args.config_json)),
                device=args.device,
                resume=args.resume,
            )
            _emit(result)
            return 0
        if args.command == "evaluate":
            if args.objective == "sft":
                if not args.sft_jsonl:
                    raise ValueError("--sft-jsonl is required for the sft objective")
                from orion_300m.sft import SFTTrainingConfig, evaluate_sft_jsonl

                config = SFTTrainingConfig.model_validate(
                    json.loads(Path(args.config_json).read_text(encoding="utf-8"))
                )
                result = evaluate_sft_jsonl(
                    args.sft_jsonl,
                    args.checkpoint_dir,
                    config,
                    device=args.device,
                    max_batches=args.max_batches,
                )
            else:
                if not args.tokens:
                    raise ValueError("--tokens is required for the causal objective")
                from orion_300m.training import CausalTrainingConfig, evaluate_causal_lm

                config = CausalTrainingConfig.model_validate(
                    json.loads(Path(args.config_json).read_text(encoding="utf-8"))
                )
                result = evaluate_causal_lm(
                    args.tokens,
                    args.checkpoint_dir,
                    config,
                    device=args.device,
                    max_batches=args.max_batches,
                )
            _emit(result)
            return 0
        if args.command == "serve":
            import uvicorn

            from orion_300m.api import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
            return 0

        documents = []
        if args.documents_json:
            documents = json.loads(Path(args.documents_json).read_text(encoding="utf-8"))
            if not isinstance(documents, list):
                raise ValueError("documents-json must contain a JSON array")
        model = Orion300M.load()
        common = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "request_id": args.request_id,
        }
        if args.command == "generate":
            result = model.generate(
                args.prompt or sys.stdin.read(), documents=documents, task="generate", **common
            )
        elif args.command == "answer":
            result = model.answer(args.prompt, documents=documents, **common)
        elif args.command == "summarize":
            result = model.summarize(documents=documents, **common)
        elif args.command == "questions":
            result = model.generate_questions(documents=documents, count=args.count, **common)
        elif args.command == "flashcards":
            result = model.generate_flashcards(
                args.prompt or "the supplied material",
                documents=documents,
                count=args.count,
                **common,
            )
        else:
            result = model.generate_mcq(
                args.prompt or "the supplied material", documents=documents, **common
            )
        _emit(result)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": {
                        "type": type(exc).__name__,
                        "code": getattr(exc, "code", "ORION-E9000"),
                        "message": str(exc),
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
