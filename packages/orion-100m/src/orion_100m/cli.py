"""Command-line interface for local ORION-100M inference and explicitly invoked runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from orion_100m.config import Orion100MConfig
from orion_100m.sdk import Orion100M


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orion100m")
    parser.add_argument("--version", action="version", version="orion-100m 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "predict", "extract"):
        item = commands.add_parser(name)
        item.add_argument("--text", help="document text; defaults to stdin")
        item.add_argument("--input-json", help="local JSON file containing a document object")
        if name in {"classify", "predict"}:
            item.add_argument("--registry-json", help="local JSON course registry")
            item.add_argument("--top-k", type=int, default=3)
        if name == "extract":
            item.add_argument("--fields", nargs="*")
    batch = commands.add_parser("predict-batch")
    batch.add_argument("input_json", help="local JSON array of document payloads")
    batch.add_argument("--batch-size", type=int, default=32)
    batch.add_argument("--top-k", type=int, default=3)
    embed = commands.add_parser("embed")
    embed.add_argument("--text", action="append", dest="texts", help="repeat for multiple texts")
    embed.add_argument("--dimension", type=int, choices=(128, 256, 640), default=640)
    for name in ("info", "health"):
        commands.add_parser(name)
    train = commands.add_parser("train")
    train.add_argument("--tokens", required=True)
    train.add_argument("--word-ids", help="aligned int32 word_ids.i32 shard for whole-word masking")
    train.add_argument("--output", required=True)
    train.add_argument("--config-json", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this training run"
    )
    finetune = commands.add_parser("finetune")
    finetune.add_argument("--data-jsonl", required=True)
    finetune.add_argument("--labels-json", required=True)
    finetune.add_argument(
        "--init-checkpoint", required=True, help="verified RTD checkpoint directory or step"
    )
    finetune.add_argument("--output", required=True)
    finetune.add_argument("--config-json", required=True)
    finetune.add_argument("--device", default="auto")
    finetune.add_argument("--resume", action="store_true")
    finetune.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this training run"
    )
    contrastive = commands.add_parser("train-contrastive")
    contrastive.add_argument("--data-jsonl", required=True)
    contrastive.add_argument("--init-checkpoint", required=True)
    contrastive.add_argument("--output", required=True)
    contrastive.add_argument("--config-json", required=True)
    contrastive.add_argument("--device", default="auto")
    contrastive.add_argument("--resume", action="store_true")
    contrastive.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this training run"
    )
    course_matching = commands.add_parser("train-course-matching")
    course_matching.add_argument("--registry-json", required=True)
    course_matching.add_argument("--data-jsonl", required=True)
    course_matching.add_argument("--labels-json", required=True)
    course_matching.add_argument("--init-checkpoint", required=True)
    course_matching.add_argument("--output", required=True)
    course_matching.add_argument("--config-json", required=True)
    course_matching.add_argument("--device", default="auto")
    course_matching.add_argument("--resume", action="store_true")
    course_matching.add_argument(
        "--confirm-training", action="store_true", help="explicitly authorize this training run"
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--tokens", required=True)
    evaluate.add_argument(
        "--word-ids", help="aligned int32 word_ids.i32 shard for whole-word masking"
    )
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--config-json", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--max-batches", type=int)
    evaluate_finetune = commands.add_parser("evaluate-finetune")
    evaluate_finetune.add_argument("--data-jsonl", required=True)
    evaluate_finetune.add_argument("--labels-json", required=True)
    evaluate_finetune.add_argument("--checkpoint-dir", required=True)
    evaluate_finetune.add_argument("--config-json", required=True)
    evaluate_finetune.add_argument("--device", default="auto")
    evaluate_finetune.add_argument("--max-batches", type=int)
    evaluate_contrastive = commands.add_parser("evaluate-contrastive")
    evaluate_contrastive.add_argument("--data-jsonl", required=True)
    evaluate_contrastive.add_argument("--checkpoint-dir", required=True)
    evaluate_contrastive.add_argument("--config-json", required=True)
    evaluate_contrastive.add_argument("--device", default="auto")
    evaluate_contrastive.add_argument("--max-batches", type=int)
    evaluate_course_matching = commands.add_parser("evaluate-course-matching")
    evaluate_course_matching.add_argument("--registry-json", required=True)
    evaluate_course_matching.add_argument("--data-jsonl", required=True)
    evaluate_course_matching.add_argument("--labels-json", required=True)
    evaluate_course_matching.add_argument("--checkpoint-dir", required=True)
    evaluate_course_matching.add_argument("--config-json", required=True)
    evaluate_course_matching.add_argument("--device", default="auto")
    evaluate_course_matching.add_argument("--max-batches", type=int)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8761)
    serve.add_argument("--device", default="auto")
    return parser


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_label_map(path: str) -> Any:
    from orion_100m.model.document_model import TaskLabelMap

    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("labels-json must contain a JSON object")
    allowed = {"subjects", "document_types", "topics", "tags", "bio_tags"}
    if set(value) - allowed:
        raise ValueError("labels-json contains unsupported fields")
    if any(not isinstance(labels, list) for labels in value.values()):
        raise ValueError("label-map values must be JSON arrays")
    result = TaskLabelMap(**{key: tuple(labels) for key, labels in value.items()})
    result.validate()
    return result


def _document(args: argparse.Namespace) -> Any:
    if args.input_json:
        return _read_json(args.input_json)
    text = args.text if args.text is not None else sys.stdin.read()
    return text


def _emit(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _static_info() -> dict[str, Any]:
    config = Orion100MConfig()
    with __import__("torch").device("meta"):
        from orion_100m.model.encoder import Orion100MEncoder

        body = Orion100MEncoder(config)
    return {
        "name": config.model_name,
        "version": config.model_version,
        "status": "software_implemented_weights_not_installed",
        "trained": False,
        "pretrained_weights_imported": False,
        "body_parameters": body.parameter_count(),
        "maximum_sequence_length": config.max_sequence_length,
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
            from orion_100m.training import RTDTrainingConfig, train_rtd

            result = train_rtd(
                args.tokens,
                args.output,
                RTDTrainingConfig.model_validate(_read_json(args.config_json)),
                word_ids_path=args.word_ids,
                device=args.device,
                resume=args.resume,
            )
            _emit(result)
            return 0
        if args.command == "finetune":
            if not args.confirm_training:
                raise ValueError(
                    "Training was not started; pass --confirm-training when you intend to run it."
                )
            from orion_100m.finetuning import FineTuningConfig, train_multitask

            _emit(
                train_multitask(
                    args.data_jsonl,
                    args.output,
                    _read_label_map(args.labels_json),
                    FineTuningConfig.model_validate(_read_json(args.config_json)),
                    initial_checkpoint=args.init_checkpoint,
                    device=args.device,
                    resume=args.resume,
                )
            )
            return 0
        if args.command == "train-contrastive":
            if not args.confirm_training:
                raise ValueError(
                    "Training was not started; pass --confirm-training when you intend to run it."
                )
            from orion_100m.contrastive import ContrastiveTrainingConfig, train_contrastive

            _emit(
                train_contrastive(
                    args.data_jsonl,
                    args.output,
                    ContrastiveTrainingConfig.model_validate(_read_json(args.config_json)),
                    initial_checkpoint=args.init_checkpoint,
                    device=args.device,
                    resume=args.resume,
                )
            )
            return 0
        if args.command == "train-course-matching":
            if not args.confirm_training:
                raise ValueError(
                    "Training was not started; pass --confirm-training when you intend to run it."
                )
            from orion_100m.course_matching import (
                CourseMatchingTrainingConfig,
                train_course_matching,
            )

            _emit(
                train_course_matching(
                    args.registry_json,
                    args.data_jsonl,
                    args.output,
                    _read_label_map(args.labels_json),
                    CourseMatchingTrainingConfig.model_validate(_read_json(args.config_json)),
                    initial_checkpoint=args.init_checkpoint,
                    device=args.device,
                    resume=args.resume,
                )
            )
            return 0
        if args.command == "evaluate":
            from orion_100m.training import RTDTrainingConfig, evaluate_rtd

            _emit(
                evaluate_rtd(
                    args.tokens,
                    args.checkpoint_dir,
                    RTDTrainingConfig.model_validate(_read_json(args.config_json)),
                    word_ids_path=args.word_ids,
                    device=args.device,
                    max_batches=args.max_batches,
                )
            )
            return 0
        if args.command == "evaluate-finetune":
            from orion_100m.finetuning import FineTuningConfig, evaluate_multitask

            _emit(
                evaluate_multitask(
                    args.data_jsonl,
                    args.checkpoint_dir,
                    _read_label_map(args.labels_json),
                    FineTuningConfig.model_validate(_read_json(args.config_json)),
                    device=args.device,
                    max_batches=args.max_batches,
                )
            )
            return 0
        if args.command == "evaluate-contrastive":
            from orion_100m.contrastive import ContrastiveTrainingConfig, evaluate_contrastive

            _emit(
                evaluate_contrastive(
                    args.data_jsonl,
                    args.checkpoint_dir,
                    ContrastiveTrainingConfig.model_validate(_read_json(args.config_json)),
                    device=args.device,
                    max_batches=args.max_batches,
                )
            )
            return 0
        if args.command == "evaluate-course-matching":
            from orion_100m.course_matching import (
                CourseMatchingTrainingConfig,
                evaluate_course_matching,
            )

            _emit(
                evaluate_course_matching(
                    args.registry_json,
                    args.data_jsonl,
                    args.checkpoint_dir,
                    _read_label_map(args.labels_json),
                    CourseMatchingTrainingConfig.model_validate(_read_json(args.config_json)),
                    device=args.device,
                    max_batches=args.max_batches,
                )
            )
            return 0
        if args.command == "serve":
            import uvicorn

            from orion_100m.api import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
            return 0
        if args.command == "predict-batch":
            documents = _read_json(args.input_json)
            if not isinstance(documents, list):
                raise ValueError("input_json must contain a JSON array")
            _emit(
                {
                    "items": [
                        item.model_dump(mode="json")
                        for item in Orion100M.load().predict_batch(
                            documents, batch_size=args.batch_size, top_k=args.top_k
                        )
                    ]
                }
            )
            return 0
        if args.command == "embed":
            texts = (
                args.texts
                if args.texts is not None
                else [line.rstrip("\r\n") for line in sys.stdin if line.strip()]
            )
            model = Orion100M.load()
            _emit(
                {
                    "model": model.model_info(),
                    "dimension": args.dimension,
                    "embeddings": model.get_embeddings(texts, dim=args.dimension).tolist(),
                }
            )
            return 0
        if args.command == "health":
            _emit(Orion100M.load().health())
            return 0
        model = Orion100M.load()
        document = _document(args)
        if args.command in {"classify", "predict"}:
            registry = _read_json(args.registry_json) if args.registry_json else None
            _emit(model.classify(document, registry=registry, top_k=args.top_k))
        else:
            _emit(model.extract(document, fields=args.fields))
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
