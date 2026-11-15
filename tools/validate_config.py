"""Validate model and training JSON configuration files without starting work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=(
            "100m",
            "300m",
            "100m-training",
            "300m-training",
            "100m-finetuning",
            "100m-contrastive",
            "100m-course-matching",
            "300m-sft",
            "300m-architecture-pilot",
        ),
        required=True,
    )
    parser.add_argument("path")
    args = parser.parse_args(argv)
    raw: Any = json.loads(Path(args.path).read_text(encoding="utf-8"))
    schema: type[BaseModel]
    if args.kind == "100m":
        from orion_100m.config import Orion100MConfig

        schema = Orion100MConfig
    elif args.kind == "300m":
        from orion_300m.config import Orion300MConfig

        schema = Orion300MConfig
    elif args.kind == "100m-training":
        from orion_100m.training import RTDTrainingConfig

        schema = RTDTrainingConfig
    elif args.kind == "300m-training":
        from orion_300m.training import CausalTrainingConfig

        schema = CausalTrainingConfig
    elif args.kind == "100m-finetuning":
        from orion_100m.finetuning import FineTuningConfig

        schema = FineTuningConfig
    elif args.kind == "100m-contrastive":
        from orion_100m.contrastive import ContrastiveTrainingConfig

        schema = ContrastiveTrainingConfig
    elif args.kind == "100m-course-matching":
        from orion_100m.course_matching import CourseMatchingTrainingConfig

        schema = CourseMatchingTrainingConfig
    elif args.kind == "300m-architecture-pilot":
        from orion_300m.architecture_pilot import ArchitecturePilotConfig

        schema = ArchitecturePilotConfig
    else:
        from orion_300m.sft import SFTTrainingConfig

        schema = SFTTrainingConfig
    config = schema.model_validate(raw)
    print(
        json.dumps(
            {"valid": True, "kind": args.kind, "configuration": config.model_dump(mode="json")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
