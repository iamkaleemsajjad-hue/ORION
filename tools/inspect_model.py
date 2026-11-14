"""Inspect model configuration contracts and exact default parameter counts."""

from __future__ import annotations

import argparse
import json

import torch

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import Orion100MEncoder
from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("orion-100m", "orion-300m"))
    args = parser.parse_args(argv)
    with torch.device("meta"):
        if args.model == "orion-100m":
            config = Orion100MConfig()
            model = Orion100MEncoder(config)
            params = model.parameter_count()
        else:
            config = Orion300MConfig()
            model = Orion300MDecoder(config)
            params = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "parameters": params,
                "weights_allocated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
