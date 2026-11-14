"""Count default model parameters without allocating full model weights."""

from __future__ import annotations

import json

import torch

from orion_100m.config import Orion100MConfig
from orion_100m.model.encoder import Orion100MEncoder
from orion_300m.config import Orion300MConfig
from orion_300m.model.decoder import Orion300MDecoder


def main() -> int:
    with torch.device("meta"):
        encoder = Orion100MEncoder(Orion100MConfig())
        decoder = Orion300MDecoder(Orion300MConfig())
    result = {
        "orion_100m": {
            "body_parameters": encoder.parameter_count(),
            "target_body_parameters": 100_291_840,
            "matches_plan": encoder.parameter_count() == 100_291_840,
        },
        "orion_300m": {
            "unique_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
            "target_parameters": 306_763_200,
            "matches_plan": sum(parameter.numel() for parameter in decoder.parameters())
            == 306_763_200,
            "embeddings_tied": decoder.token_embeddings.weight is decoder.lm_head.weight,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all(item["matches_plan"] for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
