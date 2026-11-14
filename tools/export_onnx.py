"""Export a trained local model to ONNX after hash-verified offline loading."""

from __future__ import annotations

import argparse

import torch
from torch import nn


class EncoderExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(input_ids, attention_mask=attention_mask.bool()).last_hidden_state


class DecoderExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask=attention_mask.bool()).logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("orion-100m", "orion-300m"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.model == "orion-100m":
        from orion_100m.sdk import Orion100M

        sdk = Orion100M.load(device=args.device)
        wrapper = EncoderExport(sdk.model).eval()
        width = min(16, sdk.model.config.max_sequence_length)
        dummy_ids = torch.ones((1, width), dtype=torch.long, device=sdk.device)
        dummy_mask = torch.ones_like(dummy_ids, dtype=torch.bool)
    else:
        from orion_300m.sdk import Orion300M

        sdk = Orion300M.load(device=args.device)
        wrapper = DecoderExport(sdk.model).eval()
        width = min(16, sdk.model.config.context_length)
        dummy_ids = torch.full(
            (1, width), sdk.model.config.bos_token_id, dtype=torch.long, device=sdk.device
        )
        dummy_mask = torch.ones_like(dummy_ids, dtype=torch.bool)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        args.output,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states" if args.model == "orion-100m" else "logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "hidden_states" if args.model == "orion-100m" else "logits": {
                0: "batch",
                1: "sequence",
            },
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported locally to {args.output}; validate numerical parity before deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
