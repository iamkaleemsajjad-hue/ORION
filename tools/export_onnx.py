"""Export a trained local model to ONNX after hash-verified offline loading."""

from __future__ import annotations

import argparse
from typing import cast

import torch
from torch import nn

from orion_100m.model.encoder import EncoderOutput
from orion_100m.model.tasks import Orion100MForTasks
from orion_300m.model.decoder import DecoderOutput, Orion300MDecoder


class EncoderExport(nn.Module):
    def __init__(self, model: Orion100MForTasks) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = cast(
            EncoderOutput,
            self.model.encoder(input_ids, attention_mask=attention_mask.bool()),
        )
        return output.last_hidden_state


class DecoderExport(nn.Module):
    def __init__(self, model: Orion300MDecoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = cast(DecoderOutput, self.model(input_ids, attention_mask=attention_mask.bool()))
        return output.logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("orion-100m", "orion-300m"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    wrapper: nn.Module
    if args.model == "orion-100m":
        from orion_100m.sdk import Orion100M

        sdk_100m = Orion100M.load(device=args.device)
        encoder_wrapper = EncoderExport(sdk_100m.model).eval()
        wrapper = encoder_wrapper
        width = min(16, sdk_100m.model.config.max_sequence_length)
        dummy_ids = torch.ones((1, width), dtype=torch.long, device=sdk_100m.device)
        dummy_mask = torch.ones_like(dummy_ids, dtype=torch.bool)
    else:
        from orion_300m.sdk import Orion300M

        sdk_300m = Orion300M.load(device=args.device)
        decoder_wrapper = DecoderExport(sdk_300m.model).eval()
        wrapper = decoder_wrapper
        width = min(16, sdk_300m.model.config.context_length)
        dummy_ids = torch.full(
            (1, width),
            sdk_300m.model.config.bos_token_id,
            dtype=torch.long,
            device=sdk_300m.device,
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
