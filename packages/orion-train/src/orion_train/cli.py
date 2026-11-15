from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orion-train")
    parser.add_argument("--version", action="version", version="orion-train 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="Show training entry points and safety guarantees")
    args = parser.parse_args(argv)
    if args.command == "info":
        print(
            json.dumps(
                {
                    "status": "software_only",
                    "training_started": False,
                    "entry_points": [
                        "orion_100m.training.train_rtd",
                        "orion_300m.training.train_causal_lm",
                    ],
                    "checkpoint_format": [
                        "model.safetensors",
                        "model.safetensors.sha256",
                        "optimizer.safetensors",
                        "optimizer.safetensors.sha256",
                        "rng.safetensors",
                        "rng.safetensors.sha256",
                        "training_state.json",
                        "manifest.json",
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
