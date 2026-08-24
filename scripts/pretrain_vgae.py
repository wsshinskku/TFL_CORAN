"""Compatibility VGAE pretraining entry point."""

from __future__ import annotations

import argparse

from tfl_coran.config import load_config
from tfl_coran.experiments.runner import generate_historical_contexts
from tfl_coran.models.vgae import save_vgae_checkpoint, train_vgae
from tfl_coran.utils import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--snapshots", help="legacy option; simulator history is generated from config")
    parser.add_argument("--gen-if-missing", action="store_true", help="retained for CLI compatibility")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["vgae"]["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    history = generate_historical_contexts(config, int(config["seed"]))
    artifacts = train_vgae(history, config["vgae"], device, int(config["seed"]) + 701)
    save_vgae_checkpoint(artifacts, config["vgae"], args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
