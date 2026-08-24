"""Compatibility entry point for the prototype repository's run command."""

from __future__ import annotations

import argparse
from pathlib import Path

from tfl_coran.config import load_configs
from tfl_coran.experiments import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="YAML files, merged left to right")
    parser.add_argument("--rounds", type=int, default=5, help="number of FL-equivalent rounds")
    parser.add_argument(
        "--method",
        choices=("heuristic", "drl", "fdrl", "cfdrl", "tfl_coran"),
        default=None,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_configs(args.config)
    if args.method:
        config["method"] = args.method
    config["experiment"]["episodes"] = args.rounds * int(config["federation"]["episodes_per_round"])
    output = Path(args.output or f"runs/{config['method']}_legacy")
    summary = ExperimentRunner(config, output).run()
    print(summary)


if __name__ == "__main__":
    main()
