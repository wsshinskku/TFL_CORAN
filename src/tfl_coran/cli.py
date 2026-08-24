from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import yaml

from tfl_coran.config import load_config
from tfl_coran.experiments.runner import (
    ExperimentRunner,
    generate_historical_contexts,
    run_benchmark,
    run_reproduction,
)
from tfl_coran.models.vgae import save_vgae_checkpoint, train_vgae
from tfl_coran.utils import resolve_device, seed_everything

METHODS = ["heuristic", "drl", "fdrl", "cfdrl", "tfl_coran"]


def _normalize_methods(values: list[str]) -> list[str]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with explicit methods")
        return METHODS.copy()
    return values


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if getattr(args, "method", None):
        config["method"] = args.method
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if getattr(args, "device", None):
        config["device"] = args.device
    return config


def command_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    summary = ExperimentRunner(
        config, args.output, vgae_checkpoint=args.vgae_checkpoint
    ).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    methods = _normalize_methods(args.methods)
    summaries = run_benchmark(
        config, args.output, methods=methods, vgae_checkpoint=args.vgae_checkpoint
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


def command_ablate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    config["method"] = "tfl_coran"
    variants = {
        "full": {"transfer": True, "vgae": True, "gmm": True},
        "variant_a": {"transfer": False, "vgae": True, "gmm": True},
        "variant_b": {"transfer": False, "vgae": False, "gmm": True},
        "variant_c": {"transfer": False, "vgae": True, "gmm": False},
        "variant_d": {"transfer": False, "vgae": False, "gmm": False},
    }
    results = []
    for name, toggles in variants.items():
        variant = deepcopy(config)
        variant["toggles"] = toggles
        result = ExperimentRunner(variant, Path(args.output) / name).run()
        result["variant"] = name
        result["toggles"] = toggles
        results.append(result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ablation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    methods = _normalize_methods(args.methods)
    result = run_reproduction(
        config,
        args.output,
        methods=methods,
        seeds=args.seeds,
        vgae_checkpoint=args.vgae_checkpoint,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_report(args: argparse.Namespace) -> int:
    path = Path(args.run_dir)
    candidates = [
        path / "reproduction_summary.json",
        path / "benchmark_summary.json",
        path / "ablation_summary.json",
        path / "summary.json",
    ]
    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(f"no recognized summary JSON under {path}")
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def command_pretrain(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    snapshots = generate_historical_contexts(config, int(config["seed"]))
    artifacts = train_vgae(snapshots, config["vgae"], device, int(config["seed"]) + 701)
    save_vgae_checkpoint(artifacts, config["vgae"], args.output)
    print(f"saved VGAE checkpoint to {Path(args.output).resolve()}")
    return 0


def command_doctor(_: argparse.Namespace) -> int:
    packages = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "PyYAML": yaml.__version__,
    }
    try:
        packages["tfl-coran"] = importlib.metadata.version("tfl-coran")
    except importlib.metadata.PackageNotFoundError:
        packages["tfl-coran"] = "editable source / not installed"
    packages["cuda_available"] = str(torch.cuda.is_available())
    print(json.dumps(packages, indent=2))
    return 0


def command_estimate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    num_ues = int(sum(config["environment"]["cell_ue_counts"]))
    episodes = int(config["experiment"]["episodes"])
    slots = int(config["environment"]["episode_slots"])
    state_dim = 9
    capacity = int(config["agent"]["replay_capacity"])
    transition_bytes = 2 * state_dim * 4 + 8 + 4 + 4
    maximum_replay_gib = num_ues * capacity * transition_bytes / 1024**3
    estimate = {
        "ues": num_ues,
        "environment_steps": num_ues * episodes * slots,
        "maximum_replay_gib_if_all_buffers_fill": maximum_replay_gib,
        "note": "buffers allocate lazily; runtime depends strongly on CPU/GPU and gradient settings",
    }
    print(json.dumps(estimate, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfl-coran", description="TFL-CORAN experiments for 5G O-RAN traffic control"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="show dependency and accelerator status")
    doctor.set_defaults(func=command_doctor)

    pretrain = subparsers.add_parser("pretrain-vgae", help="generate history and pretrain the VGAE")
    pretrain.add_argument("--config", "-c", required=True)
    pretrain.add_argument("--output", "-o", required=True)
    pretrain.add_argument("--seed", type=int)
    pretrain.add_argument("--device")
    pretrain.set_defaults(func=command_pretrain)

    run = subparsers.add_parser("run", help="train and evaluate one method")
    run.add_argument("--config", "-c", required=True)
    run.add_argument("--output", "-o", required=True)
    run.add_argument("--method", choices=METHODS)
    run.add_argument("--seed", type=int)
    run.add_argument("--device")
    run.add_argument("--vgae-checkpoint")
    run.set_defaults(func=command_run)

    benchmark = subparsers.add_parser("benchmark", help="run paper baselines under one configuration")
    benchmark.add_argument("--config", "-c", required=True)
    benchmark.add_argument("--output", "-o", required=True)
    benchmark.add_argument("--methods", nargs="+", default=["all"], choices=["all", *METHODS])
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--device")
    benchmark.add_argument("--vgae-checkpoint")
    benchmark.set_defaults(func=command_benchmark)

    ablate = subparsers.add_parser("ablate", help="run the component toggles from paper Table 4")
    ablate.add_argument("--config", "-c", required=True)
    ablate.add_argument("--output", "-o", required=True)
    ablate.add_argument("--seed", type=int)
    ablate.add_argument("--device")
    ablate.set_defaults(func=command_ablate)

    reproduce = subparsers.add_parser("reproduce", help="run multiple methods and seeds with mean/std/CI")
    reproduce.add_argument("--config", "-c", required=True)
    reproduce.add_argument("--output", "-o", required=True)
    reproduce.add_argument("--methods", nargs="+", default=["all"], choices=["all", *METHODS])
    reproduce.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    reproduce.add_argument("--device")
    reproduce.add_argument("--vgae-checkpoint")
    reproduce.set_defaults(func=command_reproduce)

    report = subparsers.add_parser("report", help="print a generated run summary")
    report.add_argument("run_dir")
    report.set_defaults(func=command_report)

    estimate = subparsers.add_parser("estimate", help="estimate experiment scale before running")
    estimate.add_argument("--config", "-c", required=True)
    estimate.set_defaults(func=command_estimate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
