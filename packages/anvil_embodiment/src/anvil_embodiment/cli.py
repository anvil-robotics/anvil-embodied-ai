"""Command line interface for offline embodiment adapter workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .residual import AdapterLossWeights
from .workflow import (
    cache_policy_predictions,
    evaluate_adapter_cache,
    train_residual_adapter,
    validate_adapter_contract,
)


def _add_manifest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil-adapter",
        description="Offline-first OpenArm embodiment bridge and residual adapter",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="verify hashes, dimensions, units, and sampled IK coverage"
    )
    _add_manifest(validate)
    validate.add_argument("--base-policy", type=Path)
    validate.add_argument("--dataset", type=Path)
    validate.add_argument("--stride", type=int, default=500)
    validate.add_argument("--video-backend", default="pyav")

    cache = commands.add_parser(
        "cache", help="cache frozen Hugging Face policy predictions and target chunks"
    )
    _add_manifest(cache)
    cache.add_argument("--base-policy", type=Path, required=True)
    cache.add_argument("--baseline-policy", type=Path)
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--split-info", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--task", default="Fold the T-shirt properly")
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--stride", type=int, default=10)
    cache.add_argument("--seed", type=int, default=42)
    cache.add_argument("--video-backend", default="pyav")

    train = commands.add_parser(
        "train", help="train only the bounded residual from a frozen prediction cache"
    )
    _add_manifest(train)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--joint-weight", type=float, default=1.0)
    train.add_argument("--pose-weight", type=float, default=0.25)
    train.add_argument("--velocity-weight", type=float, default=0.05)
    train.add_argument("--residual-weight", type=float, default=0.01)
    train.add_argument("--wandb-project")
    train.add_argument("--wandb-entity")
    train.add_argument("--wandb-run-name")
    train.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )

    evaluate = commands.add_parser(
        "evaluate", help="compare hold, bridge, trained adapter, and optional 5k baseline"
    )
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_adapter_contract(
            manifest=args.manifest,
            base_policy=args.base_policy,
            dataset_path=args.dataset,
            stride=args.stride,
            video_backend=args.video_backend,
        )
    elif args.command == "cache":
        result = cache_policy_predictions(
            manifest=args.manifest,
            base_policy=args.base_policy,
            baseline_policy=args.baseline_policy,
            dataset_path=args.dataset,
            split_info=args.split_info,
            output=args.output,
            task=args.task,
            device=args.device,
            stride=args.stride,
            seed=args.seed,
            video_backend=args.video_backend,
        )
    elif args.command == "train":
        result = train_residual_adapter(
            manifest=args.manifest,
            cache=args.cache,
            output=args.output,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            seed=args.seed,
            loss_weights=AdapterLossWeights(
                joint=args.joint_weight,
                pose=args.pose_weight,
                velocity=args.velocity_weight,
                residual=args.residual_weight,
            ),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
        )
    else:
        result = evaluate_adapter_cache(
            adapter=args.adapter,
            cache=args.cache,
            device=args.device,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
