#!/usr/bin/env python3
"""Validate a compiled iPhone scene and zero/one-shot policy bindings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/anvil_shared/src"))

from anvil_shared.embodiment import EmbodimentError, ExperimentContract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("zero", "one"), default="zero")
    args = parser.parse_args()
    try:
        raw = yaml.safe_load(args.config.read_text())
        if not isinstance(raw, dict):
            raise EmbodimentError("experiment config must be an object")
        experiment = ExperimentContract.from_dict(raw, args.config.parent)
        embodiment = experiment.validate(mode=args.mode)
    except (OSError, yaml.YAMLError, EmbodimentError) as exc:
        parser.error(str(exc))
    print(
        f"Valid {args.mode}-shot experiment: {len(experiment.bindings)} models, "
        f"{len(experiment.seeds)} seeds, embodiment={embodiment.robot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
