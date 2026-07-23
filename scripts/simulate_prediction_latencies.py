#!/usr/bin/env python3
"""Generate deterministic synthetic prediction-latency profiles.

This is a capacity-planning tool, not a hardware benchmark.  Baselines are kept
in ``configs/lerobot_control/prediction_latency_baselines.json`` and should be
replaced with measurements from the deployment GPU whenever they are available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = ROOT / "configs/lerobot_control/prediction_latency_baselines.json"
DEFAULT_OUTPUT = ROOT / "configs/lerobot_control/prediction_latencies.json"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def simulate_lognormal(mean_ms: float, cv: float, count: int, rng: random.Random) -> list[float]:
    sigma = math.sqrt(math.log1p(cv * cv))
    mu = math.log(mean_ms) - sigma * sigma / 2
    return [rng.lognormvariate(mu, sigma) for _ in range(count)]


def build_report(baselines: dict, samples: int, seed: int, control_hz: float) -> dict:
    models = {}
    for index, (name, baseline) in enumerate(sorted(baselines["models"].items())):
        values = simulate_lognormal(
            float(baseline["mean_ms"]),
            float(baseline["coefficient_of_variation"]),
            samples,
            random.Random(seed + index),
        )
        p95 = percentile(values, 0.95)
        models[name] = {
            "inference_mode": baseline["inference_mode"],
            "source": baseline["source"],
            "samples": samples,
            "mean_ms": round(fmean(values), 3),
            "std_ms": round(pstdev(values), 3),
            "p50_ms": round(percentile(values, 0.50), 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(percentile(values, 0.99), 3),
            "recommended_delay_steps": math.ceil(p95 * control_hz / 1000),
        }
    return {
        "schema_version": 1,
        "kind": "synthetic_prediction_latency",
        "warning": "Simulation for inference planning; replace with target-hardware measurements.",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "control_hz": control_hz,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--control-hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")

    baselines = json.loads(args.baselines.read_text())
    report = build_report(baselines, args.samples, args.seed, args.control_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(report['models'])} model profiles to {args.output}")


if __name__ == "__main__":
    main()
