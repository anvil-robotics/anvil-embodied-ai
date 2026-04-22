#!/usr/bin/env python3
"""Generate inference monitor report from a saved CSV.

Usage:
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv -o /tmp/report.png
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_from_csv(csv_path: Path, output_path: Path) -> None:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        print(f"ERROR: too few rows ({len(rows)}) in {csv_path}", file=sys.stderr)
        sys.exit(1)

    def _extract(prefix: str) -> np.ndarray:
        cols = sorted(
            [k for k in rows[0].keys() if k.startswith(prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)

    obs = _extract("obs_state_")
    raw = _extract("raw_output_")
    cmd = _extract("control_cmd_")
    n_joints = obs.shape[1]
    frames = np.arange(len(rows))

    ncols = min(4, n_joints)
    nrows = math.ceil(n_joints / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"Inference Monitor — {csv_path.name}  ({len(rows)} steps, {n_joints} DOF)",
        fontsize=12,
    )

    groups = [
        ("obs.state",   obs, "steelblue"),
        ("raw output",  raw, "darkorange"),
        ("control cmd", cmd, "forestgreen"),
    ]
    for j in range(n_joints):
        row_ax = j // ncols
        col_ax = j % ncols
        ax = axes[row_ax][col_ax]
        for label, data, color in groups:
            ax.plot(frames, data[:, j], color=color, linewidth=0.8, label=label)
        ax.set_title(f"joint [{j}]", fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused subplots
    for row_ax in range(nrows):
        for col_ax in range(ncols):
            if row_ax * ncols + col_ax >= n_joints:
                axes[row_ax][col_ax].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot inference monitor CSV")
    parser.add_argument("csv", type=Path, help="Path to inference_data.csv")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output PNG path (default: <csv_dir>/inference_report.png)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.csv.parent / "inference_report.png"
    plot_from_csv(args.csv, output)


if __name__ == "__main__":
    main()
