#!/usr/bin/env python3
"""Plot per-tick predictions from the inference node.

Each row in the CSV is one inference tick: the wall-clock timestamp, the raw
policy targets (pre-clamp) for the 7 left-arm joints + 1 gripper, and the
current robot state at the same instant. This produces an 8-subplot figure
overlaying target vs current for each joint.

Usage:
    python3 plot_predictions.py <predictions.csv> [--out predictions.png]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) if x else float("nan") for x in r] for r in reader]
    if not rows:
        sys.exit(f"no data rows in {path}")
    data = np.array(rows, dtype=np.float64)
    return header, data


def split_columns(header):
    target_cols = [i for i, h in enumerate(header) if h.startswith("target_")]
    current_cols = [i for i, h in enumerate(header) if h.startswith("current_")]
    if len(target_cols) != len(current_cols):
        sys.exit(f"target/current column count mismatch: {len(target_cols)} vs {len(current_cols)}")
    names = [header[i].removeprefix("target_") for i in target_cols]
    return target_cols, current_cols, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="predictions CSV from inference_node")
    ap.add_argument("--out", type=Path, default=None, help="output PNG path (default: <csv>.png)")
    ap.add_argument("--show", action="store_true", help="display interactively in addition to saving")
    args = ap.parse_args()

    header, data = load(args.csv)
    t_col = header.index("t")
    target_cols, current_cols, names = split_columns(header)

    t = data[:, t_col]
    t = t - t[0]

    n = len(names)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, name, ti, ci in zip(axes, names, target_cols, current_cols):
        ax.plot(t, data[:, ci], label="current", color="tab:blue", linewidth=1.2)
        ax.plot(t, data[:, ti], label="target (raw)", color="tab:orange", linewidth=1.0, alpha=0.8)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if name.endswith("grip"):
            ax.set_ylabel("m", fontsize=8)
        else:
            ax.set_ylabel("rad", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="best")
    for ax in axes[-cols:]:
        ax.set_xlabel("time (s)", fontsize=8)

    duration = t[-1] - t[0]
    fig.suptitle(
        f"{args.csv.name} — {len(t)} ticks over {duration:.1f}s "
        f"(~{len(t) / max(duration, 1e-9):.1f} Hz)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
