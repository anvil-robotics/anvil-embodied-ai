#!/usr/bin/env python3
"""Generate inference monitor report from a saved CSV.

Usage:
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv -o /tmp/report.png

The script auto-detects use_delta_actions and joint_names from metadata comment lines
written by inference_monitor_node at the top of the CSV:
    # use_delta_actions: true
    # joint_names: right_joint1,right_joint2,...

When use_delta_actions is true, a second row of subplots is shown for delta-scale signals.
Old CSVs without metadata are handled in backward-compatible mode (single row, no delta block).
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_metadata(csv_path: Path) -> tuple[bool, list[str]]:
    """Read leading comment lines to extract use_delta_actions and joint_names."""
    use_delta = False
    joint_names: list[str] = []
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if line.startswith("use_delta_actions:"):
                val = line.split(":", 1)[1].strip().lower()
                use_delta = val == "true"
            elif line.startswith("joint_names:"):
                raw = line.split(":", 1)[1].strip()
                joint_names = [n.strip() for n in raw.split(",") if n.strip()]
    return use_delta, joint_names


def plot_from_csv(csv_path: Path, output_path: Path) -> None:
    use_delta_actions, joint_names = _parse_metadata(csv_path)

    # Read CSV skipping comment lines
    rows = []
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                reader = csv.DictReader([line] + [l for l in f])
                rows = list(reader)
                break

    if len(rows) < 2:
        print(f"ERROR: too few rows ({len(rows)}) in {csv_path}", file=sys.stderr)
        sys.exit(1)

    def _extract(prefix: str) -> np.ndarray | None:
        cols = sorted(
            [k for k in rows[0].keys() if k.startswith(prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        if not cols:
            return None
        return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)

    obs = _extract("obs_state_")
    raw = _extract("raw_output_")
    cmd = _extract("control_cmd_")
    delta_cmd = _extract("delta_cmd_")  # pre-computed in monitor node; may be absent in old CSVs

    if obs is None or cmd is None:
        print("ERROR: missing obs_state or control_cmd columns", file=sys.stderr)
        sys.exit(1)

    # Fallback: compute delta_cmd from obs and cmd if not in CSV
    if delta_cmd is None:
        delta_cmd = cmd - obs[:, :cmd.shape[1]]

    n_joints = obs.shape[1]
    frames = np.arange(len(rows))

    # Joint name labels: use metadata names, fall back to indices
    def _joint_label(j: int) -> str:
        if j < len(joint_names):
            return joint_names[j]
        return f"joint[{j}]"

    ncols = min(4, n_joints)
    nrows_abs = math.ceil(n_joints / ncols)
    nrows_delta = math.ceil(n_joints / ncols) if use_delta_actions else 0
    total_rows = nrows_abs + nrows_delta

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(4 * ncols, 3 * total_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Inference Monitor — {csv_path.name}  ({len(rows)} steps, {n_joints} DOF)",
        fontsize=12,
    )

    for j in range(n_joints):
        abs_row = j // ncols
        col = j % ncols

        # ── Top block: absolute signals ──
        ax = axes[abs_row][col]
        ax.plot(frames, obs[:, j], color="steelblue", linewidth=0.8, label="obs.state")
        ax.plot(frames, cmd[:, j], color="forestgreen", linewidth=0.8, label="control cmd")
        ax.set_title(_joint_label(j), fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        # ── Bottom block: delta signals (only when use_delta_actions) ──
        if use_delta_actions:
            delta_row = nrows_abs + (j // ncols)
            ax_d = axes[delta_row][col]
            if raw is not None:
                ax_d.plot(frames, raw[:, j], color="darkorange", linewidth=0.8,
                          linestyle=":", label="raw output (delta)")
            ax_d.plot(frames, delta_cmd[:, j], color="crimson", linewidth=0.8,
                      linestyle="--", label="delta cmd")
            ax_d.set_title(f"{_joint_label(j)} [delta]", fontsize=8)
            ax_d.set_xlabel("step", fontsize=7)
            ax_d.set_ylabel("delta [rad]", fontsize=7)
            ax_d.tick_params(labelsize=6)
            if j == 0:
                ax_d.legend(fontsize=6, loc="upper right")

    # Hide unused subplots
    for block_start, nrows_block in [(0, nrows_abs), (nrows_abs, nrows_delta)]:
        for j in range(n_joints, nrows_block * ncols):
            r = block_start + j // ncols
            c = j % ncols
            if r < total_rows:
                axes[r][c].set_visible(False)

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
