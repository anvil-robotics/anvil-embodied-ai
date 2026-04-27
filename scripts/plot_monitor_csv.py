#!/usr/bin/env python3
"""Generate inference monitor report from a saved CSV.

Usage:
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv
    python scripts/plot_monitor_csv.py /tmp/monitor_smoke_test/monitor/inference_data.csv -o /tmp/report.png

The script auto-detects action_type and joint_names from metadata comment lines
written by inference_monitor_node at the top of the CSV:
    # action_type: delta_obs_t
    # joint_names: right_joint1,right_joint2,...

Old CSVs with `# use_delta_actions: true` are handled in backward-compatible mode
(treated as delta_obs_t).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _parse_metadata(csv_path: Path) -> tuple[str, list[str]]:
    """Read leading comment lines to extract action_type and joint_names."""
    action_type = "absolute"
    joint_names: list[str] = []
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if line.startswith("action_type:"):
                action_type = line.split(":", 1)[1].strip()
            elif line.startswith("use_delta_actions:"):
                # Legacy metadata: use_delta_actions: true → delta_obs_t
                if line.split(":", 1)[1].strip().lower() == "true" and action_type == "absolute":
                    action_type = "delta_obs_t"
            elif line.startswith("joint_names:"):
                raw = line.split(":", 1)[1].strip()
                joint_names = [n.strip() for n in raw.split(",") if n.strip()]
    return action_type, joint_names


def plot_from_csv(csv_path: Path, output_path: Path) -> None:
    action_type, joint_names = _parse_metadata(csv_path)

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

    if obs is None or cmd is None:
        print("ERROR: missing obs_state or control_cmd columns", file=sys.stderr)
        sys.exit(1)

    # anvil_eval.plotting provides the shared rendering function
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _repo_root = _Path(__file__).resolve().parents[1]
        _anvil_eval_src = str(_repo_root / "packages" / "anvil_eval" / "src")
        if _anvil_eval_src not in _sys.path:
            _sys.path.insert(0, _anvil_eval_src)
        from anvil_eval.plotting import plot_monitor_signals
    except ImportError as e:
        print(f"ERROR: cannot import anvil_eval.plotting: {e}", file=sys.stderr)
        sys.exit(1)

    n_joints = obs.shape[1]
    title = f"Inference Monitor — {csv_path.name}  ({len(rows)} steps, {n_joints} DOF)"
    plot_monitor_signals(obs, cmd, raw, joint_names, title, output_path, action_type=action_type)
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
