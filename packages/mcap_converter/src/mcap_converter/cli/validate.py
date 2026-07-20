#!/usr/bin/env python3
"""
Validate a Converted LeRobot Dataset

Loads the dataset through LeRobotDataset and displays basic information to verify
successful conversion. By default, also prints raw per-frame data (first 3 episodes, 5
frames each, taken from the MIDDLE of each episode — the start of an episode is usually
motionless, so it's a poor spot-check window; override with
--print-episodes/--print-frames, or pass --print-episodes 0 to skip it) straight from
the parquet files — reading directly, not through LeRobotDataset, so it's independent of
any training-time transform and shows exactly what mcap_converter wrote. This merges what
used to be a separate `dataset-inspect` command into the one tool, so there's a single
CLI for "does this dataset load" and "show me the actual numbers".
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print("[ERROR] Error: Please install lerobot first")
    print("Run: pip install lerobot")
    print(f"Details: {e}")
    import sys

    sys.exit(1)

console = Console()


def test_dataset(repo_id: str, root: str) -> bool:
    """Load the dataset and do a basic read spot-check. Returns True on success."""
    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=root)

        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column(style="bold")
        info_table.add_column()
        info_table.add_row("Repo ID", repo_id)
        info_table.add_row("Root", root)
        info_table.add_row("Episodes", str(dataset.num_episodes))
        info_table.add_row("Frames", str(dataset.num_frames))
        info_table.add_row("FPS", str(dataset.fps))
        info_table.add_row("Robot type", str(dataset.meta.robot_type))

        feat_table = Table(
            title="Features", title_style="bold", title_justify="left", padding=(0, 1)
        )
        feat_table.add_column("Name")
        feat_table.add_column("Dtype", justify="center")
        feat_table.add_column("Shape", justify="right")
        for feat_name, feat_info in dataset.features.items():
            feat_table.add_row(
                feat_name, str(feat_info.get("dtype", "N/A")), str(feat_info.get("shape", "N/A"))
            )

        if len(dataset) > 0:
            n_read = min(10, len(dataset))
            for i in range(n_read):
                _ = dataset[i]
            read_note = f"[dim]Read {n_read} frame(s) OK.[/dim]"
        else:
            read_note = "[yellow]Dataset is empty — nothing to read.[/yellow]"

        stats_note = ""
        if getattr(dataset.meta, "stats", None):
            stats_note = (
                f"\n[dim]Stats available for {len(dataset.meta.stats)} feature(s) "
                f"— see meta/stats.json for full values.[/dim]"
            )

        body = Group(info_table, "", Padding(feat_table, (0, 0, 0, 2)), "", read_note + stats_note)
        console.print(Panel(
            body,
            title="[bold green]Dataset Load Check — PASSED",
            border_style="green",
            padding=(1, 2),
        ))
        return True

    except Exception as e:
        console.print(Panel(
            str(e), title="[bold red]Dataset Load Check — FAILED", border_style="red", padding=(1, 2),
        ))
        import traceback

        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Raw per-frame data printing (merged in from the former `dataset-inspect`
# command). Reads parquet directly — independent of LeRobotDataset/training-time
# transforms, so it shows exactly what mcap_converter wrote to disk.
# ---------------------------------------------------------------------------

_SCALAR_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _load_info(dataset_root: Path) -> Dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found — is this a converted LeRobot v3.0 dataset?")
    return json.loads(info_path.read_text())


def _feature_names(info: Dict[str, Any], column: str) -> Optional[List[str]]:
    """Per-dim labels for a vector column (e.g. observation.state -> ['left_x', ...]),
    from meta/info.json's own feature schema — the exact same names writer.py wrote."""
    feature = info.get("features", {}).get(column)
    if not feature:
        return None
    return feature.get("names")


def _load_episode_df(dataset_root: Path, episode_idx: int, columns: Optional[List[str]]):
    import pandas as pd

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=columns)
        sub = df[df["episode_index"] == episode_idx]
        if len(sub):
            frames.append(sub)
    if not frames:
        return None
    return pd.concat(frames).sort_values("frame_index").reset_index(drop=True)


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_vector(name: str, vec: np.ndarray, labels: Optional[List[str]]) -> str:
    lines = [f"  {name}:"]
    if labels and len(labels) == len(vec):
        width = max(len(l) for l in labels)
        for label, v in zip(labels, vec):
            lines.append(f"    {label.ljust(width)} = {v: .6f}")
    else:
        for i, v in enumerate(vec):
            lines.append(f"    [{i:2d}] = {v: .6f}")
    return "\n".join(lines)


def _print_frame(row, df_columns: List[str], info: Dict[str, Any], no_labels: bool) -> None:
    """Print one frame's data. observation.state and action, when both present, are
    shown side by side in one table (dim-for-dim) — the common case of spot-checking
    whether action[t] tracks the next frame's observation is much easier to eyeball
    that way than as two separate vertical blocks. Any other vector column falls back
    to a plain labeled list."""
    vector_cols = [
        c for c in df_columns
        if c not in _SCALAR_COLUMNS and isinstance(row[c], (list, np.ndarray))
    ]

    if "observation.state" in vector_cols and "action" in vector_cols:
        state = np.asarray(row["observation.state"], dtype=float)
        action = np.asarray(row["action"], dtype=float)
        state_labels = None if no_labels else _feature_names(info, "observation.state")
        action_labels = None if no_labels else _feature_names(info, "action")

        table = Table(box=None, padding=(0, 2), show_edge=False)
        table.add_column("#", style="dim", justify="right")
        table.add_column("observation.state")
        table.add_column("action")
        for i in range(max(len(state), len(action))):
            s_label = state_labels[i] if state_labels and i < len(state_labels) else str(i)
            a_label = action_labels[i] if action_labels and i < len(action_labels) else str(i)
            s_val = f"{s_label} = {state[i]: .6f}" if i < len(state) else ""
            a_val = f"{a_label} = {action[i]: .6f}" if i < len(action) else ""
            table.add_row(str(i), s_val, a_val)
        console.print(table)
        vector_cols = [c for c in vector_cols if c not in ("observation.state", "action")]

    for col in vector_cols:
        labels = None if no_labels else _feature_names(info, col)
        print(_format_vector(col, np.asarray(row[col], dtype=float), labels))

    for col in df_columns:
        if col in _SCALAR_COLUMNS or col in vector_cols or col in ("observation.state", "action"):
            continue
        print(f"  {col} = {_format_scalar(row[col])}")


def print_raw_frames(
    root: str,
    n_episodes: int,
    n_frames: int,
    columns: Optional[str] = None,
    no_labels: bool = False,
) -> bool:
    """Print raw per-frame data for the first n_episodes episodes, n_frames frames each,
    taken from the MIDDLE of each episode (the start is usually motionless — a poor
    window for spot-checking real values). Returns True on success, False if the dataset
    couldn't be read this way."""
    dataset_root = Path(root)
    try:
        info = _load_info(dataset_root)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return False

    total_episodes = info.get("total_episodes", 0)
    n_episodes = min(n_episodes, total_episodes)
    if n_episodes == 0:
        console.print(f"[yellow]total_episodes=0 in {dataset_root} — nothing to print.[/yellow]")
        return False

    vector_columns = [col for col, feat in info.get("features", {}).items() if feat.get("dtype") != "video"]
    column_list = columns.split(",") if columns else None

    console.print(Panel(
        f"episodes: {total_episodes} total, showing {n_episodes}\n"
        f"frames per episode: {n_frames} (from the middle of the episode)\n"
        f"vector features: {', '.join(vector_columns)}",
        title="Raw Frame Data", border_style="cyan", padding=(1, 2),
    ))

    for ep in range(n_episodes):
        df = _load_episode_df(dataset_root, ep, column_list)
        if df is None:
            console.print(f"[yellow]Episode {ep}: no frames found[/yellow]\n")
            continue

        total_frames = len(df)
        ep_frames = min(n_frames, total_frames)
        start = max(0, (total_frames - ep_frames) // 2)

        console.print(
            f"\n[bold]Episode {ep}[/bold] [dim]({total_frames} frames total, "
            f"showing rows {start}..{start + ep_frames - 1})[/dim]"
        )
        for offset in range(ep_frames):
            i = start + offset
            row = df.iloc[i]
            header_bits = [
                f"{col}={_format_scalar(row[col])}"
                for col in ("frame_index", "episode_index", "index", "task_index", "timestamp")
                if col in df.columns
            ]
            console.print(f"[dim]-- row {i} ({', '.join(header_bits)}) --[/dim]")
            _print_frame(row, list(df.columns), info, no_labels)

    return True


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a converted LeRobot dataset by loading and reading frames, "
                    "and optionally print raw per-frame data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # prints raw per-frame data for the first 3 episodes, 5 frames each, by default
  dataset-valid --root /tmp/test-dataset
  dataset-valid --root /tmp/test-dataset --repo-id anvil_robot/my_dataset

  # customize how much raw data is printed
  dataset-valid --root /tmp/test-dataset --print-episodes 5 --print-frames 10

  # skip raw-frame printing entirely, just run the load/read checks
  dataset-valid --root /tmp/test-dataset --print-episodes 0
""",
    )
    parser.add_argument(
        "--repo-id", type=str, default="anvil_robot/manipulation_v1",
        help="dataset repository ID (default: anvil_robot/manipulation_v1)",
    )
    parser.add_argument(
        "--root", type=str, default="output_dataset",
        help="dataset root directory (default: output_dataset)",
    )
    parser.add_argument(
        "--print-episodes", type=int, default=3,
        help="also print raw per-frame data (read directly from parquet, independent of "
             "any training-time transform) for the first N episodes (default: 3; capped "
             "at the dataset's actual total_episodes via min() if fewer exist). Pass 0 to "
             "skip this entirely and just run the existing load/read checks.",
    )
    parser.add_argument(
        "--print-frames", type=int, default=5,
        help="number of frames to print per episode when --print-episodes > 0 (default: 5; "
             "capped at each episode's actual frame count via min() if fewer exist), taken "
             "from the MIDDLE of the episode — the start is usually motionless.",
    )
    parser.add_argument(
        "--columns", type=str, default=None,
        help="comma-separated column names to print (default: all non-image columns "
             "actually present in the parquet). Only used with --print-episodes.",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="don't label vector components with their meta/info.json feature names "
             "(e.g. 'left_x') — print raw index positions only. Only used with --print-episodes.",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    parsed = parse_args(args)

    # Check if directory exists
    root_path = Path(parsed.root)
    if not root_path.exists():
        console.print(f"[red]Directory not found: {parsed.root}[/red]")
        console.print("Run mcap-convert first to create a dataset.")
        exit(1)

    # Run test
    success = test_dataset(parsed.repo_id, parsed.root)

    if parsed.print_episodes > 0:
        raw_ok = print_raw_frames(
            parsed.root, parsed.print_episodes, parsed.print_frames, parsed.columns, parsed.no_labels,
        )
        success = success and raw_ok

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
