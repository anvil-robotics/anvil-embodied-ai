#!/usr/bin/env python3
"""
Validate a Converted LeRobot Dataset

Loads the dataset through LeRobotDataset and displays basic information to verify
successful conversion. By default, also prints raw per-frame data (first 3 episodes, 5
frames each — override with --print-episodes/--print-frames, or pass --print-episodes 0
to skip it) straight from the parquet files — reading directly, not through
LeRobotDataset, so it's independent of any training-time transform and shows exactly
what mcap_converter wrote. This merges what used to be a separate `dataset-inspect`
command into the one tool, so there's a single CLI for "does this dataset load" and
"show me the actual numbers".
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print("[ERROR] Error: Please install lerobot first")
    print("Run: pip install lerobot")
    print(f"Details: {e}")
    import sys

    sys.exit(1)


def test_dataset(repo_id: str, root: str):
    """Test if dataset can be loaded normally"""

    print("=" * 70)
    print("LeRobot Dataset Test")
    print("=" * 70)
    print(f"Repo ID: {repo_id}")
    print(f"Root: {root}")
    print("=" * 70)

    try:
        # Load dataset
        print("\n[1/5] Load dataset...")
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        print("[OK] Dataset loaded successfully")

        # Display basic information
        print("\n[2/5] Dataset Basic Information:")
        print(f"  - Total episodes: {dataset.num_episodes}")
        print(f"  - Total frames: {dataset.num_frames}")
        print(f"  - FPS: {dataset.fps}")
        print(f"  - Robot type: {dataset.meta.robot_type}")

        # Display features
        print("\n[3/5] Dataset Features:")
        for feat_name, feat_info in dataset.features.items():
            print(f"  - {feat_name}:")
            print(f"      dtype: {feat_info.get('dtype', 'N/A')}")
            print(f"      shape: {feat_info.get('shape', 'N/A')}")

        # Test reading first frame
        print("\n[4/5] Test reading data...")
        if len(dataset) > 0:
            frame = dataset[0]
            print("[OK] Successfully read first frame")
            print(f"  Available keys: {list(frame.keys())}")

            # Display feature shapes
            print("\n  Shape of each feature:")
            for key, value in frame.items():
                if hasattr(value, "shape"):
                    print(f"    - {key}: {value.shape}")
                elif hasattr(value, "__len__"):
                    print(f"    - {key}: len={len(value)}")
                else:
                    print(f"    - {key}: {type(value).__name__}")
        else:
            print("[WARNING] Warning: Dataset is empty")

        # Test batch reading
        print("\n[5/5] Test batch reading...")
        num_test_frames = min(10, len(dataset))
        if num_test_frames > 0:
            for i in range(num_test_frames):
                frame = dataset[i]
            print(f"[OK] Successfully read {num_test_frames} frames")

        # Statistics
        if hasattr(dataset.meta, "stats") and dataset.meta.stats:
            print("\n[Additional] Statistics:")
            for key, stats in dataset.meta.stats.items():
                if isinstance(stats, dict):
                    print(f"  - {key}:")
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, list):
                            print(f"      {stat_name}: [{len(stat_value)} values]")
                        else:
                            print(f"      {stat_name}: {stat_value}")

        print("\n" + "=" * 70)
        print("[OK] All tests passed! Dataset can be used normally")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Raw per-frame data printing (merged in from the former `dataset-inspect`
# command). Reads parquet directly — independent of LeRobotDataset/training-time
# transforms, so it shows exactly what mcap_converter wrote to disk.
# ---------------------------------------------------------------------------


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


def print_raw_frames(
    root: str,
    n_episodes: int,
    n_frames: int,
    columns: Optional[str] = None,
    no_labels: bool = False,
) -> bool:
    """Print raw per-frame data for the first n_episodes episodes, first n_frames frames
    each. Returns True on success, False if the dataset couldn't be read this way."""
    dataset_root = Path(root)
    try:
        info = _load_info(dataset_root)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    total_episodes = info.get("total_episodes", 0)
    n_episodes = min(n_episodes, total_episodes)
    if n_episodes == 0:
        print(f"[dataset-valid] total_episodes=0 in {dataset_root} — nothing to print.")
        return False

    vector_columns = [col for col, feat in info.get("features", {}).items() if feat.get("dtype") != "video"]
    scalar_columns = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    column_list = columns.split(",") if columns else None

    print("\n" + "=" * 70)
    print("Raw Frame Data")
    print("=" * 70)
    print(f"total_episodes={total_episodes}, showing first {n_episodes} episode(s), "
          f"first {n_frames} frame(s) each")
    print(f"vector features found: {vector_columns}")
    print()

    for ep in range(n_episodes):
        df = _load_episode_df(dataset_root, ep, column_list)
        if df is None:
            print(f"=== Episode {ep}: no frames found ===\n")
            continue

        print(f"=== Episode {ep} ({len(df)} total frames) ===")
        ep_frames = min(n_frames, len(df))
        for i in range(ep_frames):
            row = df.iloc[i]
            header_bits = []
            for col in ("frame_index", "episode_index", "index", "task_index", "timestamp"):
                if col in df.columns:
                    header_bits.append(f"{col}={_format_scalar(row[col])}")
            print(f"--- frame {i} ({', '.join(header_bits)}) ---")

            for col in df.columns:
                if col in scalar_columns:
                    continue
                value = row[col]
                if isinstance(value, (list, np.ndarray)):
                    labels = None if no_labels else _feature_names(info, col)
                    print(_format_vector(col, np.asarray(value, dtype=float), labels))
                else:
                    print(f"  {col} = {_format_scalar(value)}")
            print()
        print()

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
             "capped at each episode's actual frame count via min() if fewer exist)",
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
        print(f"[ERROR] Directory not found: {parsed.root}")
        print("Run mcap-convert first to create a dataset.")
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
