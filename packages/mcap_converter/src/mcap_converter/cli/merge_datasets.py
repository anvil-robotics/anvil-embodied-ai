"""Merge multiple LeRobot datasets into one.

Usage:
    uv run merge-datasets <path1> <path2> [path3 ...] --output <output_path> [options]

Examples:
    uv run merge-datasets data/datasets/ds-a data/datasets/ds-b --output data/datasets/ds-merged
    uv run merge-datasets data/datasets/ds-a data/datasets/ds-b data/datasets/ds-c \\
        --output data/datasets/ds-merged \\
        --remove-features observation.velocity,observation.effort
"""

import argparse
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge-datasets",
        description="Merge multiple LeRobot datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
feature mismatch:
  Datasets must share the same feature schema to be merged. If one dataset was
  recorded with extra sensors (e.g. velocity + effort) and another was not, use
  --remove-features to strip those features before merging. The tool creates a
  trimmed copy (<path>-trimmed) on disk and reuses it on subsequent runs.

  Example — ds-a has velocity+effort, ds-b does not:
    merge-datasets data/datasets/ds-a data/datasets/ds-b \\
        --output data/datasets/ds-merged \\
        --remove-features observation.velocity,observation.effort
        """,
    )
    parser.add_argument("datasets", nargs="+", metavar="PATH", help="Paths to datasets to merge (at least 2)")
    parser.add_argument("--output", required=True, metavar="PATH", help="Output path for the merged dataset")
    parser.add_argument(
        "--remove-features",
        default="",
        metavar="F1,F2",
        help="Comma-separated features to strip from any dataset that has them before merging",
    )
    return parser


def _load_and_trim(root: Path, remove_features: list[str]) -> LeRobotDataset:
    repo_id = root.name
    ds = LeRobotDataset(repo_id, root=root)
    print(f"  {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

    to_remove = [f for f in remove_features if f in ds.meta.features]
    if not to_remove:
        return ds

    trimmed_id = f"{repo_id}-trimmed"
    trimmed_dir = root.parent / trimmed_id
    if trimmed_dir.exists():
        print(f"  trimmed copy already exists, loading {trimmed_id} ...")
        return LeRobotDataset(trimmed_id, root=trimmed_dir)

    print(f"  removing {to_remove} ...")
    return modify_features(
        ds,
        remove_features=to_remove,
        repo_id=trimmed_id,
        output_dir=trimmed_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()
    remove_features: list[str] = [f for f in args.remove_features.split(",") if f]

    if len(args.datasets) < 2:
        print("Error: at least 2 datasets are required.", file=sys.stderr)
        sys.exit(1)

    datasets: list[LeRobotDataset] = []
    for arg in args.datasets:
        root = Path(arg).resolve()
        print(f"Loading {root} ...")
        datasets.append(_load_and_trim(root, remove_features))

    output_dir = Path(args.output).resolve()
    print(f"\nMerging {len(datasets)} datasets into {output_dir} ...")
    merged = merge_datasets(
        datasets,
        output_repo_id=output_dir.name,
        output_dir=output_dir,
    )

    print(f"\nDone! Merged dataset at: {output_dir}")
    print(f"  Total episodes : {merged.meta.total_episodes}")
    print(f"  Total frames   : {merged.meta.total_frames}")
    print(f"  Features       : {list(merged.meta.features.keys())}")


if __name__ == "__main__":
    main()
