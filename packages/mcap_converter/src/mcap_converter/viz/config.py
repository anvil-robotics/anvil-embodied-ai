"""Small pure helper for dataset-viz: deriving a cosmetic dataset name."""

from pathlib import Path


def default_repo_id(dataset_root: Path) -> str:
    """
    Derive a cosmetic default "org/dataset" identifier from a dataset root
    path, used when the user doesn't pass --repo-id. This value is purely
    decorative (passed through to `LeRobotDataset(repo_id=...)` and shown in
    Rerun's recording name) -- it never has to correspond to anything on the
    Hub. Example:
    default_repo_id(Path("/data/datasets/my-session")) -> "local/my-session"
    """
    return f"local/{Path(dataset_root).resolve().name}"
