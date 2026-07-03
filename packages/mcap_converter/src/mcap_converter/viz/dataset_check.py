"""Lightweight, dependency-free validation that a directory looks like a
converted LeRobot dataset that dataset-viz can serve.

Deliberately avoids depending on the `lerobot` package or loading a
LeRobotDataset — this must stay a fast filesystem/JSON check, unlike the
heavier `dataset-validate` CLI.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SUPPORTED_CODEBASE_VERSIONS = ("v2.0", "v2.1", "v3.0")


@dataclass
class DatasetCheck:
    ok: bool
    codebase_version: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_dataset_root(root: Path) -> DatasetCheck:
    """
    Validate that `root` looks like a converted LeRobot dataset directory.

    Checks (in order, short-circuiting where a later check would be
    meaningless without an earlier one passing):
    1. `root` exists and is a directory.
    2. `root/meta/info.json` exists and parses as JSON.
    3. `info["codebase_version"]` is one of SUPPORTED_CODEBASE_VERSIONS.
    4. `root/data/` exists and is a directory.
    5. `root/videos/` exists and has at least one subdirectory (warning, not
       error, if missing — a dataset with no videos still has usable charts).

    Returns a DatasetCheck with `ok=True` only if there are no errors
    (warnings do not affect `ok`).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not root.is_dir():
        errors.append(f"dataset root does not exist or is not a directory: {root}")
        return DatasetCheck(ok=False, errors=errors)

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append(f"not a LeRobot dataset (missing {info_path})")
        return DatasetCheck(ok=False, errors=errors)

    try:
        info = json.loads(info_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"could not read/parse {info_path}: {exc}")
        return DatasetCheck(ok=False, errors=errors)

    if not isinstance(info, dict):
        errors.append(
            f"{info_path} does not contain a JSON object (got {type(info).__name__})"
        )
        return DatasetCheck(ok=False, errors=errors)

    codebase_version = info.get("codebase_version")
    if codebase_version not in SUPPORTED_CODEBASE_VERSIONS:
        errors.append(
            f"unsupported codebase_version {codebase_version!r} "
            f"(supported: {', '.join(SUPPORTED_CODEBASE_VERSIONS)}); "
            "v3.0 is what mcap-convert produces"
        )
        return DatasetCheck(ok=False, codebase_version=codebase_version, errors=errors)

    if not (root / "data").is_dir():
        errors.append(f"missing {root / 'data'} directory")

    videos_dir = root / "videos"
    if not videos_dir.is_dir() or not any(p.is_dir() for p in videos_dir.iterdir()):
        warnings.append("no videos found under videos/; only charts will render")

    return DatasetCheck(
        ok=not errors,
        codebase_version=codebase_version,
        errors=errors,
        warnings=warnings,
    )
