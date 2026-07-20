"""Shared reader for converted LeRobot-format datasets (mcap_converter output).

Covers the on-disk artifacts any node/script in this package needs to read from a
converted dataset: ``meta/info.json``, ``conversion_config.yaml`` (action_type /
observation encoding detection), and per-episode ``data/chunk-*/file-*.parquet``
columns (actions, observations). Read parquet directly (not through
``LeRobotDataset``/training transforms) so callers get exactly what mcap_converter
wrote on disk, independent of any training-time transform.

Does NOT read images/video (``videos/*.mp4``) — those need ``LeRobotDataset``'s
video decoding, a materially different kind of reader than the parquet-glob
pattern this module is built around; no current caller needs it.

Encoding detection deliberately uses a plain, lenient ``yaml.safe_load`` of
``conversion_config.yaml`` rather than ``mcap_converter.config.loader.ConfigLoader``:
the inference Docker image does not ship ``mcap_converter``, and every dataset
these readers target was converted with the current (v1.1) schema, so the
migration-aware loader isn't needed here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


def _ensure_anvil_shared() -> None:
    """Add packages/anvil_shared/src to sys.path so ee_transform helpers are importable.

    Mirrors ee_runtime.py's helper of the same name — called lazily, inside the
    one function that needs it, so importing this module doesn't require
    anvil_shared to already be on the path.
    """
    import os
    import sys

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    _shared_src = os.path.join(_repo_root, "packages", "anvil_shared", "src")
    if _shared_src not in sys.path:
        sys.path.insert(0, _shared_src)


def load_info(dataset_root: Path) -> dict:
    """Read ``meta/info.json``. Raises ``FileNotFoundError`` if missing."""
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_root}")
    return json.loads(info_path.read_text())


def _load_conversion_config(dataset_root: Path, logger=None) -> dict:
    """Read ``conversion_config.yaml`` (raw, lenient). Empty dict if missing."""
    cfg_path = Path(dataset_root) / "conversion_config.yaml"
    if not cfg_path.exists():
        msg = (
            f"[dataset_reader] {cfg_path} not found — using schema defaults "
            "(data_space=joint, action_encoding=absolute, observation_encoding=quaternion)"
        )
        (logger.warn if logger is not None else log.warning)(msg)
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def resolve_action_type(dataset_root: Path, logger=None) -> str:
    """Resolve action_type from conversion_config.yaml's schema-v1.1 fields.

    joint -> joint_abs; ee+absolute -> ee_abs; ee+delta -> ee_delta.
    ee+relative is reserved/unimplemented in mcap_converter, so unreachable.
    """
    cfg = _load_conversion_config(dataset_root, logger)
    data_space = cfg.get("data_space", "joint")
    action_encoding = cfg.get("action_encoding", "absolute")
    if data_space != "ee":
        return "joint_abs"
    return "ee_delta" if action_encoding == "delta" else "ee_abs"


def resolve_observation_encoding(dataset_root: Path, logger=None) -> str:
    """Resolve conversion_config.yaml's ``observation_encoding`` field (default 'quaternion')."""
    cfg = _load_conversion_config(dataset_root, logger)
    return cfg.get("observation_encoding", "quaternion")


def parse_episode_spec(spec: str, total_episodes: int) -> list[int]:
    """Parse a 0-based episode index spec into a sorted list of concrete indices.

    Ported (not imported — same reasoning as this module's avoidance of
    ``mcap_converter.config.loader.ConfigLoader``: the inference Docker image
    doesn't ship ``mcap_converter``) from
    ``mcap_converter.cli.dataset_viz.parse_episodes_spec``, whose grammar this
    matches exactly: a comma-separated list of tokens, each either a single
    index or a ``start:end`` range. Colon ranges follow Python slice
    convention — the end is EXCLUSIVE, e.g. ``"1:4"`` selects episodes 1, 2, 3
    (not 4), same as ``range(1, 4)``. An omitted start defaults to 0; an
    omitted end defaults to ``total_episodes``. Negative indices and a step
    (e.g. ``"-1"``, ``"::2"``) are deliberately NOT supported — this matches
    the only other episode-spec parsers in the repo, both of which have the
    same restriction.

    Raises ``ValueError`` (naming the offending token) on: a non-integer
    token, a range where ``start >= end``, any index or range boundary
    outside ``[0, total_episodes)``, or a token with more than one ``:``.
    Duplicate indices across tokens are silently deduplicated.
    """
    result: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.count(":") > 1:
            raise ValueError(f"invalid episode spec token: '{token}' (more than one ':')")
        if ":" in token:
            start_str, end_str = token.split(":", 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            try:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else total_episodes
            except ValueError:
                raise ValueError(f"invalid episode range token: '{token}'")
            if start < 0 or end < 0:
                raise ValueError(
                    f"range '{token}' has a negative bound — negative indices are not supported"
                )
            if start >= end:
                raise ValueError(
                    f"invalid range '{token}': start must be less than end (end is exclusive)"
                )
            if end > total_episodes:
                raise ValueError(
                    f"range '{token}' out of bounds — episodes are numbered 0 to {total_episodes - 1}"
                )
            result.update(range(start, end))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"invalid episode index token: '{token}'")
            if not (0 <= idx <= total_episodes - 1):
                raise ValueError(f"episode index {idx} out of range (0-{total_episodes - 1})")
            result.add(idx)
    return sorted(result)


def load_episode_columns(
    dataset_root: Path, episode_idx: int, columns: list[str]
) -> pd.DataFrame:
    """Read specific columns for one episode, sorted by frame_index.

    Always includes ``episode_index``/``frame_index`` (used to filter/sort)
    even if not requested. Returns an empty DataFrame if the episode has no rows.
    """
    data_dir = Path(dataset_root) / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files found under {data_dir}")

    read_columns = list(dict.fromkeys([*columns, "episode_index", "frame_index"]))
    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=read_columns)
        sub = df[df["episode_index"] == episode_idx]
        if len(sub):
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=read_columns)

    return pd.concat(frames).sort_values("frame_index").reset_index(drop=True)


def load_episode_actions(dataset_root: Path, episode_idx: int) -> np.ndarray | None:
    """Read the ``action`` column for one episode, sorted by frame_index.

    Returns ``None`` if the episode has no rows.
    """
    df = load_episode_columns(dataset_root, episode_idx, ["action"])
    if df.empty:
        return None
    return np.stack(df["action"].to_numpy()).astype(np.float64)


def load_episode_observations_quat(dataset_root: Path, episode_idx: int) -> np.ndarray:
    """Read ``observation.state`` for one episode, converted to quaternion layout.

    Quaternion-encoded datasets pass through unchanged; rot6d-encoded datasets are
    converted via ``ee_rot6d_to_quat_layout``. Any other ``observation_encoding``
    (e.g. axis_angle) raises ``ValueError`` rather than silently mishandling it —
    no current caller targets a dataset converted with one.

    Returns ``(T, 8*n_arms)``, per-arm layout ``[x, y, z, qx, qy, qz, qw, gripper]``.
    """
    df = load_episode_columns(dataset_root, episode_idx, ["observation.state"])
    if df.empty:
        raise RuntimeError(
            f"no observation.state rows found for episode {episode_idx} in {dataset_root}"
        )
    obs = np.stack(df["observation.state"].to_numpy()).astype(np.float64)

    encoding = resolve_observation_encoding(dataset_root)
    if encoding == "quaternion":
        return obs
    if encoding == "rot6d":
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_rot6d_to_quat_layout

        return ee_rot6d_to_quat_layout(obs)
    raise ValueError(
        f"load_episode_observations_quat: observation_encoding={encoding!r} is not "
        "supported (only 'quaternion' and 'rot6d' are handled)"
    )
