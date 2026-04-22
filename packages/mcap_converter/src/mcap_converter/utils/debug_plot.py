"""Debug plots for converted LeRobot datasets.

Generates per-episode observation.state vs action comparison plots to visually
verify action_from_observation_n alignment after conversion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def plot_conversion_debug(
    output_dir: str,
    n_episodes: int = 5,
    action_from_observation_n: Optional[int] = None,
) -> None:
    """Generate obs_state vs action debug plots for the first N episodes.

    Reads converted parquet data and meta/info.json from output_dir.
    Saves one PNG per episode to {output_dir}/debug_plots/.

    Args:
        output_dir: Path to the converted LeRobot dataset directory.
        n_episodes: Number of episodes to plot (default 5).
        action_from_observation_n: Frame offset used during conversion, shown
            in plot titles. If None, attempts to read from conversion_config.yaml.
    """
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import numpy as np

    root = Path(output_dir)
    plots_dir = root / "debug_plots"
    plots_dir.mkdir(exist_ok=True)

    # Load joint names from meta/info.json
    info_path = root / "meta" / "info.json"
    joint_names: list[str] = []
    fps: int = 30
    if info_path.exists():
        info = json.loads(info_path.read_text())
        obs_feature = info.get("features", {}).get("observation.state", {})
        joint_names = obs_feature.get("names", [])
        fps = info.get("fps", 30)

    # Try to read action_from_observation_n from saved conversion config
    if action_from_observation_n is None:
        config_path = root / "conversion_config.yaml"
        if config_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_path.read_text())
                action_from_observation_n = cfg.get("action_from_observation_n", 10)
            except Exception:
                action_from_observation_n = 10
        else:
            action_from_observation_n = 10

    # Collect parquet files sorted by chunk/file order
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        print(f"[debug_plot] No parquet files found in {root / 'data'}")
        return

    # Read all rows for the first n_episodes episodes
    tables = []
    for f in data_files:
        tbl = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        # Filter to episodes we care about
        import pyarrow.compute as pc
        mask = pc.less(tbl["episode_index"], n_episodes)
        filtered = tbl.filter(mask)
        if filtered.num_rows > 0:
            tables.append(filtered)

    if not tables:
        print(f"[debug_plot] No data found for first {n_episodes} episodes")
        return

    import pyarrow as pa
    combined = pa.concat_tables(tables)
    ep_indices = combined["episode_index"].to_pylist()
    frame_indices = combined["frame_index"].to_pylist()
    obs_state_rows = combined["observation.state"].to_pylist()
    action_rows = combined["action"].to_pylist()

    # Group by episode
    episodes: dict[int, dict] = {}
    for ep, fr, obs, act in zip(ep_indices, frame_indices, obs_state_rows, action_rows):
        if ep not in episodes:
            episodes[ep] = {"frames": [], "obs": [], "act": []}
        episodes[ep]["frames"].append(fr)
        episodes[ep]["obs"].append(obs)
        episodes[ep]["act"].append(act)

    offset_sec = action_from_observation_n / fps
    offset_label = f"n={action_from_observation_n} ({offset_sec*1000:.0f}ms @ {fps}fps)"

    for ep_idx in sorted(episodes.keys())[:n_episodes]:
        ep_data = episodes[ep_idx]
        # Sort by frame index
        order = sorted(range(len(ep_data["frames"])), key=lambda i: ep_data["frames"][i])
        obs_arr = np.array([ep_data["obs"][i] for i in order], dtype=np.float32)
        act_arr = np.array([ep_data["act"][i] for i in order], dtype=np.float32)

        n_joints = obs_arr.shape[1] if obs_arr.ndim == 2 else 1
        names = joint_names if len(joint_names) == n_joints else [f"joint_{i}" for i in range(n_joints)]

        ncols = min(4, n_joints)
        nrows = math.ceil(n_joints / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(
            f"Episode {ep_idx} — obs.state (blue) vs action (orange) | offset {offset_label}",
            fontsize=11,
        )

        frames = np.arange(obs_arr.shape[0])
        for j, name in enumerate(names):
            row, col = divmod(j, ncols)
            ax = axes[row][col]
            ax.plot(frames, obs_arr[:, j], color="steelblue", linewidth=1.0, label="obs.state")
            ax.plot(frames, act_arr[:, j], color="darkorange", linewidth=1.0, linestyle="--", label="action")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=7)

        for j in range(n_joints, nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        out_path = plots_dir / f"episode_{ep_idx:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[debug_plot] Saved {out_path}")
