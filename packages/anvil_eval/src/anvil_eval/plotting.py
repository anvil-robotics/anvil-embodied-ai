"""Matplotlib plots for evaluation results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .metrics import EpisodeMetrics


def reorder_joint_names(joint_names: list[str]) -> list[str]:
    """Reorder joints such that finger_joint1 is last (if present)."""
    finger_joints = [jn for jn in joint_names if "finger_joint1" in jn]
    other_joints = [jn for jn in joint_names if jn not in finger_joints]
    return other_joints + sorted(finger_joints)


def plot_episode_joints(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    metrics: EpisodeMetrics,
    output_path: Path,
    raw_output: np.ndarray | None = None,
    obs_states: np.ndarray | None = None,
    action_type: str = "absolute",
    raw_ground_truth: np.ndarray | None = None,
) -> None:
    """Plot predicted vs ground-truth joint trajectories for one episode.

    Layout (per joint column):
    - Top block (absolute scale): GT, Pred, obs_state
    - Bottom block (delta scale, when action_type is delta and raw_ground_truth provided):
      raw model output and ΔGT taken directly from raw_ground_truth (pre-computed by evaluator)
    """
    import matplotlib.pyplot as plt

    new_names = reorder_joint_names(joint_names)
    idx_map = [joint_names.index(name) for name in new_names]
    n_joints = len(new_names)
    ncols = min(4, n_joints)
    nrows_abs = math.ceil(n_joints / ncols)

    show_delta = action_type in ("delta_obs_t", "delta_sequential") and raw_ground_truth is not None
    nrows_delta = math.ceil(n_joints / ncols) if show_delta else 0
    total_rows = nrows_abs + nrows_delta

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(4 * ncols, 3 * total_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Episode {metrics.episode_idx} [{metrics.split_label}] — MAE: {metrics.mae:.4f}",
        fontsize=14,
    )

    frames = np.arange(predicted.shape[0])

    for j, name in enumerate(new_names):
        orig_idx = idx_map[j]
        abs_row = j // ncols
        col = j % ncols

        # ── Top block: absolute signals ──
        ax = axes[abs_row][col]
        ax.plot(frames, ground_truth[:, orig_idx], "b-", linewidth=1.0, label="GT")
        ax.plot(frames, predicted[:, orig_idx], "r--", linewidth=1.0, label="Pred")
        if obs_states is not None:
            ax.plot(frames, obs_states[:, orig_idx], color="purple",
                    linewidth=0.9, alpha=0.7, label="Obs")
        joint_mae = metrics.per_joint_mae.get(name, 0.0)
        ax.set_title(f"{name} (MAE: {joint_mae:.4f})", fontsize=9)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("rad", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

        # ── Bottom block: delta signals ──
        if show_delta:
            delta_row = nrows_abs + (j // ncols)
            ax_d = axes[delta_row][col]
            if raw_output is not None and orig_idx < raw_output.shape[1]:
                ax_d.plot(frames, raw_output[:, orig_idx], color="darkorange",
                          linewidth=0.8, linestyle=":", label="Raw (delta)")
            if orig_idx < raw_ground_truth.shape[1]:
                ax_d.plot(frames, raw_ground_truth[:, orig_idx], color="green",
                          linewidth=0.8, linestyle="--", label="ΔGT")
            ax_d.set_title(f"{name} [delta]", fontsize=9)
            ax_d.set_xlabel("frame", fontsize=8)
            ax_d.set_ylabel("delta [rad]", fontsize=8)
            ax_d.tick_params(labelsize=7)
            if j == 0:
                ax_d.legend(fontsize=7)

    # Hide unused subplots in both blocks
    for block_start, nrows_block in [(0, nrows_abs), (nrows_abs, nrows_delta)]:
        for j in range(n_joints, nrows_block * ncols):
            r = block_start + j // ncols
            c = j % ncols
            if r < total_rows:
                axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monitor_signals(
    obs: np.ndarray,
    cmd: np.ndarray,
    raw_output: np.ndarray | None,
    joint_names: list[str],
    title: str,
    output_path: Path,
    action_type: str = "absolute",
    ncols: int = 4,
    dpi: int = 120,
) -> None:
    """Plot inference monitor signals (obs/cmd/raw) for one session.

    Layout (per joint column):
    - Top block: obs_state, control_cmd
    - Bottom block (when action_type is delta): raw model output, delta_cmd

    delta_cmd is computed from cmd based on action_type:
    - delta_obs_t:      delta_cmd[t] = cmd[t] - obs[t]
    - delta_sequential: delta_cmd[0] = cmd[0] - obs[0]; delta_cmd[t] = cmd[t] - cmd[t-1]
    """
    import matplotlib.pyplot as plt

    n_joints = obs.shape[1]
    frames = np.arange(obs.shape[0])

    show_delta = action_type in ("delta_obs_t", "delta_sequential") and raw_output is not None
    ncols = min(ncols, n_joints)
    nrows_abs = math.ceil(n_joints / ncols)
    nrows_delta = math.ceil(n_joints / ncols) if show_delta else 0
    total_rows = nrows_abs + nrows_delta

    # Compute delta_cmd in model-output space based on action_type
    delta_cmd: np.ndarray | None = None
    if show_delta:
        d = min(raw_output.shape[1], cmd.shape[1], obs.shape[1])
        if action_type == "delta_sequential":
            delta_cmd = np.zeros((obs.shape[0], d), dtype=np.float32)
            delta_cmd[0] = cmd[0, :d] - obs[0, :d]
            delta_cmd[1:] = np.diff(cmd[:, :d], axis=0)
        else:  # delta_obs_t
            delta_cmd = cmd[:, :d] - obs[:, :d]

    def _joint_label(j: int) -> str:
        return joint_names[j] if j < len(joint_names) else f"joint[{j}]"

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(4 * ncols, 3 * total_rows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=12)

    for j in range(n_joints):
        abs_row = j // ncols
        col = j % ncols

        ax = axes[abs_row][col]
        ax.plot(frames, obs[:, j], color="steelblue", linewidth=0.8, label="obs.state")
        ax.plot(frames, cmd[:, j], color="forestgreen", linewidth=0.8, label="control cmd")
        ax.set_title(_joint_label(j), fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        if show_delta:
            delta_row = nrows_abs + (j // ncols)
            ax_d = axes[delta_row][col]
            if raw_output is not None and j < raw_output.shape[1]:
                ax_d.plot(frames, raw_output[:, j], color="darkorange", linewidth=0.8,
                          linestyle=":", label="raw output (delta)")
            if delta_cmd is not None and j < delta_cmd.shape[1]:
                ax_d.plot(frames, delta_cmd[:, j], color="crimson", linewidth=0.8,
                          linestyle="--", label="delta cmd")
            ax_d.set_title(f"{_joint_label(j)} [delta]", fontsize=8)
            ax_d.set_xlabel("step", fontsize=7)
            ax_d.set_ylabel("delta [rad]", fontsize=7)
            ax_d.tick_params(labelsize=6)
            if j == 0:
                ax_d.legend(fontsize=6, loc="upper right")

    for block_start, nrows_block in [(0, nrows_abs), (nrows_abs, nrows_delta)]:
        for j in range(n_joints, nrows_block * ncols):
            r = block_start + j // ncols
            c = j % ncols
            if r < total_rows:
                axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_summary_box_plot(
    all_metrics: list[EpisodeMetrics],
    joint_names: list[str],
    output_path: Path,
) -> None:
    """Plot per-joint MAE summary box plot, grouped by split."""
    import matplotlib.pyplot as plt

    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    split_names = sorted(by_split.keys())
    n_splits = len(split_names)

    ordered_joint_names = reorder_joint_names(joint_names)
    n_joints = len(ordered_joint_names)

    if n_splits == 0 or n_joints == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_joints * 1.5), 6))

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    group_width = 0.8
    box_width = group_width / n_splits
    x = np.arange(n_joints)

    for i, split_name in enumerate(split_names):
        metrics_list = by_split[split_name]

        split_data = []
        for jn in ordered_joint_names:
            vals = [m.per_joint_mae[jn] for m in metrics_list]
            split_data.append(vals)

        offset = (i - n_splits / 2 + 0.5) * box_width
        pos = x + offset

        bp = ax.boxplot(
            split_data,
            positions=pos,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )

        color = colors[i % len(colors)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.plot([], [], color=color, label=split_name, linewidth=10, alpha=0.6)

    ax.set_xlabel("Joint")
    ax.set_ylabel("MAE")
    ax.set_title("Distribution of Per-Joint MAE by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_joint_names, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Split")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
