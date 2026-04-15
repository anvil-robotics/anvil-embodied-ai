"""Matplotlib plots for evaluation results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .metrics import EpisodeMetrics


def reorder_joint_names(joint_names: list[str]) -> list[str]:
    """Reorder joints such that finger_joint1 is last (if present)."""
    # Look for 'finger_joint1' or joints ending with it
    finger_joints = [jn for jn in joint_names if "finger_joint1" in jn]
    other_joints = [jn for jn in joint_names if jn not in finger_joints]
    
    # Sort finger joints to handle multi-arm (left_finger_joint1, right_finger_joint1)
    return other_joints + sorted(finger_joints)


def plot_episode_joints(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    metrics: EpisodeMetrics,
    output_path: Path,
) -> None:
    """Plot predicted vs ground-truth joint trajectories for one episode.

    Creates a grid of subplots, one per joint.
    """
    import matplotlib.pyplot as plt

    # 1. Determine reordered indices
    new_names = reorder_joint_names(joint_names)
    idx_map = [joint_names.index(name) for name in new_names]

    n_joints = len(new_names)
    ncols = min(4, n_joints)
    nrows = math.ceil(n_joints / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(
        f"Episode {metrics.episode_idx} [{metrics.split_label}] — MAE: {metrics.mae:.4f}",
        fontsize=14,
    )

    frames = np.arange(predicted.shape[0])

    for j, name in enumerate(new_names):
        orig_idx = idx_map[j]
        row, col = divmod(j, ncols)
        ax = axes[row][col]
        ax.plot(frames, ground_truth[:, orig_idx], "b-", linewidth=1.0, label="GT")
        ax.plot(frames, predicted[:, orig_idx], "r--", linewidth=1.0, label="Pred")
        joint_mae = metrics.per_joint_mae.get(name, 0.0)
        ax.set_title(f"{name} (MAE: {joint_mae:.4f})", fontsize=9)
        ax.set_xlabel("frame", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(n_joints, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary_box_plot(
    all_metrics: list[EpisodeMetrics],
    joint_names: list[str],
    output_path: Path,
) -> None:
    """Plot per-joint MAE summary box plot, grouped by split."""
    import matplotlib.pyplot as plt

    # Group by split
    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    split_names = sorted(by_split.keys())
    n_splits = len(split_names)
    
    # Reorder joint names for plotting
    ordered_joint_names = reorder_joint_names(joint_names)
    n_joints = len(ordered_joint_names)

    if n_splits == 0 or n_joints == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_joints * 1.5), 6))

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    
    # Calculate positions for grouped box plots
    # width of one group of boxes
    group_width = 0.8
    # width of one box
    box_width = group_width / n_splits
    
    x = np.arange(n_joints)

    for i, split_name in enumerate(split_names):
        metrics_list = by_split[split_name]
        
        # Prepare data for this split across all joints
        split_data = []
        for jn in ordered_joint_names:
            vals = [m.per_joint_mae[jn] for m in metrics_list]
            split_data.append(vals)
            
        # Offset for this split
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
        
        # Color the boxes
        color = colors[i % len(colors)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Style medians
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # Add dummy element for legend
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
