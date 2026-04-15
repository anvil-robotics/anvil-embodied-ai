"""Episode-level and summary metrics computation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluated episode."""

    episode_idx: int
    split_label: str
    num_frames: int
    mse: float
    mae: float
    rmse: float
    max_abs_error: float
    max_abs_error_joint: str
    per_joint_mse: dict[str, float]
    per_joint_mae: dict[str, float]
    per_joint_rmse: dict[str, float]
    cosine_similarity: float
    pred_smoothness_mean: float
    pred_smoothness_std: float
    gt_smoothness_mean: float
    gt_smoothness_std: float


def compute_episode_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    episode_idx: int,
    split_label: str,
) -> EpisodeMetrics:
    """Compute all metrics for a single episode.

    Args:
        predicted: (T, D) predicted actions
        ground_truth: (T, D) ground-truth actions
        joint_names: list of D joint names
        episode_idx: episode index
        split_label: split label (train/val/test/manual)
    """
    error = predicted - ground_truth  # (T, D)

    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    abs_error = np.abs(error)
    max_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    max_abs_error = float(abs_error[max_idx])
    max_abs_error_joint = joint_names[max_idx[1]] if len(joint_names) > max_idx[1] else f"joint_{max_idx[1]}"

    per_joint_mse = {
        name: float(np.mean(error[:, j] ** 2)) for j, name in enumerate(joint_names)
    }
    per_joint_mae = {
        name: float(np.mean(np.abs(error[:, j]))) for j, name in enumerate(joint_names)
    }
    per_joint_rmse = {
        name: float(np.sqrt(per_joint_mse[name])) for name in joint_names
    }

    # Cosine similarity (per-frame, then average)
    dot = np.sum(predicted * ground_truth, axis=1)
    norm_p = np.linalg.norm(predicted, axis=1)
    norm_g = np.linalg.norm(ground_truth, axis=1)
    denom = norm_p * norm_g
    cos_sim = np.where(denom > 1e-8, dot / denom, 0.0)
    cosine_similarity = float(np.mean(cos_sim))

    # Smoothness: L2 norm of consecutive action deltas
    if predicted.shape[0] > 1:
        pred_deltas = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        gt_deltas = np.linalg.norm(np.diff(ground_truth, axis=0), axis=1)
        pred_smooth_mean = float(np.mean(pred_deltas))
        pred_smooth_std = float(np.std(pred_deltas))
        gt_smooth_mean = float(np.mean(gt_deltas))
        gt_smooth_std = float(np.std(gt_deltas))
    else:
        pred_smooth_mean = pred_smooth_std = gt_smooth_mean = gt_smooth_std = 0.0

    return EpisodeMetrics(
        episode_idx=episode_idx,
        split_label=split_label,
        num_frames=predicted.shape[0],
        mse=mse,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs_error,
        max_abs_error_joint=max_abs_error_joint,
        per_joint_mse=per_joint_mse,
        per_joint_mae=per_joint_mae,
        per_joint_rmse=per_joint_rmse,
        cosine_similarity=cosine_similarity,
        pred_smoothness_mean=pred_smooth_mean,
        pred_smoothness_std=pred_smooth_std,
        gt_smoothness_mean=gt_smooth_mean,
        gt_smoothness_std=gt_smooth_std,
    )


def compute_summary_metrics(episode_metrics: list[EpisodeMetrics]) -> dict:
    """Aggregate metrics across episodes, grouped by split."""
    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in episode_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    summary: dict = {}
    for split_name, metrics_list in by_split.items():
        n = len(metrics_list)
        mses = [m.mse for m in metrics_list]
        maes = [m.mae for m in metrics_list]
        rmses = [m.rmse for m in metrics_list]
        cos_sims = [m.cosine_similarity for m in metrics_list]

        # Per-joint aggregation
        joint_names = list(metrics_list[0].per_joint_mae.keys())
        per_joint_mae_mean = {}
        per_joint_mae_std = {}
        for jn in joint_names:
            vals = [m.per_joint_mae[jn] for m in metrics_list]
            per_joint_mae_mean[jn] = float(np.mean(vals))
            per_joint_mae_std[jn] = float(np.std(vals))

        summary[split_name] = {
            "num_episodes": n,
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "cosine_similarity_mean": float(np.mean(cos_sims)),
            "cosine_similarity_std": float(np.std(cos_sims)),
            "per_joint_mae_mean": per_joint_mae_mean,
            "per_joint_mae_std": per_joint_mae_std,
        }

    return summary
