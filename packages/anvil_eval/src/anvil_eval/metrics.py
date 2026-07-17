"""Episode-level and summary metrics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EEMetrics:
    """EE Cartesian-specific metrics for a single evaluated episode.

    All values are per-arm averages over the episode unless noted.
    Pass/fail thresholds: position < 0.02 m, orientation < 5° (0.0873 rad).
    """

    # Per-arm position error (metres), averaged over frames
    position_error_m: dict[str, float] = field(default_factory=dict)
    # Per-arm orientation geodesic error (radians), averaged over frames
    orientation_error_rad: dict[str, float] = field(default_factory=dict)
    # Per-arm gripper absolute error (metres), averaged over frames
    gripper_error_m: dict[str, float] = field(default_factory=dict)
    # Per-arm, per-step position error curves: {arm: (T,)}
    position_error_per_step: dict[str, list[float]] = field(default_factory=dict)
    # Per-arm, per-step orientation error curves: {arm: (T,)}
    orientation_error_per_step: dict[str, list[float]] = field(default_factory=dict)

    @property
    def position_pass(self) -> bool:
        """True when all arms have position error < 0.02 m."""
        return all(v < 0.02 for v in self.position_error_m.values())

    @property
    def orientation_pass(self) -> bool:
        """True when all arms have orientation error < 5° (0.0873 rad)."""
        return all(v < 0.0873 for v in self.orientation_error_rad.values())


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
    # EE Cartesian metrics — populated only for ee_abs / ee_relative action types
    ee: Optional[EEMetrics] = None


def compute_ee_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    action_names: list[str],
) -> EEMetrics:
    """Compute EE Cartesian metrics: position (m), orientation geodesic (rad), gripper (m).

    Expects action layout per arm: [x, y, z, qx, qy, qz, qw, gripper] (8 dims, quaternion).
    Arms are inferred from action_names by the prefix before the first underscore.

    Args:
        predicted:     (T, 8*n_arms) predicted actions in absolute quaternion layout
        ground_truth:  (T, 8*n_arms) ground-truth actions in absolute quaternion layout
        action_names:  list of 8*n_arms feature names, e.g. ["right_x", ..., "right_gripper"]
    """
    from anvil_shared.rotation import quats_to_matrices

    n_arms = predicted.shape[1] // 8
    # action_names must be either full-length (8*n_arms) or empty; partial lists
    # can produce mixed label namespaces across episodes and corrupt aggregation.
    names_ok = len(action_names) == 8 * n_arms
    arm_labels = []
    for arm_idx in range(n_arms):
        if names_ok:
            label = action_names[arm_idx * 8].rsplit("_", 1)[0]
        else:
            label = f"arm{arm_idx}"
        arm_labels.append(label)

    pos_error:     dict[str, float] = {}
    ori_error:     dict[str, float] = {}
    grip_error:    dict[str, float] = {}
    pos_per_step:  dict[str, list[float]] = {}
    ori_per_step:  dict[str, list[float]] = {}

    for arm_idx, label in enumerate(arm_labels):
        a0 = arm_idx * 8
        pred_xyz  = predicted[:, a0:a0+3]      # (T, 3)
        gt_xyz    = ground_truth[:, a0:a0+3]
        pred_quat = predicted[:, a0+3:a0+7]    # (T, 4) [qx, qy, qz, qw]
        gt_quat   = ground_truth[:, a0+3:a0+7]
        pred_grip = predicted[:, a0+7]
        gt_grip   = ground_truth[:, a0+7]

        # Position error: per-frame Euclidean distance
        pos_err_steps = np.linalg.norm(pred_xyz - gt_xyz, axis=1)  # (T,)
        pos_error[label] = float(np.mean(pos_err_steps))
        pos_per_step[label] = pos_err_steps.tolist()

        # Orientation error: geodesic angle via vectorised rotation matrices
        # quats_to_matrices accepts (T, 4) → (T, 3, 3); normalises internally
        R_pred = quats_to_matrices(pred_quat)                     # (T, 3, 3)
        R_gt   = quats_to_matrices(gt_quat)                       # (T, 3, 3)
        # rel[t] = R_pred[t].T @ R_gt[t]  →  batched via matmul on leading dim
        rel    = R_pred.transpose(0, 2, 1) @ R_gt                 # (T, 3, 3)
        trace  = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]      # (T,)
        ori_err_steps = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))  # (T,)
        ori_error[label] = float(np.mean(ori_err_steps))
        ori_per_step[label] = ori_err_steps.tolist()

        grip_error[label] = float(np.mean(np.abs(pred_grip - gt_grip)))

    return EEMetrics(
        position_error_m=pos_error,
        orientation_error_rad=ori_error,
        gripper_error_m=grip_error,
        position_error_per_step=pos_per_step,
        orientation_error_per_step=ori_per_step,
    )


def compute_episode_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    episode_idx: int,
    split_label: str,
    action_type: str = "joint_abs",
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

    # EE mode: generic metrics (mse/mae/…) mix metres + quaternion + gripper and are
    # meaningless.  Return NaN/empty generics and populate ee only.
    # "ee_rel" is a permanent legacy alias for "ee_relative" — some callers
    # (e.g. eval_recorder_node) pass the action_type through unnormalized, so
    # accept both here rather than assuming normalize_action_type() ran upstream.
    if action_type in ("ee_abs", "ee_relative", "ee_rel") and predicted.shape[1] % 8 == 0:
        ee_metrics = compute_ee_metrics(predicted, ground_truth, joint_names)
        return EpisodeMetrics(
            episode_idx=episode_idx,
            split_label=split_label,
            num_frames=predicted.shape[0],
            mse=float("nan"),
            mae=float("nan"),
            rmse=float("nan"),
            max_abs_error=float("nan"),
            max_abs_error_joint="",
            per_joint_mse={},
            per_joint_mae={},
            per_joint_rmse={},
            cosine_similarity=float("nan"),
            pred_smoothness_mean=float("nan"),
            pred_smoothness_std=float("nan"),
            gt_smoothness_mean=float("nan"),
            gt_smoothness_std=float("nan"),
            ee=ee_metrics,
        )

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
        ee=None,
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

        split_summary: dict = {
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

        # EE Cartesian metrics (only when present)
        ee_eps = [m for m in metrics_list if m.ee is not None]
        if ee_eps:
            arm_names = list(ee_eps[0].ee.position_error_m.keys())
            pos_thresh, ori_thresh = 0.02, 0.0873  # 2 cm, 5 degrees
            ee_summary: dict = {}
            for arm in arm_names:
                pos_vals  = [m.ee.position_error_m.get(arm, float("nan")) for m in ee_eps]
                ori_vals  = [m.ee.orientation_error_rad.get(arm, float("nan")) for m in ee_eps]
                grip_vals = [m.ee.gripper_error_m.get(arm, float("nan")) for m in ee_eps]
                ee_summary[arm] = {
                    "position_error_m_mean":      float(np.nanmean(pos_vals)),
                    "position_error_m_std":       float(np.nanstd(pos_vals)),
                    "orientation_error_deg_mean": float(np.degrees(np.nanmean(ori_vals))),
                    "orientation_error_deg_std":  float(np.degrees(np.nanstd(ori_vals))),
                    "gripper_error_m_mean":       float(np.nanmean(grip_vals)),
                    "pass_position":  bool(np.nanmean(pos_vals) < pos_thresh),
                    "pass_orientation": bool(np.degrees(np.nanmean(ori_vals)) < 5.0),
                }
            split_summary["ee"] = ee_summary

        summary[split_name] = split_summary

    return summary
