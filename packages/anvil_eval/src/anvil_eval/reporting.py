"""Reporting module for evaluation metrics (JSON/CSV outputs)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import EpisodeMetrics


def _convert_for_json(obj: Any) -> Any:
    """Helper to convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def write_metrics_summary(
    all_metrics: list[EpisodeMetrics],
    output_path: Path,
) -> None:
    """Write aggregated metrics summary to a JSON file."""
    # Group by split
    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    summary: dict[str, Any] = {
        "overall": _compute_aggregate(all_metrics),
        "by_split": {},
    }

    for split_label, metrics_list in by_split.items():
        summary["by_split"][split_label] = _compute_aggregate(metrics_list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_convert_for_json(summary), f, indent=2)


def _compute_aggregate(metrics_list: list[EpisodeMetrics]) -> dict[str, Any]:
    """Compute mean metrics across a list of episode metrics."""
    if not metrics_list:
        return {}

    n = len(metrics_list)
    joint_names = list(metrics_list[0].per_joint_mae.keys())

    agg: dict[str, Any] = {
        "count": n,
        "mean_mse": sum(m.mse for m in metrics_list) / n,
        "mean_mae": sum(m.mae for m in metrics_list) / n,
        "mean_rmse": sum(m.rmse for m in metrics_list) / n,
        "mean_cosine_similarity": sum(m.cosine_similarity for m in metrics_list) / n,
        "mean_max_abs_error": sum(m.max_abs_error for m in metrics_list) / n,
        "mean_pred_smoothness_mean": sum(m.pred_smoothness_mean for m in metrics_list) / n,
        "mean_gt_smoothness_mean": sum(m.gt_smoothness_mean for m in metrics_list) / n,
        "per_joint_mae": {},
        "per_joint_mse": {},
    }

    for jn in joint_names:
        agg["per_joint_mae"][jn] = sum(m.per_joint_mae.get(jn, 0) for m in metrics_list) / n
        agg["per_joint_mse"][jn] = sum(m.per_joint_mse.get(jn, 0) for m in metrics_list) / n

    return agg


def write_metrics_csv(
    all_metrics: list[EpisodeMetrics],
    output_path: Path,
) -> None:
    """Write per-episode metrics to a CSV file."""
    if not all_metrics:
        return

    joint_names = list(all_metrics[0].per_joint_mae.keys())

    # Base columns
    fieldnames = [
        "episode_idx",
        "split_label",
        "mse",
        "mae",
        "rmse",
        "max_abs_error",
        "max_abs_error_joint",
        "cosine_similarity",
        "pred_smoothness_mean",
        "pred_smoothness_std",
        "gt_smoothness_mean",
        "gt_smoothness_std",
    ]

    # Add per-joint MAE columns
    for jn in joint_names:
        fieldnames.append(f"mae_{jn}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in all_metrics:
            row: dict[str, Any] = {
                "episode_idx": m.episode_idx,
                "split_label": m.split_label,
                "mse": m.mse,
                "mae": m.mae,
                "rmse": m.rmse,
                "max_abs_error": m.max_abs_error,
                "max_abs_error_joint": m.max_abs_error_joint,
                "cosine_similarity": m.cosine_similarity,
                "pred_smoothness_mean": m.pred_smoothness_mean,
                "pred_smoothness_std": m.pred_smoothness_std,
                "gt_smoothness_mean": m.gt_smoothness_mean,
                "gt_smoothness_std": m.gt_smoothness_std,
            }
            for jn in joint_names:
                row[f"mae_{jn}"] = m.per_joint_mae.get(jn, 0.0)

            writer.writerow(row)
