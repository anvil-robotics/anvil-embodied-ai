"""Tests for metrics computation."""

import numpy as np
import pytest

from anvil_eval.metrics import compute_episode_metrics, compute_summary_metrics


JOINT_NAMES = ["j1", "j2", "j3"]


def test_perfect_prediction():
    """When predicted == ground_truth, all errors should be 0."""
    gt = np.random.randn(50, 3)
    m = compute_episode_metrics(gt.copy(), gt, JOINT_NAMES, 0, "val")

    assert m.mse == pytest.approx(0.0, abs=1e-10)
    assert m.mae == pytest.approx(0.0, abs=1e-10)
    assert m.rmse == pytest.approx(0.0, abs=1e-10)
    assert m.max_abs_error == pytest.approx(0.0, abs=1e-10)
    assert m.cosine_similarity == pytest.approx(1.0, abs=1e-6)
    for jn in JOINT_NAMES:
        assert m.per_joint_mse[jn] == pytest.approx(0.0, abs=1e-10)
        assert m.per_joint_mae[jn] == pytest.approx(0.0, abs=1e-10)


def test_known_constant_offset():
    """Test with a constant offset to verify MSE/MAE/RMSE formulas."""
    gt = np.ones((10, 3))
    pred = gt + 0.5  # constant offset of 0.5

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "test")

    assert m.mse == pytest.approx(0.25, abs=1e-8)
    assert m.mae == pytest.approx(0.5, abs=1e-8)
    assert m.rmse == pytest.approx(0.5, abs=1e-8)
    assert m.max_abs_error == pytest.approx(0.5, abs=1e-8)
    assert m.num_frames == 10


def test_per_joint_metrics():
    """Verify per-joint metrics isolate correctly."""
    gt = np.zeros((20, 3))
    pred = np.zeros((20, 3))
    pred[:, 1] = 1.0  # only joint j2 has error

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "val")

    assert m.per_joint_mae["j1"] == pytest.approx(0.0, abs=1e-10)
    assert m.per_joint_mae["j2"] == pytest.approx(1.0, abs=1e-10)
    assert m.per_joint_mae["j3"] == pytest.approx(0.0, abs=1e-10)
    assert m.max_abs_error_joint == "j2"


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have cosine similarity ~0."""
    gt = np.zeros((10, 3))
    pred = np.zeros((10, 3))
    gt[:, 0] = 1.0
    pred[:, 1] = 1.0

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "val")
    assert m.cosine_similarity == pytest.approx(0.0, abs=1e-6)


def test_smoothness_constant():
    """Constant actions should have zero smoothness (no deltas)."""
    actions = np.ones((10, 3)) * 2.0
    m = compute_episode_metrics(actions, actions, JOINT_NAMES, 0, "val")

    assert m.pred_smoothness_mean == pytest.approx(0.0, abs=1e-10)
    assert m.gt_smoothness_mean == pytest.approx(0.0, abs=1e-10)


def test_smoothness_linear():
    """Linearly increasing actions should have constant smoothness."""
    gt = np.zeros((10, 3))
    gt[:, 0] = np.arange(10) * 0.1  # linearly increasing

    m = compute_episode_metrics(gt, gt, JOINT_NAMES, 0, "val")

    # delta = 0.1 for each step, L2 norm = 0.1
    assert m.gt_smoothness_mean == pytest.approx(0.1, abs=1e-6)
    assert m.gt_smoothness_std == pytest.approx(0.0, abs=1e-6)


def test_single_frame():
    """Single frame should not crash on smoothness."""
    gt = np.ones((1, 3))
    m = compute_episode_metrics(gt, gt, JOINT_NAMES, 0, "val")
    assert m.num_frames == 1
    assert m.pred_smoothness_mean == 0.0


def test_summary_metrics():
    """Test summary aggregation across episodes."""
    gt1 = np.zeros((10, 3))
    pred1 = gt1 + 0.1
    gt2 = np.zeros((10, 3))
    pred2 = gt2 + 0.2

    m1 = compute_episode_metrics(pred1, gt1, JOINT_NAMES, 0, "val")
    m2 = compute_episode_metrics(pred2, gt2, JOINT_NAMES, 1, "val")

    summary = compute_summary_metrics([m1, m2])
    assert "val" in summary
    assert summary["val"]["num_episodes"] == 2
    assert summary["val"]["mae_mean"] == pytest.approx(0.15, abs=1e-6)
