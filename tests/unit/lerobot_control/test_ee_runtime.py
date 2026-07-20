"""Unit tests for ee_runtime.ramp_toward_pose (no ROS/rclpy or anvil_shared dependency).

ee_runtime.py has no module-level rclpy import, and ramp_toward_pose/pose_arrival_error
are plain numpy with no anvil_shared dependency either (unlike ee_delta_restore_step etc,
which lazily import it) -- importable directly via the same sys.path convention used
elsewhere in this test package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))

from lerobot_control.ee_runtime import pose_arrival_error, ramp_toward_pose  # noqa: E402

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _pose(pos, quat=IDENTITY_QUAT, gripper=0.0):
    return np.concatenate([np.asarray(pos, dtype=float), np.asarray(quat, dtype=float), [gripper]])


def test_target_within_reach_is_returned_exactly():
    """If the target is already closer than both limits, no ramping needed."""
    current = _pose([0.0, 0.0, 0.0], gripper=0.1)
    target = _pose([0.001, 0.0, 0.0], gripper=0.2)
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=0.01, max_rot_delta_deg=2.0)
    np.testing.assert_allclose(ramped[0:3], target[0:3])
    np.testing.assert_allclose(ramped[3:7], target[3:7])
    assert ramped[7] == 0.2  # gripper passes through from target unramped


def test_position_clamped_to_max_delta_preserving_direction():
    current = _pose([0.0, 0.0, 0.0])
    target = _pose([1.0, 0.0, 0.0])  # 1m away, way beyond max_pos_delta_m
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=0.01, max_rot_delta_deg=2.0)
    ramped_dist = np.linalg.norm(ramped[0:3] - current[0:3])
    assert ramped_dist == pytest.approx(0.01)
    # Direction preserved: moved purely along +x.
    np.testing.assert_allclose(ramped[0:3], [0.01, 0.0, 0.0], atol=1e-9)


def test_diagonal_position_clamped_by_magnitude_not_per_axis():
    """A magnitude-based clamp, not per-component -- this is what distinguishes
    it from the joint-space limiter's per-axis np.clip."""
    current = _pose([0.0, 0.0, 0.0])
    target = _pose([1.0, 1.0, 0.0])  # diagonal, distance = sqrt(2)
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=0.1, max_rot_delta_deg=2.0)
    ramped_dist = np.linalg.norm(ramped[0:3] - current[0:3])
    assert ramped_dist == pytest.approx(0.1)
    # Still pointed diagonally (equal x/y components), not clamped per-axis to 0.1 each.
    assert ramped[0] == pytest.approx(ramped[1])


def test_rotation_clamped_via_slerp_toward_target():
    current = _pose([0, 0, 0], quat=[0.0, 0.0, 0.0, 1.0])  # identity
    # 90 degree rotation about Z: quat = [0, 0, sin(45deg), cos(45deg)]
    target = _pose([0, 0, 0], quat=[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=1.0, max_rot_delta_deg=10.0)
    _, rot_err_from_current = pose_arrival_error(current, ramped)
    _, rot_err_to_target = pose_arrival_error(ramped, target)
    assert rot_err_from_current == pytest.approx(10.0, abs=1e-6)
    # Remaining distance to target should be roughly 90 - 10 = 80 degrees.
    assert rot_err_to_target == pytest.approx(80.0, abs=1e-6)


def test_rotation_within_reach_snaps_to_target_exactly():
    current = _pose([0, 0, 0], quat=[0.0, 0.0, 0.0, 1.0])
    target = _pose([0, 0, 0], quat=[0.0, 0.0, np.sin(np.radians(1.0) / 2), np.cos(np.radians(1.0) / 2)])
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=1.0, max_rot_delta_deg=5.0)
    np.testing.assert_allclose(ramped[3:7], target[3:7], atol=1e-9)


def test_opposite_hemisphere_quaternion_takes_shortest_path():
    """q and -q represent the same rotation -- ramping must not take the long way
    around just because the target quat happens to be sign-flipped."""
    current = _pose([0, 0, 0], quat=[0.0, 0.0, 0.0, 1.0])
    near_identity_negated = _pose(
        [0, 0, 0],
        quat=[-x for x in [0.0, 0.0, np.sin(np.radians(1.0) / 2), np.cos(np.radians(1.0) / 2)]],
    )
    ramped = ramp_toward_pose(current, near_identity_negated, max_pos_delta_m=1.0, max_rot_delta_deg=5.0)
    # A near-identical (negated) target is within reach in one step regardless
    # of the sign flip -- confirms shortest-path handling, not a ~360 deg step.
    _, rot_err = pose_arrival_error(current, ramped)
    assert rot_err < 5.0


def test_gripper_always_passes_through_from_target_unramped():
    current = _pose([0, 0, 0], gripper=0.0)
    target = _pose([5.0, 0, 0], gripper=0.9)  # far away position, but gripper should still pass through
    ramped = ramp_toward_pose(current, target, max_pos_delta_m=0.01, max_rot_delta_deg=2.0)
    assert ramped[7] == 0.9
