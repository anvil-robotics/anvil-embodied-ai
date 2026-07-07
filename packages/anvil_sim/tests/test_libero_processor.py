"""Tests for the pure-math native-delta conversion in anvil_sim.libero_processor.

These test the coordinate/scale math independent of a live LiberoEnv — the
live-env integration (nested robot_state extraction, image flip) is
exercised separately once a trained checkpoint is available for a real
closed-loop rollout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from anvil_shared.ee_transform import ee_rel_world_forward
from anvil_shared.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
)

from anvil_sim.libero_convert import native_action_to_rot6d, native_delta_to_goal
from anvil_sim.libero_processor import (
    GRIPPER_CLOSE_CMD,
    GRIPPER_OPEN_CMD,
    NATIVE_POS_SCALE,
    NATIVE_ROT_SCALE,
    AnvilEEActionProcessorStep,
    AnvilEEObsProcessorStep,
    NativeRot6dActionProcessorStep,
    ZeroCalActionProcessorStep,
    absolute_native_action_from_target,
    native_action_from_targets,
    native_action_from_world_delta,
    recovered_delta_native_action,
    rot6d_action_to_native,
)


def test_zero_delta_gives_zero_pose_action():
    """No motion requested -> native pos/rot components should be ~zero."""
    pos = np.array([0.1, 0.2, 0.3])
    quat = matrix_to_quat(np.eye(3))
    rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_targets(
        target_pos=pos, target_rot6d=rot6d, target_gripper=0.02,
        current_pos=pos, current_quat_xyzw=quat, current_gripper=0.02,
    )
    np.testing.assert_allclose(action[:6], 0.0, atol=1e-10)


def test_position_delta_scales_by_calibrated_factor():
    """A pure world-frame translation should scale linearly by NATIVE_POS_SCALE."""
    current_pos = np.zeros(3)
    target_pos = np.array([0.001, -0.002, 0.0005])
    quat = matrix_to_quat(np.eye(3))
    rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_targets(
        target_pos=target_pos, target_rot6d=rot6d, target_gripper=0.0,
        current_pos=current_pos, current_quat_xyzw=quat, current_gripper=0.0,
    )
    expected = target_pos * NATIVE_POS_SCALE
    np.testing.assert_allclose(action[:3], expected, atol=1e-6)


def test_position_delta_is_world_frame_not_body_frame():
    """Regression guard for the calibration finding: rotating the CURRENT
    orientation must NOT change the position component of the native action
    (world-frame delta), unlike Anvil's UMI-style ee_rel which explicitly
    rotates translation into the body frame."""
    current_pos = np.zeros(3)
    target_pos = np.array([0.01, 0.0, 0.0])
    identity_quat = matrix_to_quat(np.eye(3))
    rotated_quat = matrix_to_quat(axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2])))
    rot6d = matrix_to_rot6d(np.eye(3))

    action_identity = native_action_from_targets(
        target_pos=target_pos, target_rot6d=rot6d, target_gripper=0.0,
        current_pos=current_pos, current_quat_xyzw=identity_quat, current_gripper=0.0,
    )
    action_rotated = native_action_from_targets(
        target_pos=target_pos, target_rot6d=rot6d, target_gripper=0.0,
        current_pos=current_pos, current_quat_xyzw=rotated_quat, current_gripper=0.0,
    )
    np.testing.assert_allclose(action_identity[:3], action_rotated[:3], atol=1e-10)


def test_rotation_delta_scales_by_calibrated_factor():
    current_quat = matrix_to_quat(np.eye(3))
    target_R = axis_angle_to_matrix(np.array([0.01, 0.0, 0.0]))
    target_rot6d = matrix_to_rot6d(target_R)
    pos = np.zeros(3)

    action = native_action_from_targets(
        target_pos=pos, target_rot6d=target_rot6d, target_gripper=0.0,
        current_pos=pos, current_quat_xyzw=current_quat, current_gripper=0.0,
    )
    expected_rot = np.array([0.01, 0.0, 0.0]) * NATIVE_ROT_SCALE
    np.testing.assert_allclose(action[3:6], expected_rot, atol=1e-4)


@pytest.mark.parametrize(
    ("target_gripper", "current_gripper", "expected_cmd"),
    [
        (0.04, 0.01, GRIPPER_OPEN_CMD),   # target wider than current -> open
        (0.01, 0.04, GRIPPER_CLOSE_CMD),  # target narrower than current -> close
        (-0.04, 0.01, GRIPPER_OPEN_CMD),  # sign-agnostic: |target| still larger
    ],
)
def test_gripper_is_bang_bang_toward_target(target_gripper, current_gripper, expected_cmd):
    pos = np.zeros(3)
    quat = matrix_to_quat(np.eye(3))
    rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_targets(
        target_pos=pos, target_rot6d=rot6d, target_gripper=target_gripper,
        current_pos=pos, current_quat_xyzw=quat, current_gripper=current_gripper,
    )
    assert action[6] == expected_cmd


def test_matches_matrix_to_axis_angle_directly():
    """Cross-check against the anvil_shared primitives directly (no duplicated math)."""
    current_quat = matrix_to_quat(axis_angle_to_matrix(np.array([0.3, -0.1, 0.2])))
    target_rot6d = matrix_to_rot6d(axis_angle_to_matrix(np.array([0.35, -0.05, 0.25])))
    pos = np.array([1.0, 2.0, 3.0])

    action = native_action_from_targets(
        target_pos=pos, target_rot6d=target_rot6d, target_gripper=0.0,
        current_pos=pos, current_quat_xyzw=current_quat, current_gripper=0.0,
    )
    R_current = quat_to_matrix(current_quat)
    from anvil_shared.rotation import rot6d_to_matrix
    R_target = rot6d_to_matrix(target_rot6d)
    expected_aa = matrix_to_axis_angle(R_target @ R_current.T) * NATIVE_ROT_SCALE
    np.testing.assert_allclose(action[3:6], expected_aa, atol=1e-8)


# =============================================================================
# native_action_from_world_delta -- experimental ee_delta arm (4th, not part
# of Anvil's real contract; see anvil_sim.libero_convert module docstring).
# Mirrors the native_action_from_targets tests above since it shares the
# same calibrated scales, just takes an already-relative delta instead of an
# absolute target.
# =============================================================================


def test_world_delta_zero_gives_zero_pose_action():
    zero_rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_world_delta(
        delta_pos=np.zeros(3), delta_rot6d=zero_rot6d, target_gripper=0.02, current_gripper=0.02,
    )
    np.testing.assert_allclose(action[:6], 0.0, atol=1e-10)


def test_world_delta_position_scales_by_calibrated_factor():
    delta_pos = np.array([0.001, -0.002, 0.0005])
    zero_rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_world_delta(
        delta_pos=delta_pos, delta_rot6d=zero_rot6d, target_gripper=0.0, current_gripper=0.0,
    )
    np.testing.assert_allclose(action[:3], delta_pos * NATIVE_POS_SCALE, atol=1e-6)


def test_world_delta_rotation_scales_by_calibrated_factor():
    delta_rot6d = matrix_to_rot6d(axis_angle_to_matrix(np.array([0.01, 0.0, 0.0])))
    action = native_action_from_world_delta(
        delta_pos=np.zeros(3), delta_rot6d=delta_rot6d, target_gripper=0.0, current_gripper=0.0,
    )
    expected_rot = np.array([0.01, 0.0, 0.0]) * NATIVE_ROT_SCALE
    np.testing.assert_allclose(action[3:6], expected_rot, atol=1e-4)


@pytest.mark.parametrize(
    ("target_gripper", "current_gripper", "expected_cmd"),
    [
        (0.04, 0.01, GRIPPER_OPEN_CMD),
        (0.01, 0.04, GRIPPER_CLOSE_CMD),
        (-0.04, 0.01, GRIPPER_OPEN_CMD),
    ],
)
def test_world_delta_gripper_is_bang_bang_toward_target(target_gripper, current_gripper, expected_cmd):
    zero_rot6d = matrix_to_rot6d(np.eye(3))
    action = native_action_from_world_delta(
        delta_pos=np.zeros(3), delta_rot6d=zero_rot6d,
        target_gripper=target_gripper, current_gripper=current_gripper,
    )
    assert action[6] == expected_cmd


def test_world_delta_matches_native_action_from_targets_when_composed():
    """native_action_from_world_delta(delta) must equal
    native_action_from_targets(target=current+delta) for a consistent
    current/target pair -- proves ee_delta's math is the same as ee_abs's,
    minus the "target - current" step (already done at dataset-write time
    instead of at eval time)."""
    current_pos = np.array([1.0, 2.0, 3.0])
    current_quat = matrix_to_quat(axis_angle_to_matrix(np.array([0.3, -0.1, 0.2])))
    R_current = quat_to_matrix(current_quat)

    delta_pos = np.array([0.01, -0.02, 0.005])
    R_delta = axis_angle_to_matrix(np.array([0.02, 0.01, -0.03]))
    delta_rot6d = matrix_to_rot6d(R_delta)

    target_pos = current_pos + delta_pos
    target_rot6d = matrix_to_rot6d(R_delta @ R_current)

    action_delta = native_action_from_world_delta(
        delta_pos=delta_pos, delta_rot6d=delta_rot6d, target_gripper=0.02, current_gripper=0.02,
    )
    action_target = native_action_from_targets(
        target_pos=target_pos, target_rot6d=target_rot6d, target_gripper=0.02,
        current_pos=current_pos, current_quat_xyzw=current_quat, current_gripper=0.02,
    )
    np.testing.assert_allclose(action_delta, action_target, atol=1e-5)


def test_ee_rel_action_uses_chunk_start_anchor_not_fresh_observation():
    """Regression test for a real bug: ACTPolicy.select_action() only
    re-queries the model every n_action_steps calls (it pops cached actions
    from its internal queue the rest of the time), so an ee_rel action's
    correct ee_rel_inverse reference is the observation from when its chunk
    was generated -- NOT the fresh observation at the time that action is
    individually executed. Using the fresh observation as originally
    implemented silently drifted worse and worse within each chunk."""
    obs_step = AnvilEEObsProcessorStep(action_type="ee_rel")
    action_step = AnvilEEActionProcessorStep(action_type="ee_rel", obs_step=obs_step, n_action_steps=3)

    # A zero rot6d/xyz-delta "stay here" relative action, at some fixed gripper value.
    identity_rel_action = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02])

    states = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32),  # chunk start (anchor)
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32),  # robot has moved
        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32),  # moved further
    ]

    natives = []
    for state in states:
        obs_step.last_anvil_state = state
        natives.append(action_step.action(identity_rel_action).numpy()[0])

    # All three calls belong to the same chunk (n_action_steps=3), so the
    # ee_rel_inverse reference must stay pinned to `states[0]` throughout --
    # meaning the resulting native world-frame position delta should track
    # "distance from the ANCHOR", growing as the robot moves away from it,
    # not stay near zero as it would if each call wrongly re-anchored to its
    # own fresh (already-there) state.
    expected_deltas = [(states[0][:3] - s[:3]) * NATIVE_POS_SCALE for s in states]
    for native, expected in zip(natives, expected_deltas, strict=True):
        np.testing.assert_allclose(native[:3], expected, atol=1e-4)

    # Next chunk (4th call, count%3==0 again) must re-anchor to the NOW-current state.
    obs_step.last_anvil_state = states[2]
    native_new_chunk = action_step.action(identity_rel_action).numpy()[0]
    np.testing.assert_allclose(native_new_chunk[:3], 0.0, atol=1e-4)


# =============================================================================
# rot6d_action_to_native / NativeRot6dActionProcessorStep -- experimental 5th
# arm (native_rot6d), zero-calibration isolation of rot6d vs axis-angle.
# =============================================================================


def test_rot6d_action_to_native_is_exact_inverse_of_native_action_to_rot6d():
    """No NATIVE_POS_SCALE/NATIVE_ROT_SCALE involved -- pure format
    round-trip, must recover the original native-command-scale numbers
    exactly (not approximately, no calibration residual)."""
    native_action = np.array([0.02, -0.01, 0.03, 0.15, -0.08, 0.05, -1.0], dtype=np.float32)
    rot6d_action = native_action_to_rot6d(native_action)
    recovered = rot6d_action_to_native(rot6d_action)
    np.testing.assert_allclose(recovered, native_action, atol=1e-5)


def test_native_rot6d_action_processor_step_matches_pure_function():
    """The processor step is a thin torch wrapper -- must match
    rot6d_action_to_native exactly, with no dependency on any observation
    (unlike AnvilEEActionProcessorStep, this step needs no obs_step at all)."""
    native_action = np.array([0.02, -0.01, 0.03, 0.15, -0.08, 0.05, -1.0], dtype=np.float32)
    rot6d_action = native_action_to_rot6d(native_action)

    step = NativeRot6dActionProcessorStep()
    result = step.action(torch.from_numpy(rot6d_action).unsqueeze(0)).numpy()[0]

    expected = rot6d_action_to_native(rot6d_action)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# absolute_native_action_from_target / ZeroCalActionProcessorStep -- the
# zero-calibration re-run (env.control_mode="absolute"), replacing
# NATIVE_POS_SCALE/NATIVE_ROT_SCALE reconstruction entirely.
# =============================================================================


def test_absolute_native_action_applies_zero_scaling():
    """Position/rotation must pass through with NO NATIVE_POS_SCALE/
    NATIVE_ROT_SCALE multiplication -- control_mode='absolute' takes metres
    and axis-angle radians directly (confirmed from robosuite osc.py source)."""
    target_pos = np.array([0.15, -0.08, 0.42])
    target_rot6d = matrix_to_rot6d(axis_angle_to_matrix(np.array([0.02, -0.01, 0.015])))

    action = absolute_native_action_from_target(
        target_pos=target_pos, target_rot6d=target_rot6d, target_gripper=0.02, current_gripper=0.02,
    )
    np.testing.assert_allclose(action[:3], target_pos, atol=1e-8)  # NOT scaled by NATIVE_POS_SCALE
    expected_rot = matrix_to_axis_angle(axis_angle_to_matrix(np.array([0.02, -0.01, 0.015])))
    np.testing.assert_allclose(action[3:6], expected_rot, atol=1e-8)  # NOT scaled by NATIVE_ROT_SCALE


@pytest.mark.parametrize(
    ("target_gripper", "current_gripper", "expected_cmd"),
    [
        (0.04, 0.01, GRIPPER_OPEN_CMD),
        (0.01, 0.04, GRIPPER_CLOSE_CMD),
    ],
)
def test_absolute_native_action_gripper_is_bang_bang(target_gripper, current_gripper, expected_cmd):
    action = absolute_native_action_from_target(
        target_pos=np.zeros(3), target_rot6d=matrix_to_rot6d(np.eye(3)),
        target_gripper=target_gripper, current_gripper=current_gripper,
    )
    assert action[6] == expected_cmd


def _make_zero_cal_step(mode, n_action_steps=1):
    obs_step = AnvilEEObsProcessorStep(action_type="ee_rel" if mode in ("rel_world", "rel_hand") else "ee_abs")
    return obs_step, ZeroCalActionProcessorStep(mode=mode, obs_step=obs_step, n_action_steps=n_action_steps)


def test_zero_cal_abs_mode_uses_policy_output_directly():
    obs_step, step = _make_zero_cal_step("abs")
    obs_step.last_anvil_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)

    target_pos = np.array([0.15, 0.25, 0.35])
    target_rot6d = matrix_to_rot6d(np.eye(3))
    act10 = torch.tensor([*target_pos, *target_rot6d, 0.02])

    result = step.action(act10).numpy()[0]
    expected = absolute_native_action_from_target(target_pos, target_rot6d, 0.02, 0.02)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_zero_cal_rel_hand_mode_matches_ee_rel_inverse():
    """mode='rel_hand' must reconstruct via the EXISTING ee_rel_inverse
    (unchanged, no calibration issue there) then feed the result to
    absolute_native_action_from_target (zero-cal) -- reusing the existing
    ee_rel checkpoint's data, just replacing the final scaled-delta step."""
    from anvil_shared.ee_transform import ee_rel_inverse

    anchor = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    rel_action = np.array([0.01, -0.01, 0.005, *matrix_to_rot6d(axis_angle_to_matrix([0.02, 0.0, 0.0])), 0.03])

    obs_step, step = _make_zero_cal_step("rel_hand")
    obs_step.last_anvil_state = anchor

    result = step.action(torch.from_numpy(rel_action.astype(np.float32))).numpy()[0]

    abs10 = ee_rel_inverse(rel_action.reshape(1, 10), anchor.reshape(1, 8))[0]
    expected = absolute_native_action_from_target(abs10[:3], abs10[3:9], abs10[9], anchor[7])
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_zero_cal_rel_world_mode_matches_ee_rel_world_inverse():
    from anvil_shared.ee_transform import ee_rel_world_inverse

    anchor = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    rel_action = np.array([0.01, -0.01, 0.005, *matrix_to_rot6d(axis_angle_to_matrix([0.02, 0.0, 0.0])), 0.03])

    obs_step, step = _make_zero_cal_step("rel_world")
    obs_step.last_anvil_state = anchor

    result = step.action(torch.from_numpy(rel_action.astype(np.float32))).numpy()[0]

    abs10 = ee_rel_world_inverse(rel_action.reshape(1, 10), anchor.reshape(1, 8))[0]
    expected = absolute_native_action_from_target(abs10[:3], abs10[3:9], abs10[9], anchor[7])
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_zero_cal_rel_world_seq_accumulates_consecutive_deltas_to_reconstruct_trajectory():
    """mode='rel_world_seq' (reused ee_delta checkpoint) must accumulate
    per-step consecutive world-frame deltas onto a running target starting
    from the chunk anchor, exactly reconstructing the true trajectory when
    fed the true per-step deltas (anvil_state_to_delta_action's convention:
    R_delta = R_next @ R_current.T, matching ee_rel_world_forward)."""
    state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    state1 = np.array(
        [0.01, 0.02, -0.01, *matrix_to_quat(axis_angle_to_matrix([0.0, 0.0, 0.05])), 0.02], dtype=np.float32
    )
    state2 = np.array(
        [0.03, 0.01, 0.0, *matrix_to_quat(axis_angle_to_matrix([0.0, 0.0, 0.09])), 0.04], dtype=np.float32
    )

    def _consecutive_delta(cur, nxt):
        abs10 = np.concatenate([nxt[:3], matrix_to_rot6d(quat_to_matrix(nxt[3:7])), [nxt[7]]])
        return ee_rel_world_forward(abs10.reshape(1, 10), cur.reshape(1, 8))[0]

    delta0 = _consecutive_delta(state0, state1)
    delta1 = _consecutive_delta(state1, state2)

    obs_step, step = _make_zero_cal_step("rel_world_seq", n_action_steps=2)
    obs_step.last_anvil_state = state0  # chunk start

    native0 = step.action(torch.from_numpy(delta0.astype(np.float32))).numpy()[0]
    expected0 = absolute_native_action_from_target(state1[:3], matrix_to_rot6d(quat_to_matrix(state1[3:7])), state1[7], state0[7])
    np.testing.assert_allclose(native0, expected0, atol=1e-4)

    native1 = step.action(torch.from_numpy(delta1.astype(np.float32))).numpy()[0]
    expected1 = absolute_native_action_from_target(state2[:3], matrix_to_rot6d(quat_to_matrix(state2[3:7])), state2[7], state0[7])
    np.testing.assert_allclose(native1, expected1, atol=1e-4)


def test_zero_cal_rel_hand_seq_accumulates_consecutive_deltas_to_reconstruct_trajectory():
    """mode='rel_hand_seq' (7th-round hand-n(n-1) condition, the one
    genuinely new condition) must accumulate per-step consecutive HAND-frame
    deltas onto a running target starting from the chunk anchor, exactly
    reconstructing the true trajectory when fed the true per-step deltas
    (ee_rel_forward's convention: R_delta = R_prev.T @ R_next, body-frame
    translation) -- mirrors test_zero_cal_rel_world_seq_accumulates_..., just
    with ee_rel_forward (hand) instead of ee_rel_world_forward (world)."""
    from anvil_shared.ee_transform import ee_rel_forward

    state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    state1 = np.array(
        [0.01, 0.02, -0.01, *matrix_to_quat(axis_angle_to_matrix([0.0, 0.0, 0.05])), 0.02], dtype=np.float32
    )
    state2 = np.array(
        [0.03, 0.01, 0.0, *matrix_to_quat(axis_angle_to_matrix([0.0, 0.0, 0.09])), 0.04], dtype=np.float32
    )

    def _consecutive_delta(cur, nxt):
        abs10 = np.concatenate([nxt[:3], matrix_to_rot6d(quat_to_matrix(nxt[3:7])), [nxt[7]]])
        return ee_rel_forward(abs10.reshape(1, 10), cur.reshape(1, 8))[0]

    delta0 = _consecutive_delta(state0, state1)
    delta1 = _consecutive_delta(state1, state2)

    obs_step, step = _make_zero_cal_step("rel_hand_seq", n_action_steps=2)
    obs_step.last_anvil_state = state0  # chunk start

    native0 = step.action(torch.from_numpy(delta0.astype(np.float32))).numpy()[0]
    expected0 = absolute_native_action_from_target(
        state1[:3], matrix_to_rot6d(quat_to_matrix(state1[3:7])), state1[7], state0[7]
    )
    np.testing.assert_allclose(native0, expected0, atol=1e-4)

    native1 = step.action(torch.from_numpy(delta1.astype(np.float32))).numpy()[0]
    expected1 = absolute_native_action_from_target(
        state2[:3], matrix_to_rot6d(quat_to_matrix(state2[3:7])), state2[7], state0[7]
    )
    np.testing.assert_allclose(native1, expected1, atol=1e-4)


# =============================================================================
# recovered_delta_native_action / ZeroCalActionProcessorStep(deliver="relative")
# -- 7th-round "goal" target family fix. v1 fed a SCALED absolute target to
# env.control_mode="absolute" and got 0% pc_success across all 5 conditions
# because the scale it assumed (robosuite's OSC_POSE output_max=0.05/0.5)
# doesn't match the real per-step displacement (empirically ~22% of that,
# due to impedance-controller lag -- see native_delta_to_goal's docstring).
# The fix: never scale ourselves; recover a native-delta-shaped quantity
# relative to the CURRENT real state and deliver via control_mode="relative",
# letting robosuite's own scale_action apply the true physical scale.
# =============================================================================


def test_recovered_delta_native_action_roundtrips_with_native_delta_to_goal():
    """The key regression check for the bug that slipped through v1's unit
    tests: reconstruct a goal via native_delta_to_goal(state, native_delta),
    then recover it via recovered_delta_native_action(..., current_state=state,
    ...) -- for the k=0 / 'abs' case (same state used on both sides, no
    anchor drift), this must recover the EXACT original native_delta with
    zero error, since nothing in between should introduce approximation."""
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.1, -0.2, 0.05])))
    state = np.array([0.3, -0.1, 0.5, *quat0, 0.02], dtype=np.float32)
    native_delta = np.array([0.4, -0.6, 0.3, 0.2, -0.4, 0.15, -1.0], dtype=np.float32)  # within [-1,1]

    goal = native_delta_to_goal(state, native_delta)
    goal_rot6d = matrix_to_rot6d(quat_to_matrix(goal[3:7]))

    recovered = recovered_delta_native_action(
        reconstructed_pos=goal[:3],
        reconstructed_rot6d=goal_rot6d,
        reconstructed_gripper=goal[7],
        current_state=state,
        current_gripper=state[7],
    )

    np.testing.assert_allclose(recovered[:6], native_delta[:6], atol=1e-5)


def test_recovered_delta_native_action_uses_current_state_not_a_stale_one():
    """If the CURRENT real state has drifted from whatever state the
    reconstructed target was implicitly built against, the recovered delta
    must reflect that drift (this is exactly the closed-loop self-correction
    n-0 anchoring relies on) -- i.e. the function must genuinely use
    current_state, not silently ignore it."""
    reconstructed_pos = np.array([0.5, 0.5, 0.5])
    reconstructed_rot6d = matrix_to_rot6d(np.eye(3))

    near_state = np.array([0.49, 0.49, 0.49, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    far_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)

    near_delta = recovered_delta_native_action(
        reconstructed_pos, reconstructed_rot6d, 0.02, near_state, 0.02
    )
    far_delta = recovered_delta_native_action(
        reconstructed_pos, reconstructed_rot6d, 0.02, far_state, 0.02
    )

    np.testing.assert_allclose(near_delta[:3], [0.01, 0.01, 0.01], atol=1e-6)
    # far_state is 0.5 away in each dim (still within [-1,1], not clipped) -- must differ from near_delta.
    np.testing.assert_allclose(far_delta[:3], [0.5, 0.5, 0.5], atol=1e-6)


def test_recovered_delta_native_action_clips_to_unit_range():
    reconstructed_pos = np.array([10.0, -10.0, 10.0])
    reconstructed_rot6d = matrix_to_rot6d(np.eye(3))
    current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)

    delta = recovered_delta_native_action(
        reconstructed_pos, reconstructed_rot6d, 0.02, current_state, 0.02
    )
    np.testing.assert_allclose(delta[:3], [1.0, -1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    ("reconstructed_gripper", "current_gripper", "expected_cmd"),
    [
        (0.04, 0.01, GRIPPER_OPEN_CMD),
        (0.01, 0.04, GRIPPER_CLOSE_CMD),
    ],
)
def test_recovered_delta_native_action_gripper_is_bang_bang(
    reconstructed_gripper, current_gripper, expected_cmd
):
    action = recovered_delta_native_action(
        reconstructed_pos=np.zeros(3),
        reconstructed_rot6d=matrix_to_rot6d(np.eye(3)),
        reconstructed_gripper=reconstructed_gripper,
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, current_gripper], dtype=np.float32),
        current_gripper=current_gripper,
    )
    assert action[6] == expected_cmd


def test_zero_cal_deliver_relative_uses_recovered_delta_not_absolute_target():
    """deliver='relative' (7th-round 'goal' family) must route through
    recovered_delta_native_action, not absolute_native_action_from_target --
    the default deliver='absolute' (6th-round) path must be completely
    unaffected (see test_zero_cal_abs_mode_uses_policy_output_directly)."""
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")
    step = ZeroCalActionProcessorStep(mode="abs", obs_step=obs_step, deliver="relative")
    current_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    obs_step.last_anvil_state = current_state

    target_pos = np.array([0.15, 0.25, 0.35])
    target_rot6d = matrix_to_rot6d(np.eye(3))
    act10 = torch.tensor([*target_pos, *target_rot6d, 0.02])

    result = step.action(act10).numpy()[0]
    expected = recovered_delta_native_action(
        target_pos, target_rot6d, 0.02, current_state, current_gripper=0.02
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_zero_cal_deliver_defaults_to_absolute():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")
    step = ZeroCalActionProcessorStep(mode="abs", obs_step=obs_step)
    assert step.deliver == "absolute"


# =============================================================================
# gripper_mode="native_cmd" -- the 4th real bug, found by the GT-replay tool
# on its first run: the goalabs dataset family stores the gripper as LIBERO's
# native +/-1 COMMAND, but the bang-bang comparator abs(target)<abs(qpos) is
# only meaningful for qpos-scale targets -- fed +/-1 it is ALWAYS False, the
# gripper never closes, and every goalabs rollout fails at exactly 0%
# regardless of policy quality.
# =============================================================================


@pytest.mark.parametrize(
    ("native_cmd_gripper", "expected"),
    [
        (1.0, 1.0),    # open command passes through
        (-1.0, -1.0),  # close command passes through -- the case the old comparator broke
        (2.5, 1.0),    # out-of-range clipped
    ],
)
def test_recovered_delta_gripper_native_cmd_passes_through(native_cmd_gripper, expected):
    action = recovered_delta_native_action(
        reconstructed_pos=np.zeros(3),
        reconstructed_rot6d=matrix_to_rot6d(np.eye(3)),
        reconstructed_gripper=native_cmd_gripper,
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32),
        current_gripper=0.02,
        gripper_mode="native_cmd",
    )
    assert action[6] == expected


def test_qpos_comparator_always_opens_for_native_cmd_targets():
    """Regression documentation of the bug itself: a +/-1 native command fed
    through the DEFAULT qpos comparator always yields OPEN — including for
    the CLOSE command — which is why every goalabs rollout scored 0%."""
    for cmd in (1.0, -1.0):
        action = recovered_delta_native_action(
            reconstructed_pos=np.zeros(3),
            reconstructed_rot6d=matrix_to_rot6d(np.eye(3)),
            reconstructed_gripper=cmd,
            current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32),
            current_gripper=0.02,
        )
        assert action[6] == GRIPPER_OPEN_CMD  # even for the close command -- the bug


def test_absolute_native_action_gripper_native_cmd_passes_through():
    action = absolute_native_action_from_target(
        target_pos=np.zeros(3),
        target_rot6d=matrix_to_rot6d(np.eye(3)),
        target_gripper=-1.0,
        current_gripper=0.02,
        gripper_mode="native_cmd",
    )
    assert action[6] == -1.0


def test_zero_cal_rejects_invalid_gripper_mode():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")
    with pytest.raises(ValueError, match="gripper_mode"):
        ZeroCalActionProcessorStep(mode="abs", obs_step=obs_step, gripper_mode="bogus")


# =============================================================================
# reset_episode_state -- real bug #5: the chunk-anchor call counter ran on
# ACROSS episodes while the policy replans from episode-local step 0 after
# every policy.reset(). Unless an episode's length was a multiple of
# n_action_steps, every episode after the first reconstructed targets
# against an anchor captured at the wrong time (initially one from the
# PREVIOUS episode's scene). Exposed by the GT-replay diagnostic: world-n0
# replay at n_action_steps=100 scored 20% where the forward/inverse
# identity predicts parity with abs (80%).
# =============================================================================


def _rel_world_step(n_action_steps):
    obs_step = AnvilEEObsProcessorStep(action_type="ee_rel")
    step = ZeroCalActionProcessorStep(
        mode="rel_world", obs_step=obs_step, n_action_steps=n_action_steps, deliver="absolute"
    )
    return obs_step, step


def test_bug5_regression_anchor_desync_across_episodes():
    """Without reset_episode_state, an episode whose length is NOT a
    multiple of n_action_steps leaves the counter mid-chunk, so the next
    episode's first action is reconstructed against the PREVIOUS episode's
    anchor. With the reset, the next episode re-anchors at its own start."""
    ep1_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    ep2_state = np.array([0.7, -0.4, 0.9, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    identity_rel = np.concatenate(
        [np.zeros(3), matrix_to_rot6d(np.eye(3)), [0.02]]
    ).astype(np.float32)

    # --- WITHOUT the fix: broken behavior (documented) ---
    obs_step, step = _rel_world_step(n_action_steps=2)
    obs_step.last_anvil_state = ep1_state
    step.action(torch.from_numpy(identity_rel))  # ep1 step 0: anchors at ep1_state
    # episode 1 ends after ONE step (not a multiple of 2); episode 2 begins:
    obs_step.last_anvil_state = ep2_state
    native = step.action(torch.from_numpy(identity_rel)).numpy()[0]
    # counter=1 -> NOT treated as chunk start -> stale ep1 anchor is used:
    np.testing.assert_allclose(native[:3], ep1_state[:3], atol=1e-6)  # the bug

    # --- WITH the fix ---
    obs_step, step = _rel_world_step(n_action_steps=2)
    obs_step.last_anvil_state = ep1_state
    step.action(torch.from_numpy(identity_rel))
    step.reset_episode_state()  # what the rollout wrapper now does
    obs_step.last_anvil_state = ep2_state
    native = step.action(torch.from_numpy(identity_rel)).numpy()[0]
    np.testing.assert_allclose(native[:3], ep2_state[:3], atol=1e-6)  # re-anchored


def test_bug5_reset_clears_running_target_too():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")
    step = ZeroCalActionProcessorStep(
        mode="rel_world_seq", obs_step=obs_step, n_action_steps=2, deliver="absolute"
    )
    obs_step.last_anvil_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    delta = np.concatenate([[0.01, 0.0, 0.0], matrix_to_rot6d(np.eye(3)), [0.02]]).astype(np.float32)
    step.action(torch.from_numpy(delta))
    assert step._running_target is not None
    step.reset_episode_state()
    assert step._running_target is None and step._call_count == 0 and step._chunk_anchor is None


def test_bug5_anvil_ee_step_reset():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_rel")
    step = AnvilEEActionProcessorStep(action_type="ee_rel", obs_step=obs_step, n_action_steps=2)
    obs_step.last_anvil_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    rel = np.concatenate([np.zeros(3), matrix_to_rot6d(np.eye(3)), [0.02]]).astype(np.float32)
    step.action(torch.from_numpy(rel))
    assert step._call_count == 1 and step._chunk_anchor is not None
    step.reset_episode_state()
    assert step._call_count == 0 and step._chunk_anchor is None


def test_zero_cal_rejects_invalid_deliver():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")
    with pytest.raises(ValueError, match="deliver"):
        ZeroCalActionProcessorStep(mode="abs", obs_step=obs_step, deliver="bogus")
