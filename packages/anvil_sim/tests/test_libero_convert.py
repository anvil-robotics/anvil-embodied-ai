"""Tests for the pure math in libero_convert.py's dataset-construction
functions: native_rot6d (rotation-encoding isolation), native_hand (frame
isolation), and the goalabs/native_abs/native_n0 "goal" target family
(native_delta_to_goal and its axis-angle/n-0 derivatives)."""

import numpy as np
from anvil_shared.ee_transform import ee_rel_world_forward, ee_rel_world_inverse
from anvil_shared.rotation import (
    axis_angle_to_matrix,
    matrix_to_quat,
    quat_to_matrix,
)

from anvil_sim.studies.libero_ee.libero_convert import (
    anvil_state_to_abs_action,
    convert_episode_goal_aa_actions,
    convert_episode_goal_abs_actions,
    convert_episode_goal_n0_aa_actions,
    convert_episode_goal_states,
    convert_episode_native_hand_actions,
    goal_state_to_axis_angle_action,
    native_action_to_hand,
    native_action_to_rot6d,
    native_delta_to_goal,
)
from anvil_sim.studies.libero_ee.libero_processor import (
    axis_angle_action_to_rot6d,
    hand_action_to_native,
    recovered_delta_native_action,
    rot6d_action_to_native,
)


def test_native_action_to_rot6d_roundtrip():
    """The native_rot6d arm must round-trip EXACTLY --
    native_action_to_rot6d/rot6d_action_to_native is a lossless format
    re-encoding with zero calibration. Covers near-zero rotation (the common case for this
    task) and a larger rotation, using arbitrary native-command-scale
    numbers (NOT real radians -- that's the whole point, see module
    docstring)."""
    native_actions = np.array(
        [
            [0.01, -0.02, 0.005, 0.0, 0.0, 0.0, 1.0],          # zero rotation
            [0.01, -0.02, 0.005, 0.001, -0.0008, 0.0012, -1.0],  # tiny rotation (typical for this task)
            [0.3, 0.0, -0.1, 0.5, -0.3, 0.2, 1.0],              # larger rotation
        ],
        dtype=np.float32,
    )

    for native_action in native_actions:
        rot6d_action = native_action_to_rot6d(native_action)
        assert rot6d_action.shape == (10,)

        recovered = rot6d_action_to_native(rot6d_action)
        np.testing.assert_allclose(recovered, native_action, atol=1e-5)

    # Position and gripper must pass through completely unchanged (no
    # axis-angle vs rot6d choice applies to them).
    rot6d_action = native_action_to_rot6d(native_actions[2])
    np.testing.assert_allclose(rot6d_action[:3], native_actions[2][:3], atol=1e-8)
    assert rot6d_action[9] == native_actions[2][6]


# --- Experiment 7 ("goal" target family: goal = state + scale_action(delta)) ---


def test_native_delta_to_goal_matches_unscaled_formal_composition():
    """v2 (post-bugfix): goal = state + native_delta, with ZERO scaling --
    native_delta's pos/rot components are treated purely formally (same
    trick as native_action_to_rot6d), not real metres/radians. v1 multiplied
    by robosuite's output_max (0.05/0.5) assuming that reconstructed the
    controller's exact internal target; an empirical check against real
    lerobot/libero episodes showed the true per-step displacement is only
    ~22% of that (impedance controller lag, not a bug), so v1's "goal" was
    ~4.5x too large and caused catastrophic closed-loop failure (0% across
    all 5 Experiment 7 conditions) when fed directly as an absolute target.
    See native_delta_to_goal's docstring."""
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.1, -0.2, 0.05])))
    state = np.array([0.3, -0.1, 0.5, *quat0, 0.02], dtype=np.float32)
    native_delta = np.array([0.4, -0.6, 0.8, 0.2, -0.4, 0.6, -1.0], dtype=np.float32)  # within [-1,1]

    goal = native_delta_to_goal(state, native_delta)

    expected_pos = state[:3] + native_delta[:3]
    np.testing.assert_allclose(goal[:3], expected_pos, atol=1e-6)

    expected_R = axis_angle_to_matrix(native_delta[3:6]) @ quat_to_matrix(quat0)
    np.testing.assert_allclose(quat_to_matrix(goal[3:7]), expected_R, atol=1e-6)

    assert goal[7] == native_delta[6]  # gripper: native's own command, passthrough


def test_native_delta_to_goal_clips_out_of_range_delta():
    """input_min/input_max = -1/1 -- values outside that range must still be
    clipped (matching robosuite's own scale_action() np.clip call), even
    though we no longer apply its output_max scaling ourselves."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02], dtype=np.float32)
    native_delta = np.array([2.0, -3.0, 1.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    goal = native_delta_to_goal(state, native_delta)

    # clip(2.0,-1,1)=1.0, clip(-3.0,-1,1)=-1.0, clip(1.5,-1,1)=1.0
    np.testing.assert_allclose(goal[:3], np.array([1.0, -1.0, 1.0]), atol=1e-6)


def test_native_delta_to_goal_zero_delta_is_identity():
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.3, 0.1, -0.2])))
    state = np.array([0.1, 0.2, 0.3, *quat0, 0.04], dtype=np.float32)
    zero_delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)

    goal = native_delta_to_goal(state, zero_delta)

    np.testing.assert_allclose(goal[:7], state[:7], atol=1e-6)
    assert goal[7] == 0.04


def test_goalabs_action_column_roundtrips_through_ee_rel_world_transform():
    """Data-consistency check for the shared `goalabs` dataset (conditions
    #1/#2/#3): anvil_trainer's EERelWorldTransform relativizes the stored
    action (encode(G[t])) against the REAL observed state at the anchor
    (observation.state[-1], NOT G itself -- see EERelWorldTransform.apply())
    at load time. Confirm round-tripping through
    ee_rel_world_forward/inverse with that anchor exactly recovers the
    original encode(G[t]) -- i.e. the (anvil_states, goal_states) pairing
    this module writes into the dataset is consistent with what the
    load-time transform expects."""
    quat_s0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.05, 0.0, 0.0])))
    anvil_states = np.array([[0.1, 0.2, 0.3, *quat_s0, 0.02]], dtype=np.float32)
    native_actions = np.array([[0.4, -0.2, 0.6, 0.3, -0.1, 0.2, -1.0]], dtype=np.float32)

    goal_states = convert_episode_goal_states(anvil_states, native_actions)
    action_goal_abs = convert_episode_goal_abs_actions(goal_states)

    anchor = anvil_states[0]  # observation.state[-1], as EERelWorldTransform reads it
    action_rel = ee_rel_world_forward(action_goal_abs, anchor)
    recovered = ee_rel_world_inverse(action_rel, anchor)

    np.testing.assert_allclose(recovered, action_goal_abs, atol=1e-5)


def test_convert_episode_native_hand_actions_roundtrips_to_native_command():
    """The native_hand dataset column, rotated back to world per-step against
    each frame's OWN EE orientation (raw observation.state[3:6] axis-angle),
    must exactly reconstruct the source native command -- this is the eval
    rotate-back applied to the stored data, i.e. the GT-replay oracle at the
    data level. Uses raw LIBERO 8-dim states ([pos(3), axis-angle(3),
    gripper_qpos(2)]) with a clearly non-identity orientation so a wrong
    frame convention would fail."""
    raw_states = np.array(
        [
            [0.1, 0.2, 0.3, 0.2, -0.5, 0.3, 0.02, -0.02],
            [0.15, 0.18, 0.31, 0.4, 0.1, -0.2, 0.03, -0.03],
        ],
        dtype=np.float32,
    )
    native_actions = np.array(
        [
            [0.4, -0.2, 0.6, 0.3, -0.1, 0.2, -1.0],
            [-0.3, 0.5, 0.1, -0.2, 0.15, -0.05, 1.0],
        ],
        dtype=np.float32,
    )

    hand_actions = convert_episode_native_hand_actions(raw_states, native_actions)

    # native_hand must NOT be a no-op copy of native (frame genuinely applied).
    assert not np.allclose(hand_actions[:, :6], native_actions[:, :6], atol=1e-3)

    for t in range(len(native_actions)):
        recovered = hand_action_to_native(hand_actions[t], raw_states[t][3:6])
        np.testing.assert_allclose(recovered, native_actions[t], atol=1e-5)


def test_native_action_to_hand_matches_episode_helper():
    """The per-episode helper is just the pure transform applied per step."""
    raw_state = np.array([0.1, 0.2, 0.3, 0.2, -0.5, 0.3, 0.02, -0.02], dtype=np.float32)
    native = np.array([0.4, -0.2, 0.6, 0.3, -0.1, 0.2, -1.0], dtype=np.float32)
    per_step = native_action_to_hand(native, raw_state[3:6])
    episode = convert_episode_native_hand_actions(
        raw_state.reshape(1, 8), native.reshape(1, 7)
    )[0]
    np.testing.assert_allclose(episode, per_step, atol=1e-7)


# --- axis-angle "goal" family (native_abs / native_n0) ---


def test_goal_state_to_axis_angle_action_decodes_to_the_rot6d_goal_action():
    """native_abs/native_n0 store the goal rotation as AXIS-ANGLE; decoding it
    back to rot6d (axis_angle_action_to_rot6d) must reproduce EXACTLY the
    10-dim rot6d action the rot6d `goalabs` group stores
    (anvil_state_to_abs_action) — the lossless rot6d<->axis-angle swap is the
    ONLY difference between the two families, proven zero-error just like
    native_rot6d."""
    quat = matrix_to_quat(axis_angle_to_matrix(np.array([0.3, -0.5, 0.2])))
    goal = np.array([0.4, -0.1, 0.6, *quat, -1.0], dtype=np.float32)

    aa_action = goal_state_to_axis_angle_action(goal)
    assert aa_action.shape == (7,)

    decoded_rot6d = axis_angle_action_to_rot6d(aa_action)
    expected_rot6d = anvil_state_to_abs_action(goal)
    np.testing.assert_allclose(decoded_rot6d, expected_rot6d, atol=1e-6)
    # pos and gripper pass through untouched in the axis-angle layout.
    np.testing.assert_allclose(aa_action[:3], goal[:3], atol=1e-8)
    assert aa_action[6] == goal[7]


def test_native_abs_action_recovers_the_native_command_to_float_precision():
    """The native_abs identity (the whole point of the GT-replay oracle, at
    the data level): decode the stored axis-angle goal back to rot6d and
    recover a native-delta against the SAME state the goal was built from —
    this must reproduce native's own command exactly (goal = state +
    native_delta, so goal - state = native_delta)."""
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.1, -0.2, 0.05])))
    states = np.array([[0.3, -0.1, 0.5, *quat0, 0.02]], dtype=np.float32)
    native_actions = np.array([[0.4, -0.6, 0.3, 0.2, -0.4, 0.15, -1.0]], dtype=np.float32)

    goal_states = convert_episode_goal_states(states, native_actions)
    aa_actions = convert_episode_goal_aa_actions(goal_states)
    assert aa_actions.shape == (1, 7)

    rot6d10 = axis_angle_action_to_rot6d(aa_actions[0])
    recovered = recovered_delta_native_action(
        reconstructed_pos=rot6d10[:3],
        reconstructed_rot6d=rot6d10[3:9],
        reconstructed_gripper=float(rot6d10[9]),
        current_state=states[0],
        current_gripper=float(states[0][7]),
        gripper_mode="native_cmd",
    )
    expected = np.clip(native_actions[0], -1.0, 1.0)
    np.testing.assert_allclose(recovered, expected, atol=1e-5)


def test_native_n0_action_recovers_the_native_command_to_float_precision():
    """The native_n0 identity at the data level (n_action_steps=1, anchor ==
    current state): the stored action is the goal ALREADY relativized per-frame
    against its own obs pose (ee_rel_world_forward). Un-relativizing it via
    ee_rel_world_inverse against that same state recovers the absolute goal,
    and recovering a native-delta against the same state reproduces native's
    own command — proving native_n0 and native_abs collapse to the identical
    native delta when the n-0 anchor is the current frame (the GT-replay
    regime), differing only for chunked (n>1) reconstruction."""
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.1, -0.2, 0.05])))
    states = np.array([[0.3, -0.1, 0.5, *quat0, 0.02]], dtype=np.float32)
    native_actions = np.array([[0.4, -0.6, 0.3, 0.2, -0.4, 0.15, -1.0]], dtype=np.float32)

    goal_states = convert_episode_goal_states(states, native_actions)
    n0_actions = convert_episode_goal_n0_aa_actions(goal_states, states)
    assert n0_actions.shape == (1, 7)

    rel10 = axis_angle_action_to_rot6d(n0_actions[0])
    abs10 = ee_rel_world_inverse(rel10.reshape(1, 10), states[0].reshape(1, 8))[0]
    recovered = recovered_delta_native_action(
        reconstructed_pos=abs10[:3],
        reconstructed_rot6d=abs10[3:9],
        reconstructed_gripper=float(abs10[9]),
        current_state=states[0],
        current_gripper=float(states[0][7]),
        gripper_mode="native_cmd",
    )
    expected = np.clip(native_actions[0], -1.0, 1.0)
    np.testing.assert_allclose(recovered, expected, atol=1e-5)
