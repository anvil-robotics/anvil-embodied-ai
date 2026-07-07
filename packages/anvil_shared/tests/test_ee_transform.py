"""Tests for the experimental world-frame SE(3) relative transform
(ee_rel_world_forward/inverse) -- added to isolate whether UMI's body-frame
translation choice (used by the production ee_rel_forward/inverse) explains
part of ee_rel's lower closed-loop performance in the anvil-sim LIBERO
benchmark. NOT part of the ee_abs/ee_rel production contract.
"""

import numpy as np

from anvil_shared.ee_transform import (
    ee_obs_rel_forward,
    ee_obs_rel_world_forward,
    ee_rel_world_forward,
    ee_rel_world_inverse,
)
from anvil_shared.rotation import axis_angle_to_matrix, matrix_to_quat, matrix_to_rot6d


def _state(pos, quat, gripper=0.02):
    return np.concatenate([pos, quat, [gripper]]).astype(np.float64)


def _abs_action(pos, rot6d, gripper=0.02):
    return np.concatenate([pos, rot6d, [gripper]]).astype(np.float64)


def test_roundtrip_single_state():
    state = _state(
        np.array([0.1, 0.2, 0.3]),
        matrix_to_quat(axis_angle_to_matrix(np.array([0.3, -0.1, 0.2]))),
    )
    action_abs = _abs_action(
        np.array([0.15, 0.18, 0.28]),
        matrix_to_rot6d(axis_angle_to_matrix(np.array([0.35, -0.05, 0.25]))),
        gripper=0.03,
    )

    rel = ee_rel_world_forward(action_abs, state)
    recovered = ee_rel_world_inverse(rel, state)
    np.testing.assert_allclose(recovered, action_abs, atol=1e-8)


def test_translation_is_world_frame_not_body_frame():
    """Regression guard: rotating the CURRENT orientation must NOT change the
    translation component of the relative action -- unlike ee_rel_forward
    (UMI-style), which explicitly projects translation into the body frame
    via R_state.T @ world_delta."""
    pos = np.zeros(3)
    target_pos = np.array([0.01, 0.0, 0.0])
    rot6d = matrix_to_rot6d(np.eye(3))
    identity_quat = matrix_to_quat(np.eye(3))
    rotated_quat = matrix_to_quat(axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2])))

    action_abs = _abs_action(target_pos, rot6d)
    rel_identity = ee_rel_world_forward(action_abs, _state(pos, identity_quat))
    rel_rotated = ee_rel_world_forward(action_abs, _state(pos, rotated_quat))

    np.testing.assert_allclose(rel_identity[:3], rel_rotated[:3], atol=1e-10)
    np.testing.assert_allclose(rel_identity[:3], target_pos, atol=1e-10)  # no projection at all


def test_rotation_composition_matches_world_frame_convention():
    """delta_rot6d must decode to R_target @ R_current.T (world-frame
    convention, matching native_action_from_targets/native_action_from_world_delta),
    NOT R_current.T @ R_target (ee_rel_forward's body-frame convention)."""
    R_current = axis_angle_to_matrix(np.array([0.3, -0.1, 0.2]))
    R_target = axis_angle_to_matrix(np.array([0.35, -0.05, 0.25]))
    state = _state(np.zeros(3), matrix_to_quat(R_current))
    action_abs = _abs_action(np.zeros(3), matrix_to_rot6d(R_target))

    rel = ee_rel_world_forward(action_abs, state)
    from anvil_shared.rotation import rot6d_to_matrix

    R_rel = rot6d_to_matrix(rel[3:9])
    expected_world = R_target @ R_current.T
    np.testing.assert_allclose(R_rel, expected_world, atol=1e-8)

    # Sanity: must NOT match the body-frame convention instead.
    expected_body = R_current.T @ R_target
    assert not np.allclose(R_rel, expected_body, atol=1e-4)


def test_batched_per_sample_state_matches_single_calls():
    """Vectorised (per-sample-state) path must match looping single calls."""
    n = 5
    rng = np.random.default_rng(0)
    states = np.stack(
        [
            _state(rng.normal(size=3) * 0.1, matrix_to_quat(axis_angle_to_matrix(rng.normal(size=3) * 0.2)))
            for _ in range(n)
        ]
    )
    actions = np.stack(
        [
            _abs_action(
                rng.normal(size=3) * 0.1, matrix_to_rot6d(axis_angle_to_matrix(rng.normal(size=3) * 0.2))
            )
            for _ in range(n)
        ]
    )

    batched_rel = ee_rel_world_forward(actions, states)
    looped_rel = np.stack([ee_rel_world_forward(actions[i], states[i]) for i in range(n)])
    np.testing.assert_allclose(batched_rel, looped_rel, atol=1e-8)

    batched_abs = ee_rel_world_inverse(batched_rel, states)
    np.testing.assert_allclose(batched_abs, actions, atol=1e-6)


def test_obs_rel_world_matches_body_frame_when_self_anchored():
    """When obs == anchor (the eval-time self-anchored case), world-frame and
    body-frame obs relativisation must be IDENTICAL (both reduce to zero
    translation + identity rotation) -- this is why AnvilEEObsProcessorStep
    doesn't need a separate world-frame obs encoding for single-step obs."""
    state = _state(
        np.array([0.1, -0.2, 0.3]),
        matrix_to_quat(axis_angle_to_matrix(np.array([0.3, -0.1, 0.2]))),
    )
    world_rel = ee_obs_rel_world_forward(state, state)
    body_rel = ee_obs_rel_forward(state, state)
    np.testing.assert_allclose(world_rel, body_rel, atol=1e-10)
    np.testing.assert_allclose(world_rel[:3], 0.0, atol=1e-10)


def test_obs_rel_world_differs_from_body_frame_for_distinct_states():
    """For genuinely distinct obs/anchor (the n_obs_steps>1 multi-step case),
    world-frame and body-frame obs relativisation must differ -- this is why
    Diffusion's stats computation needs the world-frame variant."""
    anchor = _state(np.zeros(3), matrix_to_quat(axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2]))))
    obs = _state(np.array([0.05, 0.0, 0.0]), matrix_to_quat(np.eye(3)))

    world_rel = ee_obs_rel_world_forward(obs, anchor)
    body_rel = ee_obs_rel_forward(obs, anchor)
    assert not np.allclose(world_rel[:3], body_rel[:3], atol=1e-4)
