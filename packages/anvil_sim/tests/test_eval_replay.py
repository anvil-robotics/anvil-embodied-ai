"""Tests for the pure-math parts of anvil_sim.eval_replay (GT-action
provider + action-type mapping). The env-in-the-loop replay itself is
exercised by the bench pipeline's gt-replay gate against a live LiberoEnv,
not unit-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest
from anvil_shared.ee_transform import ee_rel_forward, ee_rel_world_forward
from anvil_shared.rotation import axis_angle_to_matrix, matrix_to_quat, matrix_to_rot6d

from anvil_sim.eval_replay import GtActionProvider
from anvil_sim.studies.libero_ee.libero_processor import AnvilEEObsProcessorStep
from anvil_sim.studies.libero_ee.replay_adapter import (
    _DIVERGENCE_POS_THRESHOLD,
    _DIVERGENCE_ROT_THRESHOLD,
    _provider_mode,
    _state_divergence,
    build_replay_adapter,
)


@pytest.mark.parametrize(
    ("action_type", "expected"),
    [
        # direct: stored form IS the per-step policy-output form
        ("native", "direct"),
        ("native_rot6d", "direct"),
        ("native_hand", "direct"),
        ("native_abs", "direct"),
        ("native_n0", "direct"),
        ("zerocal_goal_abs", "direct"),
        # n-0 relative: stored is ABSOLUTE, policy output is anchor-relative
        ("zerocal_goal_hand_n0", "rel_hand"),
        ("zerocal_goal_world_n0", "rel_world"),
    ],
)
def test_provider_mode_mapping(action_type, expected):
    assert _provider_mode(action_type) == expected


def test_provider_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported"):
        _provider_mode("bogus_type")


def test_direct_provider_passes_through():
    provider = GtActionProvider(mode="direct")
    stored = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(provider(stored), stored)


def _obs_step_with_state(state8: np.ndarray) -> AnvilEEObsProcessorStep:
    step = AnvilEEObsProcessorStep(action_type="ee_abs")
    step.last_anvil_state = state8
    return step


def _abs10(pos, aa, gripper):
    return np.concatenate(
        [pos, matrix_to_rot6d(axis_angle_to_matrix(np.asarray(aa))), [gripper]]
    ).astype(np.float32)


@pytest.mark.parametrize(
    ("mode", "forward"),
    [("rel_hand", ee_rel_forward), ("rel_world", ee_rel_world_forward)],
)
def test_rel_provider_matches_forward_transform_against_live_anchor(mode, forward):
    """A perfect policy trained on n-0 relativized targets outputs the
    forward transform of the absolute GT against the chunk anchor — the
    provider must reproduce exactly that, reading the anchor from the live
    obs step just like the paired action processor does."""
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.05, -0.1, 0.2])))
    live_state = np.array([0.3, -0.1, 0.5, *quat0, 0.02], dtype=np.float32)
    obs_step = _obs_step_with_state(live_state)

    stored_abs = _abs10([0.35, -0.05, 0.55], [0.1, 0.0, -0.05], 0.03)
    provider = GtActionProvider(mode=mode, obs_step=obs_step, n_action_steps=1)

    provided = provider(stored_abs)
    expected = forward(stored_abs.reshape(1, 10), live_state.reshape(1, 8))[0]
    np.testing.assert_allclose(provided, expected, atol=1e-6)


def test_rel_provider_anchors_at_chunk_start_not_every_step():
    """With n_action_steps=2, calls 0 and 1 must use the SAME anchor (the
    live state at call 0), and call 2 must re-anchor — mirroring the action
    processor's chunk state machine."""
    quat_a = matrix_to_quat(np.eye(3))
    state_a = np.array([0.0, 0.0, 0.0, *quat_a, 0.02], dtype=np.float32)
    state_b = np.array([0.5, 0.5, 0.5, *quat_a, 0.02], dtype=np.float32)
    obs_step = _obs_step_with_state(state_a)

    stored = _abs10([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.02)
    provider = GtActionProvider(mode="rel_world", obs_step=obs_step, n_action_steps=2)

    out0 = provider(stored)  # chunk start: anchor = state_a
    obs_step.last_anvil_state = state_b  # live state moves...
    out1 = provider(stored)  # ...but anchor must STILL be state_a
    np.testing.assert_allclose(out0, out1, atol=1e-6)

    out2 = provider(stored)  # new chunk: anchor re-captured = state_b
    expected2 = ee_rel_world_forward(stored.reshape(1, 10), state_b.reshape(1, 8))[0]
    np.testing.assert_allclose(out2, expected2, atol=1e-6)
    assert not np.allclose(out2, out0)


def test_rel_provider_requires_observation_first():
    obs_step = AnvilEEObsProcessorStep(action_type="ee_abs")  # no obs processed yet
    provider = GtActionProvider(mode="rel_world", obs_step=obs_step, n_action_steps=1)
    with pytest.raises(RuntimeError, match="observation"):
        provider(_abs10([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.02))


# --- state_divergence: moved (behavior-preserving) from an inline
# eval_replay.replay() computation into the study ReplayAdapter, so the
# generic harness no longer hardcodes the EE-encoding layout. These tests
# lock down that the extraction changed nothing.


def test_state_divergence_zero_when_states_match():
    quat0 = matrix_to_quat(axis_angle_to_matrix(np.array([0.1, -0.05, 0.2])))
    demo_state = np.array([0.3, -0.1, 0.5, 0.1, -0.05, 0.2, 0.02], dtype=np.float32)
    actual_state8 = np.array([0.3, -0.1, 0.5, *quat0, 0.02], dtype=np.float32)

    pos_err, rot_err = _state_divergence(demo_state, actual_state8)
    assert pos_err == pytest.approx(0.0, abs=1e-6)
    assert rot_err == pytest.approx(0.0, abs=1e-6)


def test_state_divergence_reports_real_position_and_rotation_offsets():
    quat0 = matrix_to_quat(np.eye(3))  # actual: identity rotation
    demo_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.02], dtype=np.float32)  # 90deg about z
    actual_state8 = np.array([0.01, 0.0, 0.0, *quat0, 0.02], dtype=np.float32)  # 1cm off, no rotation

    pos_err, rot_err = _state_divergence(demo_state, actual_state8)
    assert pos_err == pytest.approx(0.01, abs=1e-6)
    assert rot_err == pytest.approx(np.pi / 2, abs=1e-6)


def test_replay_adapter_thresholds_are_robosuite_osc_output_max():
    """The 'notable divergence' bar reuses robosuite OSC_POSE's own
    output_max (see libero_convert.py) — a principled scale, not arbitrary."""
    from anvil_sim.studies.libero_ee.libero_convert import OSC_OUTPUT_MAX_POS, OSC_OUTPUT_MAX_ROT

    assert _DIVERGENCE_POS_THRESHOLD == OSC_OUTPUT_MAX_POS
    assert _DIVERGENCE_ROT_THRESHOLD == OSC_OUTPUT_MAX_ROT

    adapter = build_replay_adapter()
    assert adapter.divergence_pos_threshold == OSC_OUTPUT_MAX_POS
    assert adapter.divergence_rot_threshold == OSC_OUTPUT_MAX_ROT
    assert adapter.state_divergence is _state_divergence
