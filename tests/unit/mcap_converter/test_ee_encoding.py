"""Unit tests for EE Cartesian-space encoding in the mcap_converter.

Tests the headline outputs of the EE feature:
- _align_ee_signals: output shapes, rot6d slot values, per-arm concat order
- _define_features (writer): EE feature schema names/shapes
- gripper propagation: identical in state and action
- insertion-order contract: observation_topics order = concat order
- action_encoding="delta": baked per-frame Delta(n->n+1) action (forward-looking,
  finalized via _finalize_pending_action), observation.state unaffected, config
  validation
- observation_encoding="quaternion"|"rot6d"|"axis_angle": observation.state
  rotation representation, independent of action_encoding
- strict=True/False unrecognized-key handling; the "relative" reserved value
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.schema import ConfigurationError
from mcap_converter.core.extractor import BufferedStreamExtractor
from mcap_converter.core.writer import LeRobotWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ee_config(
    arms: dict[str, str],
    action_encoding: str = "absolute",
    observation_encoding: str = "quaternion",
) -> object:
    """Build a minimal EE DataConfig with the given {arm_id: topic} map."""
    return ConfigLoader.from_dict({
        "data_space": "ee",
        "action_encoding": action_encoding,
        "observation_encoding": observation_encoding,
        "observation_topics": arms,
        "action_topics": {},
        "camera_topics": ["/cam_chest/image_raw/compressed"],
        "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        "image_resolution": [640, 480],
    })


def _make_ee_buffer(pos, quat, gripper, ts=0.0) -> deque:
    """One-sample EE buffer (timestamp, pos, quat, gripper)."""
    buf = deque()
    buf.append((ts, np.asarray(pos, dtype=np.float64),
                    np.asarray(quat, dtype=np.float64),
                    float(gripper)))
    return buf


def _identity_quat():
    return [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# _align_ee_signals — left-only
# ---------------------------------------------------------------------------


class TestAlignEESignals:
    """_align_ee_signals only computes observation.state + action_abs_own (this
    frame's own pose in action representation) — it no longer bakes "action"
    itself, since action[t] = observation[t+1] needs the NEXT frame's pose,
    which extract_frames()'s 1-frame lookahead supplies (see
    TestAlignEESignalsDeltaEncoding / TestActFromObsLookahead below). These
    tests check action_abs_own's per-frame math (rot6d encoding, gripper
    passthrough, insertion order) independent of that lookahead wiring.
    """

    def test_left_only_shapes(self):
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], _identity_quat(), 0.02)}
        out, state_quat, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert out is not None
        assert "action" not in out
        assert out["observation.state"].shape == (8,)
        assert action_abs_own.shape == (10,)
        assert state_quat.shape == (8,)

    def test_left_only_state_layout(self):
        """State = [xyz, qx, qy, qz, qw, gripper] for identity rotation."""
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        pos, quat, g = [0.1, 0.2, 0.3], _identity_quat(), 0.025
        ee_buffers = {"left": _make_ee_buffer(pos, quat, g)}
        out, _, _ = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        state = out["observation.state"]
        np.testing.assert_allclose(state[:3], pos, atol=1e-7)
        np.testing.assert_allclose(state[3:7], quat, atol=1e-7)  # xyzw
        np.testing.assert_allclose(state[7], g, atol=1e-7)

    def test_left_only_action_rot6d_identity(self):
        """Identity quaternion → rot6d = first two columns of I = [1,0,0, 0,1,0]."""
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], _identity_quat(), 0.01)}
        _, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        np.testing.assert_allclose(action_abs_own[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)

    def test_left_only_gripper_matches_state(self):
        """Gripper slot in state == gripper slot in action_abs_own."""
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        ee_buffers = {"left": _make_ee_buffer([0.0, 0.0, 0.3], _identity_quat(), 0.034)}
        out, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert np.isclose(out["observation.state"][7], action_abs_own[9])

    def test_left_only_xyz_matches_state_and_action(self):
        """xyz is identical in both state and action_abs_own."""
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        pos = [0.45, -0.12, 0.61]
        ee_buffers = {"left": _make_ee_buffer(pos, _identity_quat(), 0.0)}
        out, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        np.testing.assert_allclose(out["observation.state"][:3], pos, atol=1e-7)
        np.testing.assert_allclose(action_abs_own[:3], pos, atol=1e-7)

    def test_bimanual_shapes(self):
        cfg = _make_ee_config({"left": "/ee_pose_left", "right": "/ee_pose_right"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        ee_buffers = {
            "left":  _make_ee_buffer([0.1, 0.2, 0.3], _identity_quat(), 0.01),
            "right": _make_ee_buffer([0.4, 0.5, 0.6], _identity_quat(), 0.02),
        }
        out, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert out["observation.state"].shape == (16,)
        assert action_abs_own.shape == (20,)

    def test_bimanual_concat_insertion_order(self):
        """Concat order = observation_topics insertion order (left, right)."""
        cfg = _make_ee_config({"left": "/ee_pose_left", "right": "/ee_pose_right"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        pos_l, pos_r = [0.1, 0.0, 0.0], [0.5, 0.0, 0.0]
        ee_buffers = {
            "left":  _make_ee_buffer(pos_l, _identity_quat(), 0.01),
            "right": _make_ee_buffer(pos_r, _identity_quat(), 0.02),
        }
        out, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        # Left arm occupies indices 0-7 (state) and 0-9 (action)
        np.testing.assert_allclose(out["observation.state"][:3], pos_l, atol=1e-7)
        np.testing.assert_allclose(out["observation.state"][8:11], pos_r, atol=1e-7)
        np.testing.assert_allclose(action_abs_own[:3], pos_l, atol=1e-7)
        np.testing.assert_allclose(action_abs_own[10:13], pos_r, atol=1e-7)

    def test_reversed_order_right_then_left(self):
        """If config lists right then left, right occupies the first slots."""
        cfg = _make_ee_config({"right": "/ee_pose_right", "left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        pos_r, pos_l = [0.5, 0.0, 0.0], [0.1, 0.0, 0.0]
        ee_buffers = {
            "right": _make_ee_buffer(pos_r, _identity_quat(), 0.02),
            "left":  _make_ee_buffer(pos_l, _identity_quat(), 0.01),
        }
        out, _, _ = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        # Right is listed first → occupies indices 0-7
        np.testing.assert_allclose(out["observation.state"][:3], pos_r, atol=1e-7)
        np.testing.assert_allclose(out["observation.state"][8:11], pos_l, atol=1e-7)

    def test_missing_arm_returns_none(self):
        """If one arm's buffer is empty, return None (skip frame) — not a 3-tuple of
        Nones; the whole call returns bare None on failure."""
        cfg = _make_ee_config({"left": "/ee_pose_left", "right": "/ee_pose_right"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        ee_buffers = {
            "left":  _make_ee_buffer([0.0, 0.0, 0.0], _identity_quat(), 0.0),
            "right": deque(),  # empty
        }
        assert ext._align_ee_signals(ee_buffers, target_ts=0.0) is None

    def test_rot6d_non_identity(self):
        """A 90-degree rotation about Z → known rot6d."""
        from anvil_shared.rotation import quat_to_matrix, matrix_to_rot6d
        # 90° about Z: quat = [0, 0, sin45, cos45]
        s = np.sin(np.pi / 4)
        quat_90z = [0.0, 0.0, s, s]
        expected_rot6d = matrix_to_rot6d(quat_to_matrix(quat_90z))

        cfg = _make_ee_config({"left": "/ee_pose_left"})
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
        ee_buffers = {"left": _make_ee_buffer([0.0, 0.0, 0.5], quat_90z, 0.0)}
        _, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        np.testing.assert_allclose(action_abs_own[3:9], expected_rot6d, atol=1e-6)


# ---------------------------------------------------------------------------
# _define_features (writer) — EE feature schema
# ---------------------------------------------------------------------------


class TestWriterEEFeatures:
    def _writer(self, arms, **kwargs):
        cfg = _make_ee_config(arms, **kwargs)
        return LeRobotWriter(output_dir="/tmp/_test_ee", repo_id="r/x",
                             config=cfg, quiet=True)

    def test_left_only_shapes(self):
        feats = self._writer({"left": "/ee_pose_left"})._define_features({}, ["chest"])
        assert feats["observation.state"]["shape"] == (8,)
        assert feats["action"]["shape"] == (10,)

    def test_bimanual_shapes(self):
        feats = self._writer({"left": "/ee_pose_left", "right": "/ee_pose_right"})._define_features({}, ["chest"])
        assert feats["observation.state"]["shape"] == (16,)
        assert feats["action"]["shape"] == (20,)

    def test_state_names_layout(self):
        feats = self._writer({"left": "/ee_pose_left"})._define_features({}, ["chest"])
        assert feats["observation.state"]["names"] == [
            "left_x", "left_y", "left_z",
            "left_qx", "left_qy", "left_qz", "left_qw",
            "left_gripper",
        ]

    def test_action_names_layout(self):
        feats = self._writer({"left": "/ee_pose_left"})._define_features({}, ["chest"])
        assert feats["action"]["names"] == [
            "left_x", "left_y", "left_z",
            "left_r0", "left_r1", "left_r2", "left_r3", "left_r4", "left_r5",
            "left_gripper",
        ]

    def test_bimanual_names_insertion_order(self):
        """right then left → right names first."""
        feats = self._writer({"right": "/ee_pose_right", "left": "/ee_pose_left"})._define_features({}, ["chest"])
        names = feats["observation.state"]["names"]
        assert names[0] == "right_x"
        assert names[8] == "left_x"

    def test_no_velocity_effort_in_ee_mode(self):
        feats = self._writer({"left": "/ee_pose_left"})._define_features({}, ["chest"])
        assert "observation.velocity" not in feats
        assert "observation.effort" not in feats

    def test_rot6d_observation_encoding_shape_and_names(self):
        feats = self._writer(
            {"left": "/ee_pose_left"}, observation_encoding="rot6d"
        )._define_features({}, ["chest"])
        assert feats["observation.state"]["shape"] == (10,)
        assert feats["observation.state"]["names"] == [
            "left_x", "left_y", "left_z",
            "left_r0", "left_r1", "left_r2", "left_r3", "left_r4", "left_r5",
            "left_gripper",
        ]
        # action is unaffected — always rot6d regardless of observation_encoding.
        assert feats["action"]["shape"] == (10,)

    def test_axis_angle_observation_encoding_shape_and_names(self):
        feats = self._writer(
            {"left": "/ee_pose_left"}, observation_encoding="axis_angle"
        )._define_features({}, ["chest"])
        assert feats["observation.state"]["shape"] == (7,)
        assert feats["observation.state"]["names"] == [
            "left_x", "left_y", "left_z",
            "left_ax", "left_ay", "left_az",
            "left_gripper",
        ]


# ---------------------------------------------------------------------------
# action_encoding="delta" — baked per-frame Delta(n->n+1)
# ---------------------------------------------------------------------------


class TestActionEncodingConfig:
    def test_default_is_absolute(self):
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        assert cfg.action_encoding == "absolute"
        assert cfg.is_action_delta is False

    def test_output_subdir_ee_absolute(self):
        cfg = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="absolute")
        assert cfg.output_subdir == "ee-abs"

    def test_output_subdir_joint(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "joint",
            "observation_topics": {"left": "/joint_states"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        assert cfg.output_subdir == "joint-abs"

    def test_delta_flag_sets_is_action_delta(self):
        cfg = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="delta")
        assert cfg.action_encoding == "delta"
        assert cfg.is_action_delta is True
        assert cfg.output_subdir == "ee-delta"

    def test_invalid_encoding_rejected_by_validate(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "action_encoding": "bogus",
            "observation_topics": {"left": "/ee_pose_left"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        with pytest.raises(ConfigurationError, match="action_encoding"):
            cfg.validate()

    def test_relative_is_reserved_not_implemented(self):
        """'relative' is a structurally-valid value (accepted by the loader) but must be
        rejected at validate() time with a message distinguishing it from a typo."""
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "action_encoding": "relative",
            "observation_topics": {"left": "/ee_pose_left"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        assert cfg.action_encoding == "relative"  # loader accepts it structurally
        with pytest.raises(ConfigurationError, match="reserved for future use"):
            cfg.validate()

    def test_delta_encoding_rejected_in_joint_mode(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "joint",
            "action_encoding": "delta",
            "observation_topics": {"left": "/joint_states"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        with pytest.raises(ConfigurationError, match="action_encoding"):
            cfg.validate()

    def test_existing_absolute_ee_config_unaffected(self):
        """Existing configs that don't set action_encoding at all must
        default to byte-identical absolute behavior — no silent change."""
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "observation_topics": {"left": "/ee_pose_left"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        assert cfg.action_encoding == "absolute"
        assert cfg.is_action_delta is False
        cfg.validate()  # must not raise


class TestAlignEESignalsDeltaEncoding:
    """action_encoding="delta" is now finalized by _finalize_pending_action, using
    the CURRENT frame's own state_quat as anchor and the NEXT frame's
    action_abs_own as target — a forward-looking, single-frame transform
    (t -> t+1), not the old backward-looking (t-1 -> t) self-anchor design.
    There is no more "first frame self-anchor, zero delta" special case —
    every frame with a successor gets a real delta; the episode's LAST frame
    (no successor) is dropped entirely by extract_frames()'s lookahead, not
    handled here (see TestActFromObsLookahead)."""

    def test_delta_matches_ee_delta_forward_directly(self):
        """The finalized action must equal calling ee_delta_forward(next_action_abs,
        anchor=this_frame's_own_state_quat) directly — cross-checked independently
        of _finalize_pending_action's internals."""
        from anvil_shared.ee_transform import ee_delta_forward

        cfg = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="delta")
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        cur_pos, cur_quat, cur_grip = [0.1, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0], 0.01
        s = np.sin(np.pi / 4)
        next_pos, next_quat, next_grip = [0.15, 0.05, 0.32], [0.0, 0.0, s, s], 0.015

        ee_buffers_cur = {"left": _make_ee_buffer(cur_pos, cur_quat, cur_grip)}
        ee_buffers_next = {"left": _make_ee_buffer(next_pos, next_quat, next_grip)}

        frame_cur, state_quat_cur, _ = ext._align_ee_signals(ee_buffers_cur, target_ts=0.0)
        _, _, action_abs_own_next = ext._align_ee_signals(ee_buffers_next, target_ts=1.0)

        ext._finalize_pending_action(frame_cur, state_quat_cur, action_abs_own_next)

        expected = ee_delta_forward(
            action_abs_own_next.astype(np.float64), state_quat_cur.astype(np.float64)
        )
        np.testing.assert_allclose(frame_cur["action"], expected, atol=1e-5)

    def test_observation_state_unaffected_by_encoding(self):
        """observation.state must be byte-identical between absolute and delta
        encoding for the same input — only `action` differs (and only once finalized)."""
        pos, quat, g = [0.2, -0.1, 0.4], [0.0, 0.0, 0.0, 1.0], 0.02
        ee_buffers_abs = {"left": _make_ee_buffer(pos, quat, g)}
        ee_buffers_delta = {"left": _make_ee_buffer(pos, quat, g)}

        cfg_abs = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="absolute")
        cfg_delta = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="delta")
        out_abs, _, _ = BufferedStreamExtractor(cfg_abs, fps=30, quiet=True)._align_ee_signals(
            ee_buffers_abs, target_ts=0.0
        )
        out_delta, _, _ = BufferedStreamExtractor(cfg_delta, fps=30, quiet=True)._align_ee_signals(
            ee_buffers_delta, target_ts=0.0
        )

        np.testing.assert_allclose(
            out_abs["observation.state"], out_delta["observation.state"], atol=1e-7
        )

    def test_absolute_mode_action_is_next_frame_pose_unchanged(self):
        """Under absolute encoding, _finalize_pending_action must use the next
        frame's action_abs_own as-is (no delta transform applied)."""
        cfg = _make_ee_config({"left": "/ee_pose_left"}, action_encoding="absolute")
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)

        pos, quat, g = [0.3, 0.1, 0.2], _identity_quat(), 0.03
        ee_buffers_cur = {"left": _make_ee_buffer(pos, quat, g)}
        next_pos = [0.31, 0.11, 0.21]
        ee_buffers_next = {"left": _make_ee_buffer(next_pos, quat, g)}

        frame_cur, state_quat_cur, _ = ext._align_ee_signals(ee_buffers_cur, target_ts=0.0)
        _, _, action_abs_own_next = ext._align_ee_signals(ee_buffers_next, target_ts=1.0)

        ext._finalize_pending_action(frame_cur, state_quat_cur, action_abs_own_next)

        np.testing.assert_allclose(frame_cur["action"], action_abs_own_next, atol=1e-7)

    def test_delta_anchor_stays_quaternion_regardless_of_observation_encoding(self):
        """The critical interaction this refactor introduced: action_encoding="delta"
        must produce the IDENTICAL baked action column regardless of observation_encoding
        — the delta anchor is always the quaternion-encoded state_quat, never whatever
        rotation representation observation.state happens to be written as on disk."""
        cur_pos, cur_quat, cur_grip = [0.1, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0], 0.01
        next_pos, next_quat, next_grip = [0.2, 0.1, 0.25], [0.0, 0.7071068, 0.0, 0.7071068], 0.02

        results = {}
        for obs_enc in ("quaternion", "rot6d", "axis_angle"):
            cfg = _make_ee_config(
                {"left": "/ee_pose_left"},
                action_encoding="delta",
                observation_encoding=obs_enc,
            )
            ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
            ee_buffers_cur = {"left": _make_ee_buffer(cur_pos, cur_quat, cur_grip)}
            ee_buffers_next = {"left": _make_ee_buffer(next_pos, next_quat, next_grip)}

            frame_cur, state_quat_cur, _ = ext._align_ee_signals(ee_buffers_cur, target_ts=0.0)
            _, _, action_abs_own_next = ext._align_ee_signals(ee_buffers_next, target_ts=1.0)
            ext._finalize_pending_action(frame_cur, state_quat_cur, action_abs_own_next)

            results[obs_enc] = frame_cur["action"]
            # state_quat (the anchor) is always quaternion, regardless of what
            # observation_encoding selected for the on-disk observation.state.
            np.testing.assert_allclose(state_quat_cur[3:7], cur_quat, atol=1e-6)

        np.testing.assert_allclose(results["quaternion"], results["rot6d"], atol=1e-6)
        np.testing.assert_allclose(results["quaternion"], results["axis_angle"], atol=1e-6)

    def test_state_quat_thread_is_quaternion_shaped_even_with_non_quaternion_obs(self):
        """state_quat returned for use as the delta anchor must always be 8-dim
        quaternion-per-arm, even when observation.state itself is rot6d/axis_angle
        (10/7-dim) — proves the two are genuinely decoupled, not just numerically equal
        by coincidence in the quaternion case."""
        cfg = _make_ee_config(
            {"left": "/ee_pose_left"}, action_encoding="delta", observation_encoding="rot6d"
        )
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
        ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], _identity_quat(), 0.01)}
        out, state_quat, _ = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert out["observation.state"].shape == (10,)  # rot6d on disk
        assert state_quat.shape == (8,)  # anchor is always quaternion


# ---------------------------------------------------------------------------
# observation_encoding — quaternion / rot6d / axis_angle
# ---------------------------------------------------------------------------


class TestObservationEncoding:
    def test_default_is_quaternion(self):
        cfg = _make_ee_config({"left": "/ee_pose_left"})
        assert cfg.observation_encoding == "quaternion"

    def test_rot6d_state_matches_direct_conversion(self):
        from anvil_shared.rotation import quat_to_matrix, matrix_to_rot6d

        s = np.sin(np.pi / 4)
        quat = [0.0, 0.0, s, s]
        expected_rot6d = matrix_to_rot6d(quat_to_matrix(quat))

        cfg = _make_ee_config({"left": "/ee_pose_left"}, observation_encoding="rot6d")
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
        ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], quat, 0.05)}
        out, _, _ = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert out["observation.state"].shape == (10,)
        np.testing.assert_allclose(out["observation.state"][3:9], expected_rot6d, atol=1e-6)
        np.testing.assert_allclose(out["observation.state"][9], 0.05, atol=1e-7)

    def test_axis_angle_state_matches_direct_conversion(self):
        from anvil_shared.rotation import quat_to_matrix, matrix_to_axis_angle

        s = np.sin(np.pi / 4)
        quat = [0.0, 0.0, s, s]
        expected_aa = matrix_to_axis_angle(quat_to_matrix(quat))

        cfg = _make_ee_config({"left": "/ee_pose_left"}, observation_encoding="axis_angle")
        ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
        ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], quat, 0.05)}
        out, _, _ = ext._align_ee_signals(ee_buffers, target_ts=0.0)

        assert out["observation.state"].shape == (7,)
        np.testing.assert_allclose(out["observation.state"][3:6], expected_aa, atol=1e-6)
        np.testing.assert_allclose(out["observation.state"][6], 0.05, atol=1e-7)

    def test_action_always_rot6d_regardless_of_observation_encoding(self):
        """action_abs_own is an independent knob from observation_encoding — always rot6d."""
        s = np.sin(np.pi / 4)
        quat = [0.0, 0.0, s, s]
        for obs_enc in ("quaternion", "rot6d", "axis_angle"):
            cfg = _make_ee_config({"left": "/ee_pose_left"}, observation_encoding=obs_enc)
            ext = BufferedStreamExtractor(cfg, fps=30, quiet=True)
            ee_buffers = {"left": _make_ee_buffer([0.1, 0.2, 0.3], quat, 0.05)}
            _, _, action_abs_own = ext._align_ee_signals(ee_buffers, target_ts=0.0)
            assert action_abs_own.shape == (10,)

    def test_invalid_observation_encoding_rejected_by_validate(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "observation_encoding": "euler",
            "observation_topics": {"left": "/ee_pose_left"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        with pytest.raises(ConfigurationError, match="observation_encoding"):
            cfg.validate()

    def test_non_quaternion_observation_encoding_rejected_in_joint_mode(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "joint",
            "observation_encoding": "rot6d",
            "observation_topics": {"left": "/joint_states"},
            "camera_topics": ["/cam_chest/image_raw/compressed"],
            "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
        })
        with pytest.raises(ConfigurationError, match="observation_encoding"):
            cfg.validate()


# ---------------------------------------------------------------------------
# strict=True/False unrecognized-key handling
# ---------------------------------------------------------------------------


class TestStrictLenientLoading:
    # Pre-unification legacy key (`robot_state_topic`, singular) alongside otherwise
    # well-shaped current-schema fields — an unrecognized TOP-LEVEL key with no other
    # shape problem, isolating exactly what strict/lenient governs (a malformed VALUE for
    # a still-recognized key, e.g. old topic-keyed action_topics, is a structural parsing
    # error independent of strict/lenient — not what this test targets).
    _LEGACY_SHAPE = {
        "robot_state_topic": "/joint_states",
        "data_space": "joint",
        "observation_topics": {"left": "/joint_states"},
        "camera_topics": ["/cam_chest/image_raw/compressed"],
        "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
    }

    def test_unrecognized_key_rejected_under_strict(self):
        with pytest.raises(ConfigurationError, match="Unrecognized"):
            ConfigLoader.from_dict(dict(self._LEGACY_SHAPE), strict=True)

    def test_unrecognized_key_tolerated_under_lenient(self):
        """Must not raise — this is exactly the regression GT-replay/debug-plot depend on
        when reading a dataset converted before this refactor."""
        cfg = ConfigLoader.from_dict(dict(self._LEGACY_SHAPE), strict=False)
        assert cfg.observation_topics == {"left": "/joint_states"}
        assert cfg.data_space == "joint"

    def test_typo_key_rejected_under_strict_with_offending_name_in_message(self):
        with pytest.raises(ConfigurationError, match="some_typo_field"):
            ConfigLoader.from_dict({
                "data_space": "ee",
                "observation_topics": {"left": "/ee_pose_left"},
                "camera_topics": ["/cam_chest/image_raw/compressed"],
                "camera_topic_mapping": {"/cam_chest/image_raw/compressed": "chest"},
                "some_typo_field": "oops",
            }, strict=True)
