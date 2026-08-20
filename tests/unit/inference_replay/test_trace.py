"""Loading and channel-alignment behaviour for inference_data.csv traces."""

from __future__ import annotations

import numpy as np
import pytest

from inference_replay.trace import (
    ARM_CHANNELS,
    GRIPPER_CHANNELS,
    MODEL_TO_CTRL,
    N_CHANNELS,
    NAMES,
    URDF_JOINT_NAMES,
    TraceAlignmentError,
    load_trace,
    undersampling_warning,
)

from .conftest import HZ, smooth_motion, write_trace


class TestChannelLayout:
    def test_permutation_is_a_bijection(self):
        assert sorted(MODEL_TO_CTRL.tolist()) == list(range(N_CHANNELS))

    def test_permutation_puts_grippers_on_the_gripper_channels(self):
        # Controller order is (joint1..7, finger) per arm, so the grippers live at CSV indices
        # 7 and 15 and must land on model channels 0 and 8.
        assert MODEL_TO_CTRL[0] == 7
        assert MODEL_TO_CTRL[8] == 15

    def test_names_describe_model_order(self):
        assert len(NAMES) == N_CHANNELS
        assert NAMES[0] == "left_finger_joint1"
        assert NAMES[8] == "right_finger_joint1"
        assert NAMES[1:8] == [f"left_joint{i}" for i in range(1, 8)]

    def test_arm_and_gripper_channels_partition_the_layout(self):
        assert set(ARM_CHANNELS) | set(GRIPPER_CHANNELS) == set(range(N_CHANNELS))
        assert not set(ARM_CHANNELS) & set(GRIPPER_CHANNELS)

    def test_urdf_names_translate_the_monitor_vocabulary(self):
        # The monitor says left/right; the vendored URDF says follower_l/follower_r.
        assert URDF_JOINT_NAMES[0] == "follower_l_finger_joint1"
        assert URDF_JOINT_NAMES[1] == "follower_l_joint1"
        assert URDF_JOINT_NAMES[8] == "follower_r_finger_joint1"
        assert len(URDF_JOINT_NAMES) == N_CHANNELS
        assert len(set(URDF_JOINT_NAMES)) == N_CHANNELS


class TestLoad:
    def test_loads_a_valid_trace(self, good_trace_path):
        trace = load_trace(good_trace_path)
        assert trace.n == 120
        assert trace.obs.shape == (120, N_CHANNELS)
        assert trace.action_type == "absolute"
        assert trace.hz == pytest.approx(HZ, rel=0.05)
        assert trace.rel[0] == 0.0

    def test_channels_round_trip_through_the_permutation(self, tmp_path):
        # A distinct value per channel proves the file's columns land where NAMES says.
        obs = smooth_motion(n=10)
        obs[:, :] = np.arange(N_CHANNELS) * 0.01
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, obs.copy())
        trace = load_trace(path)
        np.testing.assert_allclose(trace.obs[0], np.arange(N_CHANNELS) * 0.01)

    def test_clamp_identity_holds_on_a_valid_trace(self, good_trace_path):
        assert load_trace(good_trace_path).residual < 1e-5

    @pytest.mark.parametrize("action_type", ["delta_obs_t", "delta_sequential"])
    def test_rejects_delta_action_types(self, tmp_path, action_type):
        obs = smooth_motion(n=20)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, obs + 0.01, action_type=action_type)
        with pytest.raises(TraceAlignmentError, match="delta encoding"):
            load_trace(path)

    def test_accepts_the_joint_abs_alias(self, tmp_path):
        obs = smooth_motion(n=20)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, obs + 0.01, action_type="joint_abs")
        assert load_trace(path).action_type == "joint_abs"

    def test_rejects_a_mispermuted_trace(self, tmp_path):
        """The gate must fire when control_cmd is written in a different order to obs_state.

        This is the failure it exists for: without it the replay renders a plausible-looking
        trajectory built from mismatched channels.
        """
        obs = smooth_motion(n=60)
        raw = obs + 0.01
        # Roll the command channels; the clamp identity can no longer reproduce cmd from raw/obs.
        cmd = np.roll(raw, shift=3, axis=1)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw, cmd=cmd)
        with pytest.raises(TraceAlignmentError, match="clamp identity failed"):
            load_trace(path)

    def test_rejects_joint_scale_gripper(self, tmp_path):
        """A joint-scale value on a gripper channel means the arm/gripper split is wrong."""
        obs = smooth_motion(n=40)
        raw = obs.copy()
        raw[:, 0] = obs[:, 0] + 0.9  # far beyond a gripper's travel
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw)
        with pytest.raises(TraceAlignmentError, match="gripper"):
            load_trace(path)

    @pytest.mark.parametrize("channel", GRIPPER_CHANNELS)
    def test_rejects_joint_scale_on_either_gripper(self, tmp_path, channel):
        """Both grippers are guarded: the permutation lands them at opposite ends of the
        layout, so checking only the left one lets a right-arm mix-up render."""
        obs = smooth_motion(n=40)
        raw = obs.copy()
        raw[:, channel] = obs[:, channel] + 0.9
        path = tmp_path / f"inference_data_{channel}.csv"
        write_trace(path, obs, raw)
        with pytest.raises(TraceAlignmentError, match="gripper"):
            load_trace(path)

    @pytest.mark.parametrize("channel", GRIPPER_CHANNELS)
    def test_accepts_a_gripper_commanded_its_full_stroke(self, tmp_path, channel):
        """A full open from fully closed is legitimate, not a misalignment.

        The URDF's finger joints are prismatic over 0.0 -> 0.05, so this is the largest
        deviation normal operation can produce. A threshold set at the stroke itself rejected
        real traces where the gripper opened all the way.
        """
        obs = smooth_motion(n=40)
        obs[:, channel] = 0.0
        raw = obs.copy()
        raw[:, channel] = 0.05
        path = tmp_path / f"inference_data_{channel}.csv"
        write_trace(path, obs, raw)
        assert load_trace(path).raw[:, channel].max() == pytest.approx(0.05)

    def test_rejects_a_truncated_channel_set(self, tmp_path):
        path = tmp_path / "inference_data.csv"
        path.write_text("# action_type: absolute\n# joint_names: \ntimestamp,obs_state_0\n1.0,0.0\n")
        with pytest.raises(TraceAlignmentError, match="missing column"):
            load_trace(path)

    def test_rejects_an_empty_trace(self, tmp_path):
        path = tmp_path / "inference_data.csv"
        write_trace(path, smooth_motion(n=0), smooth_motion(n=0))
        with pytest.raises(TraceAlignmentError, match="no data rows"):
            load_trace(path)


class TestUndersamplingWarning:
    def test_silent_when_logging_kept_up(self, good_trace_path):
        assert undersampling_warning(load_trace(good_trace_path), 30.0) is None

    def test_silent_when_control_frequency_unknown(self, good_trace_path):
        assert undersampling_warning(load_trace(good_trace_path), None) is None

    def test_warns_and_quantifies_when_commands_outpaced_logging(self, good_trace_path):
        warning = undersampling_warning(load_trace(good_trace_path), 60.0)
        assert warning is not None
        # 30 Hz logged out of 60 Hz commanded: about half the commands are absent.
        assert "50%" in warning
