"""End-to-end tests for the 1-frame-lookahead act-from-obs mechanism.

action_from_observation now bakes action[t] = observation[t+1] (the NEXT
frame's observation), not observation[t] (same-frame copy) — a deliberate
change from the original design. mcap_converter's extract_frames() generator
implements this via a 1-frame lookahead: a frame is held back until the next
frame's own pose/state is known, then its "action" is spliced in and it's
yielded; the episode's own last frame has no successor and is dropped.

These tests run extract_frames() directly against the real smoke-test MCAP
fixtures (not synthetic buffers) to verify the full streaming mechanism —
buffer eviction, subsampling, and the lookahead bookkeeping — together,
covering what the per-function unit tests in test_ee_encoding.py and
test_action_gap_fill.py test in isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.core.extractor import BufferedStreamExtractor

FIXTURES = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
JOINT_MCAP = str(FIXTURES / "test-session" / "0001" / "0001_0.mcap")
EE_MCAP = str(FIXTURES / "ee-session" / "0001" / "0001_0.mcap")


def _joint_afo_config():
    return ConfigLoader.from_yaml(str(FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml"))


def _ee_config(action_encoding: str = "absolute"):
    cfg_dict = {
        "data_space": "ee",
        "action_encoding": action_encoding,
        "observation_topics": {"right": "/ee_pose_right"},
        "action_topics": {},
        "camera_topics": [
            "/cam_waist/image_raw/compressed",
            "/cam_wrist_r/image_raw/compressed",
            "/cam_chest/image_raw/compressed",
        ],
        "camera_topic_mapping": {
            "/cam_waist/image_raw/compressed": "waist",
            "/cam_wrist_r/image_raw/compressed": "wrist_r",
            "/cam_chest/image_raw/compressed": "chest",
        },
        "image_resolution": [640, 480],
    }
    return ConfigLoader.from_dict(cfg_dict)


class TestJointActFromObsLookahead:
    def test_action_equals_next_frame_observation_exactly(self):
        """action[i] must be byte-identical to observation.state[i+1] — the
        act-from-obs case has no encoding/transform in between, just a raw
        1-frame index shift."""
        cfg = _joint_afo_config()
        ext = BufferedStreamExtractor(cfg, fps=10, quiet=True)
        frames = list(ext.extract_frames(JOINT_MCAP))

        assert len(frames) >= 2, "need at least 2 output frames to check the shift"
        for i in range(len(frames) - 1):
            np.testing.assert_array_equal(
                frames[i]["action"], frames[i + 1]["observation.state"]
            )

    def test_all_yielded_frames_have_action(self):
        cfg = _joint_afo_config()
        ext = BufferedStreamExtractor(cfg, fps=10, quiet=True)
        frames = list(ext.extract_frames(JOINT_MCAP))
        assert len(frames) > 0
        for f in frames:
            assert "action" in f


class TestEEAbsoluteLookahead:
    def test_action_matches_next_frame_pose_rot6d_encoded(self):
        """action[i] (absolute) must be the NEXT frame's own pose (position and
        gripper identical; rotation the rot6d encoding of the next frame's quat)."""
        from anvil_shared.rotation import matrix_to_rot6d, quat_to_matrix

        cfg = _ee_config(action_encoding="absolute")
        ext = BufferedStreamExtractor(cfg, fps=10, quiet=True)
        frames = list(ext.extract_frames(EE_MCAP))

        assert len(frames) >= 2
        for i in range(len(frames) - 1):
            action = frames[i]["action"]
            next_state = frames[i + 1]["observation.state"]  # quaternion layout (default)
            np.testing.assert_allclose(action[:3], next_state[:3], atol=1e-5)  # xyz
            expected_rot6d = matrix_to_rot6d(quat_to_matrix(next_state[3:7].astype(np.float64)))
            np.testing.assert_allclose(action[3:9], expected_rot6d, atol=1e-4)
            np.testing.assert_allclose(action[9], next_state[7], atol=1e-6)  # gripper


class TestEEDeltaLookahead:
    def test_delta_inverse_reconstructs_next_frame_state(self):
        """ee_delta_inverse(action[i], state[i]) must reconstruct state[i+1] —
        the forward-looking (t -> t+1) delta convention, on a real MCAP."""
        from anvil_shared.ee_transform import ee_delta_inverse
        from anvil_shared.rotation import rot6d_to_matrix, matrix_to_quat

        cfg = _ee_config(action_encoding="delta")
        ext = BufferedStreamExtractor(cfg, fps=10, quiet=True)
        frames = list(ext.extract_frames(EE_MCAP))

        assert len(frames) >= 2
        for i in range(len(frames) - 1):
            action = frames[i]["action"].astype(np.float64)
            state = frames[i]["observation.state"].astype(np.float64)
            next_state = frames[i + 1]["observation.state"].astype(np.float64)

            recon = ee_delta_inverse(action[None, :], state[None, :])[0]
            np.testing.assert_allclose(recon[:3], next_state[:3], atol=1e-5)  # xyz
            recon_quat = matrix_to_quat(rot6d_to_matrix(recon[3:9]))
            # quaternion double-cover: q and -q are the same rotation.
            sign = 1.0 if np.dot(recon_quat, next_state[3:7]) >= 0 else -1.0
            np.testing.assert_allclose(sign * recon_quat, next_state[3:7], atol=1e-4)
            np.testing.assert_allclose(recon[9], next_state[7], atol=1e-6)  # gripper

    def test_frame_count_is_one_less_than_absolute_encoding(self):
        """Both encodings share the exact same 1-frame lookahead/drop mechanism
        — they must yield the same number of frames for the same input."""
        ext_abs = BufferedStreamExtractor(_ee_config("absolute"), fps=10, quiet=True)
        ext_delta = BufferedStreamExtractor(_ee_config("delta"), fps=10, quiet=True)
        n_abs = len(list(ext_abs.extract_frames(EE_MCAP)))
        n_delta = len(list(ext_delta.extract_frames(EE_MCAP)))
        assert n_abs == n_delta
        assert n_abs > 0
