"""Forward kinematics against the vendored OpenArm description."""

from __future__ import annotations

import numpy as np
import pytest

from inference_replay.kinematics import RobotKinematics, default_urdf_path
from inference_replay.trace import N_CHANNELS, URDF_JOINT_NAMES

pytestmark = pytest.mark.skipif(
    not default_urdf_path().is_file(),
    reason="vendored URDF absent; run scripts/sync_openarm_assets.py",
)


@pytest.fixture(scope="module")
def kin():
    return RobotKinematics()


def _zero_config() -> np.ndarray:
    return np.zeros(N_CHANNELS)


class TestDescription:
    def test_actuates_exactly_the_traced_joints(self, kin):
        # A mismatch here means the CSV and the URDF disagree about the robot, which the
        # constructor is meant to catch rather than silently mis-pose.
        poses = kin.link_poses(_zero_config())
        assert len(URDF_JOINT_NAMES) == N_CHANNELS
        assert poses  # every visual link resolved

    def test_visual_meshes_all_exist_on_disk(self, kin):
        missing = [v.mesh_path for v in kin.visuals if not v.mesh_path.is_file()]
        assert missing == []

    def test_both_arms_are_present(self, kin):
        links = set(kin.link_names)
        assert any(name.startswith("follower_l_") for name in links)
        assert any(name.startswith("follower_r_") for name in links)

    def test_rejects_a_wrong_channel_count(self, kin):
        with pytest.raises(ValueError, match="expected 16 channels"):
            kin.link_poses(np.zeros(8))

    def test_missing_urdf_is_reported_with_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="sync_openarm_assets"):
            RobotKinematics(tmp_path / "nope.urdf")


class TestForwardKinematics:
    def test_poses_are_homogeneous_transforms(self, kin):
        for matrix in kin.link_poses(_zero_config()).values():
            assert matrix.shape == (4, 4)
            np.testing.assert_allclose(matrix[3], [0, 0, 0, 1], atol=1e-9)
            # A rigid transform's rotation block is orthonormal with determinant +1.
            rotation = matrix[:3, :3]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)

    def test_geometry_is_metre_scale(self, kin):
        """Mesh scale is baked in by the sync script; a mm-scale slip would show up here."""
        positions = np.array([m[:3, 3] for m in kin.link_poses(_zero_config()).values()])
        assert np.abs(positions).max() < 2.0

    def test_zero_config_is_reproducible(self, kin):
        first = kin.link_poses(_zero_config())
        second = kin.link_poses(_zero_config())
        for link in first:
            np.testing.assert_allclose(first[link], second[link], atol=1e-12)

    def test_moving_a_joint_moves_that_arm(self, kin):
        rest = kin.link_poses(_zero_config())["follower_l_hand"][:3, 3]
        config = _zero_config()
        config[URDF_JOINT_NAMES.index("follower_l_joint2")] = 0.5
        moved = kin.link_poses(config)["follower_l_hand"][:3, 3]
        assert np.linalg.norm(moved - rest) > 0.05

    def test_moving_the_left_arm_leaves_the_right_arm_alone(self, kin):
        """Guards against the two arms sharing a joint index by mistake."""
        rest = kin.link_poses(_zero_config())["follower_r_hand"][:3, 3]
        config = _zero_config()
        config[URDF_JOINT_NAMES.index("follower_l_joint2")] = 0.5
        after = kin.link_poses(config)["follower_r_hand"][:3, 3]
        np.testing.assert_allclose(after, rest, atol=1e-9)

    @pytest.mark.parametrize("side", ["follower_l", "follower_r"])
    @pytest.mark.parametrize("joint", range(1, 8))
    def test_each_arm_joint_has_an_effect_on_its_hand(self, kin, side, joint):
        """Every channel must be wired to something; a silently-ignored joint would freeze it."""
        hand = f"{side}_hand"
        rest = kin.link_poses(_zero_config())[hand]
        config = _zero_config()
        config[URDF_JOINT_NAMES.index(f"{side}_joint{joint}")] = 0.4
        # The last joint is a wrist roll that can leave the hand *origin* nearly fixed, so
        # compare the full pose rather than just the translation.
        assert np.abs(kin.link_poses(config)[hand] - rest).max() > 1e-6

    def test_gripper_channel_drives_the_finger(self, kin):
        closed = kin.link_poses(_zero_config())
        config = _zero_config()
        config[URDF_JOINT_NAMES.index("follower_l_finger_joint1")] = 0.04
        opened = kin.link_poses(config)
        finger_links = [name for name in closed if "finger" in name and name.startswith("follower_l")]
        assert finger_links, "no left finger links in the description"
        assert any(
            np.abs(opened[link][:3, 3] - closed[link][:3, 3]).max() > 1e-4 for link in finger_links
        )
