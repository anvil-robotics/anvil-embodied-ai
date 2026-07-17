"""Tests for joint ordering, canonical sort, and action reordering.

Verifies:
1. parse_joint_name() correctly parses quest-mode joint names
2. Canonical sort produces deterministic alphabetical order
3. Action reorder permutation maps [j1..j7, fj1] -> [fj1, j1..j7]
4. ActionTopicConfig round-trips through loader
"""

import numpy as np
import pytest

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.schema import (
    ActionTopicConfig,
    ActionTopicSpec,
    ConfigurationError,
    DataConfig,
    JointNamePattern,
)
from mcap_converter.core.extractor import parse_joint_name


# =============================================================================
# parse_joint_name tests
# =============================================================================


class TestParseJointName:
    """Test joint name parsing with quest-mode config."""

    @pytest.fixture
    def quest_pattern(self):
        return JointNamePattern(
            source={"follower": "observation"},
            arms={"r": "right", "l": "left"},
            separator="_",
        )

    def test_follower_left_joint(self, quest_pattern):
        role, robot, joint_id = parse_joint_name("follower_l_joint1", quest_pattern)
        assert role == "observation"
        assert robot == "left"
        assert joint_id == "joint1"

    def test_follower_right_joint(self, quest_pattern):
        role, robot, joint_id = parse_joint_name("follower_r_joint5", quest_pattern)
        assert role == "observation"
        assert robot == "right"
        assert joint_id == "joint5"

    def test_follower_finger_joint(self, quest_pattern):
        role, robot, joint_id = parse_joint_name(
            "follower_l_finger_joint1", quest_pattern
        )
        assert role == "observation"
        assert robot == "left"
        assert joint_id == "finger_joint1"

    def test_unknown_prefix_raises(self, quest_pattern):
        from mcap_converter.exceptions import DataExtractionError

        with pytest.raises(DataExtractionError):
            parse_joint_name("leader_r_joint1", quest_pattern)

    def test_leader_follower_pattern(self):
        pattern = JointNamePattern(
            source={"leader": "action", "follower": "observation"},
            arms={"r": "right", "l": "left"},
            separator="_",
        )
        role, robot, joint_id = parse_joint_name("leader_r_joint3", pattern)
        assert role == "action"
        assert robot == "right"
        assert joint_id == "joint3"


# =============================================================================
# Canonical sort tests
# =============================================================================


class TestCanonicalSort:
    """Test that canonical sort produces correct alphabetical ordering."""

    def test_observation_sort_order(self):
        """Joints from /joint_states should be sorted alphabetically by joint_id."""
        # Simulates the order that comes from the ROS message
        unsorted_ids = [
            "finger_joint1",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        positions = [0.05, 0.01, -0.37, -0.13, 1.78, -0.29, -0.45, -0.49]

        # Apply canonical sort (same logic as extractor)
        sort_indices = sorted(
            range(len(unsorted_ids)), key=lambda idx: unsorted_ids[idx]
        )
        sorted_ids = [unsorted_ids[idx] for idx in sort_indices]
        sorted_pos = [positions[idx] for idx in sort_indices]

        # finger_joint1 < joint1 < joint2 ... alphabetically
        assert sorted_ids == [
            "finger_joint1",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        # finger_joint1 position should be first
        assert sorted_pos[0] == 0.05

    def test_already_sorted_is_stable(self):
        """If joints are already sorted, order should not change."""
        ids = ["joint1", "joint2", "joint3"]
        sort_indices = sorted(range(len(ids)), key=lambda idx: ids[idx])
        assert sort_indices == [0, 1, 2]


# =============================================================================
# Action reorder permutation tests
# =============================================================================


class TestActionReorder:
    """Test action position reordering from config joint_order to canonical order."""

    def test_reorder_finger_last_to_first(self):
        """Action data [j1..j7, fj1] should reorder to [fj1, j1..j7]."""
        joint_order = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
        ]
        # The action data as it comes from the Float64MultiArray
        action_data = np.array(
            [0.01, -0.35, -0.10, 1.69, -0.27, -0.46, -0.49, 0.05],
            dtype=np.float32,
        )

        # Compute canonical reorder (same logic as extractor)
        reorder = np.array(
            sorted(range(len(joint_order)), key=lambda idx: joint_order[idx]),
            dtype=np.intp,
        )
        reordered = action_data[reorder]

        # After reorder: canonical order is [finger_joint1, joint1..joint7]
        canonical_names = sorted(joint_order)
        assert canonical_names == [
            "finger_joint1",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # finger_joint1 was at index 7 (value 0.05), should now be first
        assert reordered[0] == pytest.approx(0.05)
        # joint1 was at index 0 (value 0.01), should now be second
        assert reordered[1] == pytest.approx(0.01)

    def test_reorder_matches_observation_order(self):
        """After reordering, action joints should match observation joints."""
        # Observation joints (already canonically sorted in _buffer_joint_state)
        obs_joint_ids = [
            "finger_joint1",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # Action joint_order from config
        action_joint_order = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
        ]

        # Canonical sort of action joint_order
        canonical_action = sorted(action_joint_order)

        assert canonical_action == obs_joint_ids

    def test_identity_when_already_sorted(self):
        """If joint_order is already alphabetical, reorder should be identity."""
        joint_order = ["finger_joint1", "joint1", "joint2"]
        reorder = sorted(range(len(joint_order)), key=lambda idx: joint_order[idx])
        assert reorder == [0, 1, 2]


# =============================================================================
# Config loader tests
# =============================================================================


class TestActionTopicParsing:
    """Test the new arm-keyed action_topics format and the derived
    action_command_topics property used by the joint extractor."""

    def _base_joint_yaml(self) -> dict:
        return {
            "data_space": "joint",
            "observation_topics": {"left": "/joint_states", "right": "/joint_states"},
            "joint_names": {
                "separator": "_",
                "source": {"follower": "observation"},
                "arms": {"l": "left", "r": "right"},
            },
            "camera_topics": ["/cam"],
            "camera_topic_mapping": {"/cam": "head"},
        }

    def test_new_format_parsing(self):
        cfg_dict = self._base_joint_yaml()
        cfg_dict["action_topics"] = {
            "left": {
                "topic": "/left_cmd",
                "joint_order": ["joint1", "joint2", "finger_joint1"],
            },
            "right": {
                "topic": "/right_cmd",
                "joint_order": ["joint1", "joint2", "finger_joint1"],
            },
        }
        config = ConfigLoader.from_dict(cfg_dict)

        assert isinstance(config.action_topics["left"], ActionTopicSpec)
        assert config.action_topics["left"].topic == "/left_cmd"
        assert config.action_topics["left"].joint_order == [
            "joint1",
            "joint2",
            "finger_joint1",
        ]
        assert config.action_topics["right"].topic == "/right_cmd"

        # Derived property used by the joint extractor matches the legacy
        # topic-keyed shape exactly.
        derived = config.action_command_topics
        assert set(derived.keys()) == {"/left_cmd", "/right_cmd"}
        assert isinstance(derived["/left_cmd"], ActionTopicConfig)
        assert derived["/left_cmd"].arm == "left"
        assert derived["/left_cmd"].joint_order == [
            "joint1",
            "joint2",
            "finger_joint1",
        ]
        assert derived["/right_cmd"].arm == "right"

    def test_empty_action_topics_means_act_from_obs(self):
        cfg_dict = self._base_joint_yaml()
        cfg_dict["action_topics"] = {}
        config = ConfigLoader.from_dict(cfg_dict)
        assert config.action_topics == {}
        assert config.action_command_topics == {}

    def test_legacy_format_rejected(self):
        """Old topic-keyed {topic: {arm, joint_order}} must fail (new-format-only)."""
        cfg_dict = self._base_joint_yaml()
        # legacy shape: key is topic, value lacks an explicit 'topic' field
        cfg_dict["action_topics"] = {
            "/left_cmd": {"arm": "left", "joint_order": ["joint1"]},
        }
        # ConfigLoader now raises ConfigurationError uniformly for all loader-level
        # errors (structural/type and value-legality alike), replacing the previous
        # split between bare ValueError (loader) and ConfigurationError (validators).
        with pytest.raises(ConfigurationError):
            ConfigLoader.from_dict(cfg_dict)


# =============================================================================
# Validator tests
# =============================================================================


def _minimal_joint_dict(**overrides) -> dict:
    base = {
        "data_space": "joint",
        "observation_topics": {"left": "/joint_states", "right": "/joint_states"},
        "action_topics": {
            "left": {
                "topic": "/left_cmd",
                "joint_order": ["joint1", "finger_joint1"],
            },
            "right": {
                "topic": "/right_cmd",
                "joint_order": ["joint1", "finger_joint1"],
            },
        },
        "joint_names": {
            "separator": "_",
            "source": {"follower": "observation"},
            "arms": {"l": "left", "r": "right"},
        },
        "camera_topics": ["/cam"],
        "camera_topic_mapping": {"/cam": "head"},
    }
    base.update(overrides)
    return base


class TestValidateConfig:
    """Exercise validate_config against the new unified schema."""

    def test_valid_joint_config(self):
        cfg = ConfigLoader.from_dict(_minimal_joint_dict())
        cfg.validate()  # no raise

    def test_joint_empty_joint_order_rejected(self):
        cfg = ConfigLoader.from_dict(_minimal_joint_dict(
            action_topics={
                "left": {"topic": "/left_cmd", "joint_order": []},
                "right": {"topic": "/right_cmd", "joint_order": ["joint1"]},
            }
        ))
        with pytest.raises(ConfigurationError, match="joint_order"):
            cfg.validate()

    def test_joint_action_arm_missing_from_observation_rejected(self):
        cfg = ConfigLoader.from_dict(_minimal_joint_dict(
            action_topics={
                "left": {"topic": "/left_cmd", "joint_order": ["joint1"]},
                "ghost": {"topic": "/ghost_cmd", "joint_order": ["joint1"]},
            }
        ))
        with pytest.raises(ConfigurationError, match="not present in observation_topics"):
            cfg.validate()

    def test_ee_with_action_topics_rejected(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "observation_topics": {"left": "/ee_pose_left"},
            "action_topics": {
                "left": {"topic": "/x", "joint_order": []},
            },
            "camera_topics": ["/cam"],
            "camera_topic_mapping": {"/cam": "head"},
        })
        with pytest.raises(ConfigurationError, match="action_topics must be empty in ee mode"):
            cfg.validate()

    def test_ee_valid_empty_action_topics(self):
        cfg = ConfigLoader.from_dict({
            "data_space": "ee",
            "observation_topics": {"left": "/ee_pose_left", "right": "/ee_pose_right"},
            "action_topics": {},
            "camera_topics": ["/cam"],
            "camera_topic_mapping": {"/cam": "head"},
        })
        cfg.validate()  # no raise
