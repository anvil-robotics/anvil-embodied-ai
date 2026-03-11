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
from mcap_converter.config.schema import ActionTopicConfig, DataConfig, JointNamePattern
from mcap_converter.config.validators import validate_action_topics, validate_config
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


class TestActionTopicConfig:
    """Test ActionTopicConfig parsing through the config loader."""

    def test_new_format_parsing(self):
        """New nested format with arm and joint_order."""
        config_dict = {
            "robot_state_topic": "/joint_states",
            "action_topics": {
                "/left_cmd": {
                    "arm": "left",
                    "joint_order": ["joint1", "joint2", "finger_joint1"],
                },
                "/right_cmd": {
                    "arm": "right",
                    "joint_order": ["joint1", "joint2", "finger_joint1"],
                },
            },
            "camera_topics": ["/cam"],
            "camera_topic_mapping": {"/cam": "head"},
        }
        config = ConfigLoader.from_dict(config_dict)

        assert isinstance(config.action_topics["/left_cmd"], ActionTopicConfig)
        assert config.action_topics["/left_cmd"].arm == "left"
        assert config.action_topics["/left_cmd"].joint_order == [
            "joint1",
            "joint2",
            "finger_joint1",
        ]
        assert config.action_topics["/right_cmd"].arm == "right"

    def test_legacy_format_parsing(self):
        """Legacy plain string format should produce ActionTopicConfig with empty joint_order."""
        config_dict = {
            "robot_state_topic": "/joint_states",
            "action_topics": {
                "/left_cmd": "left",
            },
            "camera_topics": ["/cam"],
            "camera_topic_mapping": {"/cam": "head"},
        }
        with pytest.warns(DeprecationWarning, match="plain string format"):
            config = ConfigLoader.from_dict(config_dict)

        assert isinstance(config.action_topics["/left_cmd"], ActionTopicConfig)
        assert config.action_topics["/left_cmd"].arm == "left"
        assert config.action_topics["/left_cmd"].joint_order == []


# =============================================================================
# Validator tests
# =============================================================================


class TestValidateActionTopics:
    """Test action topic validation."""

    def test_valid_config(self):
        topics = {
            "/cmd": ActionTopicConfig(
                arm="left", joint_order=["joint1", "finger_joint1"]
            ),
        }
        errors = validate_action_topics(topics)
        assert errors == []

    def test_empty_arm(self):
        topics = {
            "/cmd": ActionTopicConfig(arm="", joint_order=["joint1"]),
        }
        errors = validate_action_topics(topics)
        assert any("arm identifier" in e for e in errors)

    def test_empty_joint_order(self):
        topics = {
            "/cmd": ActionTopicConfig(arm="left", joint_order=[]),
        }
        errors = validate_action_topics(topics)
        assert any("joint_order" in e for e in errors)
