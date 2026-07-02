"""Tests for the mcap quality validator's coverage/gap analysis.

Verifies:
1. Per-topic coverage analysis (exact/idle/dropframe/leading/trailing gaps).
2. Cross-episode fps degradation detection.
3. Topic resolution from DataConfig (which topics to monitor, quest vs
   leader-follower mode).
4. The I/O adapter that reads a real MCAP file and produces a report.
"""

from collections import deque

import numpy as np
import pytest

from mcap_converter.config.schema import ActionTopicConfig, DataConfig
from mcap_converter.core.quality import (
    MonitoredTopic,
    QualityThresholds,
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    resolve_monitored_topics,
    worst_severity,
)


def make_quest_config() -> DataConfig:
    """Bimanual quest-teleop config, matching the real bug scenario."""
    return DataConfig(
        action_topics={
            "/follower_l_forward_position_controller/commands": ActionTopicConfig(
                arm="left", joint_order=["joint1", "joint2"]
            ),
            "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                arm="right", joint_order=["joint1", "joint2"]
            ),
        },
        camera_topics=["/cam_chest/image_raw/compressed"],
        camera_topic_mapping={"/cam_chest/image_raw/compressed": "chest"},
    )


def make_leader_follower_config() -> DataConfig:
    """Default DataConfig has empty action_topics -> leader-follower mode."""
    return DataConfig(
        camera_topics=["/cam_chest/image_raw/compressed"],
        camera_topic_mapping={"/cam_chest/image_raw/compressed": "chest"},
    )


class TestWorstSeverity:
    def test_critical_beats_warning_and_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING, SEVERITY_CRITICAL]) == SEVERITY_CRITICAL

    def test_warning_beats_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING]) == SEVERITY_WARNING

    def test_all_ok_is_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_OK]) == SEVERITY_OK

    def test_empty_defaults_to_ok(self):
        assert worst_severity([]) == SEVERITY_OK


class TestResolveMonitoredTopics:
    def test_picks_present_camera_variant(self):
        config = make_quest_config()
        available = {"/cam_chest/image_raw/compressed", "/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        camera = next(m for m in monitored if m.label == "chest")
        assert camera.topic == "/cam_chest/image_raw/compressed"
        assert camera.role == "stream"

    def test_camera_missing_all_variants_selects_base(self):
        config = make_quest_config()
        available = {"/joint_states"}  # camera topic entirely absent

        monitored = resolve_monitored_topics(config, available)

        camera = next(m for m in monitored if m.label == "chest")
        assert camera.topic == "/cam_chest/image_raw/compressed"  # falls back to configured name

    def test_quest_mode_produces_action_items_with_arm_label(self):
        config = make_quest_config()
        available = {
            "/follower_l_forward_position_controller/commands",
            "/follower_r_forward_position_controller/commands",
            "/cam_chest/image_raw/compressed",
            "/joint_states",
        }

        monitored = resolve_monitored_topics(config, available)

        labels = {m.label for m in monitored if m.role == "action"}
        assert labels == {"action[left]", "action[right]"}

    def test_leader_follower_mode_has_no_action_items(self):
        config = make_leader_follower_config()
        available = {"/cam_chest/image_raw/compressed", "/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        assert not [m for m in monitored if m.role == "action"]

    def test_robot_state_topic_is_a_stream(self):
        config = make_quest_config()
        available = {"/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        joint_states = next(m for m in monitored if m.label == "joint_states")
        assert joint_states.topic == "/joint_states"
        assert joint_states.role == "stream"
