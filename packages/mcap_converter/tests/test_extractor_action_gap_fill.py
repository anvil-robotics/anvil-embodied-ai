"""Tests for BufferedStreamExtractor's action gap-fill behavior."""

from collections import deque

import numpy as np
import pytest

from mcap_converter.config.schema import ActionTopicConfig, DataConfig
from mcap_converter.core.extractor import BufferedStreamExtractor


def make_config() -> DataConfig:
    """Bimanual quest-teleop-style config matching the real bug scenario."""
    return DataConfig(
        action_topics={
            "/follower_l_forward_position_controller/commands": ActionTopicConfig(
                arm="left", joint_order=["joint1", "joint2"]
            ),
            "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                arm="right", joint_order=["joint1", "joint2"]
            ),
        },
    )


def make_extractor() -> BufferedStreamExtractor:
    return BufferedStreamExtractor(config=make_config(), buffer_seconds=5.0, fps=60, quiet=True)


class FakeFloat64MultiArrayMessage:
    """Minimal stand-in for an mcap message wrapping std_msgs/Float64MultiArray.

    Float64MultiArray has no header, so message_timestamp() falls back to
    log_time_ns — this fake only needs that attribute plus ros_msg.data.
    """

    def __init__(self, data: list[float], log_time_s: float):
        self.log_time_ns = int(log_time_s * 1e9)
        self.ros_msg = type("RosMsg", (), {"data": data})()


def test_buffer_action_command_updates_last_known_action():
    extractor = make_extractor()
    joint_buffers = {}

    msg = FakeFloat64MultiArrayMessage(data=[1.0, 2.0], log_time_s=7.0)
    extractor._buffer_action_command(
        msg, "/follower_r_forward_position_controller/commands", joint_buffers
    )

    assert "right" in extractor._last_known_action
    np.testing.assert_array_equal(
        extractor._last_known_action["right"], np.array([1.0, 2.0], dtype=np.float32)
    )
