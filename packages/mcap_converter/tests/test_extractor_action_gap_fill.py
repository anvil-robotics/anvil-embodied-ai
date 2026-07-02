"""Tests for BufferedStreamExtractor's action gap-fill behavior."""

from collections import deque

import numpy as np

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


def test_resolve_action_position_exact_match_from_buffer():
    extractor = make_extractor()
    buffer = deque([(1.0, np.array([1.0, 1.0], dtype=np.float32), np.array([]), np.array([]))])

    pos, fill_kind = extractor._resolve_action_position("left", buffer, 1.0, obs_data={})

    np.testing.assert_array_equal(pos, np.array([1.0, 1.0], dtype=np.float32))
    assert fill_kind == "exact"


def test_resolve_action_position_holds_last_known_when_buffer_empty():
    extractor = make_extractor()
    extractor._last_known_action["right"] = np.array([3.0, 4.0], dtype=np.float32)
    empty_buffer = deque()

    pos, fill_kind = extractor._resolve_action_position("right", empty_buffer, 12.0, obs_data={})

    np.testing.assert_array_equal(pos, np.array([3.0, 4.0], dtype=np.float32))
    assert fill_kind == "hold_last"


def test_resolve_action_position_falls_back_to_observation_when_never_published():
    extractor = make_extractor()
    empty_buffer = deque()
    obs_data = {"right": {"pos": np.array([5.0, 6.0], dtype=np.float32), "vel": None, "eff": None}}

    pos, fill_kind = extractor._resolve_action_position("right", empty_buffer, 2.0, obs_data)

    np.testing.assert_array_equal(pos, np.array([5.0, 6.0], dtype=np.float32))
    assert fill_kind == "fallback_to_observation"


def test_resolve_action_position_drops_when_no_fallback_available():
    extractor = make_extractor()
    empty_buffer = deque()

    pos, fill_kind = extractor._resolve_action_position("right", empty_buffer, 2.0, obs_data={})

    assert pos is None
    assert fill_kind == "dropped"


def test_record_action_fill_accumulates_counts_per_robot():
    extractor = make_extractor()

    extractor._record_action_fill("right", "exact")
    extractor._record_action_fill("right", "hold_last")
    extractor._record_action_fill("right", "hold_last")
    extractor._record_action_fill("left", "exact")

    stats = extractor.get_action_fill_stats()
    assert stats["right"] == {"exact": 1, "hold_last": 2, "fallback_to_observation": 0, "dropped": 0}
    assert stats["left"] == {"exact": 1, "hold_last": 0, "fallback_to_observation": 0, "dropped": 0}
