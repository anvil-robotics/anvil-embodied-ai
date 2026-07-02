"""Tests for BufferedStreamExtractor's action gap-fill behavior.

Verifies:
1. _buffer_action_command() caches the last known action per robot, independent
   of the sliding time-window buffer's eviction.
2. _resolve_action_position()'s fallback chain: exact buffer match -> hold the
   last known command -> fall back to the robot's measured observation
   position -> give up (dropped).
3. _record_action_fill()/get_action_fill_stats() correctly count each fallback
   tier per robot.
4. _align_joint_states() forward-fills action data through mid-episode and
   head-of-episode gaps instead of dropping the whole frame, while leaving
   observation-role handling unchanged (still required, no fallback).
5. _init_joint_buffers() pre-seeds an action buffer for every configured
   robot so the fallback-to-observation tier is reachable even before that
   robot has ever published a command.
"""

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


def _joint_buffer(ts_pos_pairs):
    """Build a (timestamp, position, velocity, effort) buffer, as used for
    both observation and action entries in joint_buffers."""
    buf = deque()
    for ts, pos in ts_pos_pairs:
        buf.append((ts, np.array(pos, dtype=np.float32), np.array([]), np.array([])))
    return buf


class FakeFloat64MultiArrayMessage:
    """Minimal stand-in for an mcap message wrapping std_msgs/Float64MultiArray.

    Float64MultiArray has no header, so message_timestamp() falls back to
    log_time_ns — this fake only needs that attribute plus ros_msg.data.
    """

    def __init__(self, data: list[float], log_time_s: float):
        self.log_time_ns = int(log_time_s * 1e9)
        self.ros_msg = type("RosMsg", (), {"data": data})()


# =============================================================================
# _buffer_action_command
# =============================================================================


class TestBufferActionCommand:
    def test_updates_last_known_action(self):
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


# =============================================================================
# _resolve_action_position
# =============================================================================


class TestResolveActionPosition:
    def test_exact_match_from_buffer(self):
        extractor = make_extractor()
        buffer = _joint_buffer([(1.0, [1.0, 1.0])])

        pos, fill_kind = extractor._resolve_action_position("left", buffer, 1.0, obs_data={})

        np.testing.assert_array_equal(pos, np.array([1.0, 1.0], dtype=np.float32))
        assert fill_kind == "exact"

    def test_holds_last_known_when_buffer_empty(self):
        extractor = make_extractor()
        extractor._last_known_action["right"] = np.array([3.0, 4.0], dtype=np.float32)

        pos, fill_kind = extractor._resolve_action_position(
            "right", deque(), 12.0, obs_data={}
        )

        np.testing.assert_array_equal(pos, np.array([3.0, 4.0], dtype=np.float32))
        assert fill_kind == "hold_last"

    def test_falls_back_to_observation_when_never_published(self):
        extractor = make_extractor()
        obs_data = {"right": {"pos": np.array([5.0, 6.0], dtype=np.float32), "vel": None, "eff": None}}

        pos, fill_kind = extractor._resolve_action_position("right", deque(), 2.0, obs_data)

        np.testing.assert_array_equal(pos, np.array([5.0, 6.0], dtype=np.float32))
        assert fill_kind == "fallback_to_observation"

    def test_drops_when_no_fallback_available(self):
        extractor = make_extractor()

        pos, fill_kind = extractor._resolve_action_position("right", deque(), 2.0, obs_data={})

        assert pos is None
        assert fill_kind == "dropped"

    def test_prefers_exact_over_hold_last_and_fallback(self):
        """When all three sources are available, the live buffer must win —
        it's the most current information, not just the first branch checked."""
        extractor = make_extractor()
        extractor._last_known_action["right"] = np.array([9.0, 9.0], dtype=np.float32)
        obs_data = {"right": {"pos": np.array([0.0, 0.0], dtype=np.float32), "vel": None, "eff": None}}
        buffer = _joint_buffer([(1.0, [1.0, 1.0])])

        pos, fill_kind = extractor._resolve_action_position("right", buffer, 1.0, obs_data)

        np.testing.assert_array_equal(pos, np.array([1.0, 1.0], dtype=np.float32))
        assert fill_kind == "exact"

    def test_returned_positions_are_independent_copies(self):
        """Mutating a resolved position must not corrupt the cache/buffer it
        came from — a future frame's resolution would otherwise pick up the
        mutation instead of the original recorded value."""
        extractor = make_extractor()

        last_known = np.array([3.0, 4.0], dtype=np.float32)
        extractor._last_known_action["right"] = last_known
        hold_last_pos, _ = extractor._resolve_action_position("right", deque(), 1.0, obs_data={})
        hold_last_pos[:] = -1.0
        np.testing.assert_array_equal(extractor._last_known_action["right"], last_known)

        obs_pos = np.array([5.0, 6.0], dtype=np.float32)
        obs_data = {"left": {"pos": obs_pos, "vel": None, "eff": None}}
        fallback_pos, _ = extractor._resolve_action_position("left", deque(), 1.0, obs_data)
        fallback_pos[:] = -1.0
        np.testing.assert_array_equal(obs_data["left"]["pos"], obs_pos)


# =============================================================================
# _record_action_fill / get_action_fill_stats
# =============================================================================


class TestRecordActionFill:
    def test_accumulates_counts_per_robot(self):
        extractor = make_extractor()

        extractor._record_action_fill("right", "exact")
        extractor._record_action_fill("right", "hold_last")
        extractor._record_action_fill("right", "hold_last")
        extractor._record_action_fill("left", "exact")

        stats = extractor.get_action_fill_stats()
        assert stats["right"] == {"exact": 1, "hold_last": 2, "fallback_to_observation": 0, "dropped": 0}
        assert stats["left"] == {"exact": 1, "hold_last": 0, "fallback_to_observation": 0, "dropped": 0}


# =============================================================================
# _align_joint_states
# =============================================================================


class TestAlignJointStatesActionFill:
    def test_keeps_episode_going_through_mid_episode_gap(self):
        """Reproduces the real bug: right arm goes idle mid-episode, frame must survive."""
        extractor = make_extractor()
        # Right arm published once at t=7.0 then went idle (disengaged) — its
        # buffer has been evicted by the time we're aligning t=12.0.
        extractor._last_known_action["right"] = np.array([9.0, 9.0], dtype=np.float32)

        joint_buffers = {
            ("observation", "left"): {"buffer": _joint_buffer([(12.0, [1.0, 1.0])])},
            ("observation", "right"): {"buffer": _joint_buffer([(12.0, [9.0, 9.0])])},
            ("action", "left"): {"buffer": _joint_buffer([(12.0, [2.0, 2.0])])},
            ("action", "right"): {"buffer": deque()},  # idle — nothing live
        }

        result = extractor._align_joint_states(joint_buffers, target_ts=12.0)

        assert result is not None
        np.testing.assert_array_equal(
            result["action"], np.array([2.0, 2.0, 9.0, 9.0], dtype=np.float32)
        )
        stats = extractor.get_action_fill_stats()
        assert stats["right"]["hold_last"] == 1
        assert stats["left"]["exact"] == 1

    def test_falls_back_to_observation_at_episode_start(self):
        """Reproduces the head-of-episode gap: right arm never engaged yet."""
        extractor = make_extractor()
        # No entry in _last_known_action["right"] — it has never published.

        joint_buffers = {
            ("observation", "left"): {"buffer": _joint_buffer([(0.5, [1.0, 1.0])])},
            ("observation", "right"): {"buffer": _joint_buffer([(0.5, [0.0, 0.0])])},
            ("action", "left"): {"buffer": _joint_buffer([(0.5, [1.5, 1.5])])},
            ("action", "right"): {"buffer": deque()},
        }

        result = extractor._align_joint_states(joint_buffers, target_ts=0.5)

        assert result is not None
        np.testing.assert_array_equal(
            result["action"], np.array([1.5, 1.5, 0.0, 0.0], dtype=np.float32)
        )
        assert extractor.get_action_fill_stats()["right"]["fallback_to_observation"] == 1

    def test_still_requires_observation_data(self):
        """Observation-role handling must be unchanged: missing observation still drops the frame."""
        extractor = make_extractor()

        joint_buffers = {
            ("observation", "left"): {"buffer": deque()},  # missing observation — must still drop
            ("observation", "right"): {"buffer": _joint_buffer([(1.0, [0.0, 0.0])])},
            ("action", "left"): {"buffer": _joint_buffer([(1.0, [1.0, 1.0])])},
            ("action", "right"): {"buffer": _joint_buffer([(1.0, [0.0, 0.0])])},
        }

        result = extractor._align_joint_states(joint_buffers, target_ts=1.0)

        assert result is None

    def test_falls_back_to_observation_when_action_key_never_created(self):
        """End-to-end reproduction using the REAL initializer (not a hand-built
        dict), proving the fallback actually fires when a robot's action key
        only exists because of pre-seeding, not because of a prior message."""
        extractor = make_extractor()
        joint_buffers = extractor._init_joint_buffers()
        # Simulate: left has already published a command; right never has.
        joint_buffers[("action", "left")]["buffer"].append(
            (0.5, np.array([1.5, 1.5], dtype=np.float32), np.array([]), np.array([]))
        )
        joint_buffers[("observation", "left")] = {"buffer": _joint_buffer([(0.5, [1.0, 1.0])])}
        joint_buffers[("observation", "right")] = {"buffer": _joint_buffer([(0.5, [0.0, 0.0])])}

        result = extractor._align_joint_states(joint_buffers, target_ts=0.5)

        assert result is not None
        np.testing.assert_array_equal(
            result["action"], np.array([1.5, 1.5, 0.0, 0.0], dtype=np.float32)
        )
        assert extractor.get_action_fill_stats()["right"]["fallback_to_observation"] == 1


# =============================================================================
# _init_joint_buffers
# =============================================================================


class TestInitJointBuffers:
    def test_preseeds_action_key_for_every_configured_robot(self):
        """Reproduces the real key-lifecycle bug: before this fix, an
        ("action", robot) key didn't exist in joint_buffers until that robot's
        first command message arrived, so _align_joint_states's action pass
        would never visit it and the fallback-to-observation tier was
        unreachable in production."""
        extractor = make_extractor()

        joint_buffers = extractor._init_joint_buffers()

        assert ("action", "left") in joint_buffers
        assert ("action", "right") in joint_buffers
        assert joint_buffers[("action", "left")]["buffer"] == deque()
        assert joint_buffers[("action", "right")]["buffer"] == deque()
