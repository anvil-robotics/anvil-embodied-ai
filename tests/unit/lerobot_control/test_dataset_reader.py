"""Unit tests for dataset_reader.parse_episode_spec (no ROS/rclpy dependency).

lerobot_control is a ROS2 ament_python package, not a uv workspace member —
rclpy isn't installed in this venv, but dataset_reader.py has no rclpy
dependency, so it's importable directly via the same sys.path convention
gt_replay_correctness_test.py already uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))

from lerobot_control.dataset_reader import parse_episode_spec  # noqa: E402


@pytest.mark.parametrize(
    "spec,total_episodes,expected",
    [
        ("0,1,2", 5, [0, 1, 2]),
        ("0:10", 15, list(range(0, 10))),
        (":3", 10, [0, 1, 2]),
        ("5:", 8, [5, 6, 7]),
        ("0,1:3,5", 10, [0, 1, 2, 5]),
        ("2,2,2", 5, [2]),  # duplicates dedup
        (" 0 , 1 , 2 ", 5, [0, 1, 2]),  # whitespace tolerated
        ("0:5", 5, [0, 1, 2, 3, 4]),  # end == total_episodes is in-bounds
    ],
)
def test_parse_episode_spec_valid(spec, total_episodes, expected):
    assert parse_episode_spec(spec, total_episodes) == expected


@pytest.mark.parametrize(
    "spec,total_episodes",
    [
        ("3:1", 10),        # start >= end
        ("3:3", 10),        # start == end
        ("abc", 10),        # not an integer
        ("1:abc", 10),      # range end not an integer
        ("-1", 10),         # negative index
        ("-1:3", 10),       # negative range start
        ("1:2:3", 10),      # more than one ':'
        ("20", 10),         # out of bounds (single index)
        ("0:20", 10),       # out of bounds (range end)
    ],
)
def test_parse_episode_spec_rejects(spec, total_episodes):
    with pytest.raises(ValueError):
        parse_episode_spec(spec, total_episodes)


def test_parse_episode_spec_empty_tokens_ignored():
    # A trailing/stray comma shouldn't blow up -- empty tokens are just skipped.
    assert parse_episode_spec("0,1,", 5) == [0, 1]
    assert parse_episode_spec("", 5) == []
