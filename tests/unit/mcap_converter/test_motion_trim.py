"""Tests for motion-aware LeRobot episode trimming."""

import json
from pathlib import Path

import numpy as np
import pytest

from mcap_converter.cli.trim_dataset import _reference_preroll
from mcap_converter.core.motion_trim import MotionTrimConfig, detect_motion_window

NAMES = [
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_gripper.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_gripper.pos",
]


def _fold_like_actions() -> np.ndarray:
    actions = np.zeros((120, len(NAMES)), dtype=np.float32)
    actions[20:, 0] = np.linspace(0.0, 0.8, 100)
    actions[35:, 3] = np.linspace(0.0, -0.6, 85)
    actions[60:, 2] = 0.03
    actions[70:, 5] = 0.025
    actions[100:] = actions[99]
    return actions


def test_motion_mode_keeps_preroll_and_final_context() -> None:
    window = detect_motion_window(
        _fold_like_actions(),
        NAMES,
        MotionTrimConfig(
            start_mode="motion",
            arm_threshold=0.02,
            start_offset_frames=-4,
            end_postroll_frames=6,
        ),
    )

    assert window.start_event == 23
    assert window.start == 19
    assert window.end == 103
    assert window.start_event_found
    assert window.final_settle_found


def test_gripper_mode_aligns_to_first_sustained_interaction() -> None:
    actions = _fold_like_actions()
    actions[10, 2] = 0.02  # One-frame noise must not become the event.
    window = detect_motion_window(
        actions,
        NAMES,
        MotionTrimConfig(start_mode="gripper", start_offset_frames=-5),
    )

    assert window.start_event == 60
    assert window.start == 55


def test_displacement_mode_trims_beyond_home() -> None:
    window = detect_motion_window(
        _fold_like_actions(),
        NAMES,
        MotionTrimConfig(
            start_mode="displacement",
            displacement_threshold=0.20,
            start_offset_frames=0,
        ),
    )

    assert window.start_event > 40
    assert window.start == window.start_event


def test_action_names_are_required() -> None:
    with pytest.raises(ValueError, match="joint names"):
        detect_motion_window(np.zeros((10, 2)), [], MotionTrimConfig())


def test_reference_preroll_uses_median_detected_event(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "source": {"repo_id": "reference/folds"},
                "episodes": [
                    {"start_event": 20, "start_event_found": True},
                    {"start_event": 53, "start_event_found": True},
                    {"start_event": 999, "start_event_found": False},
                ],
            }
        )
    )

    frames, details = _reference_preroll(path)

    assert frames == 37
    assert details["median_pre_gripper_frames"] == 36.5
    assert details["repo_id"] == "reference/folds"
