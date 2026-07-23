"""Motion-aware episode-window detection for LeRobot datasets.

The detector operates only on low-dimensional actions. The selected window can
then be applied identically to actions, observations, and every camera stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

StartMode = Literal["motion", "displacement", "gripper"]


@dataclass(frozen=True)
class MotionTrimConfig:
    """Thresholds used to find the task-relevant portion of an episode."""

    start_mode: StartMode = "motion"
    baseline_frames: int = 15
    sustain_frames: int = 5
    arm_threshold: float = 0.02
    displacement_threshold: float = 0.10
    gripper_threshold: float = 0.01
    start_offset_frames: int = -10
    end_arm_threshold: float = 0.02
    end_gripper_threshold: float = 0.005
    end_postroll_frames: int = 10
    min_frames: int = 30

    def validate(self) -> None:
        if self.start_mode not in {"motion", "displacement", "gripper"}:
            raise ValueError(f"Unsupported start mode: {self.start_mode}")
        if self.baseline_frames < 1:
            raise ValueError("baseline_frames must be at least 1")
        if self.sustain_frames < 1:
            raise ValueError("sustain_frames must be at least 1")
        if self.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        for name in (
            "arm_threshold",
            "displacement_threshold",
            "gripper_threshold",
            "end_arm_threshold",
            "end_gripper_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MotionTrimWindow:
    """A half-open frame window plus the events that produced it."""

    start: int
    end: int
    start_event: int
    final_settle_event: int
    start_event_found: bool
    final_settle_found: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def joint_groups(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return arm and gripper indices inferred from action feature names."""
    if not names:
        raise ValueError("The action feature must include joint names")
    gripper = np.asarray(
        [index for index, name in enumerate(names) if "gripper" in name.lower()], dtype=np.int64
    )
    arm = np.asarray(
        [index for index, name in enumerate(names) if "gripper" not in name.lower()], dtype=np.int64
    )
    if arm.size == 0:
        raise ValueError("No arm joints found in action feature names")
    return arm, gripper


def _sustained_mask(mask: np.ndarray, run_length: int) -> np.ndarray:
    """Keep samples that belong to a true run of at least run_length."""
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask)
    run_start: int | None = None
    for index, active in enumerate(np.append(mask, False)):
        if active and run_start is None:
            run_start = index
        elif not active and run_start is not None:
            if index - run_start >= run_length:
                result[run_start:index] = True
            run_start = None
    return result


def _deviation_signal(
    values: np.ndarray,
    baseline: np.ndarray,
    arm_indices: np.ndarray,
    gripper_indices: np.ndarray,
    arm_threshold: float,
    gripper_threshold: float,
) -> np.ndarray:
    arm_active = (
        np.max(np.abs(values[:, arm_indices] - baseline[arm_indices]), axis=1) > arm_threshold
    )
    if gripper_indices.size == 0:
        return arm_active
    gripper_active = (
        np.max(np.abs(values[:, gripper_indices] - baseline[gripper_indices]), axis=1)
        > gripper_threshold
    )
    return arm_active | gripper_active


def detect_motion_window(
    actions: np.ndarray,
    action_names: Sequence[str],
    config: MotionTrimConfig,
) -> MotionTrimWindow:
    """Detect a task window using sustained departure and final settling.

    start is inclusive and end is exclusive. A positive start_offset_frames
    trims farther into the task; a negative value keeps pre-event context.
    """
    config.validate()
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"actions must have shape (frames, joints), got {actions.shape}")
    arm_indices, gripper_indices = joint_groups(action_names)
    if actions.shape[1] != len(action_names):
        raise ValueError(
            f"Action width {actions.shape[1]} does not match {len(action_names)} names"
        )

    frame_count = actions.shape[0]
    baseline_count = min(config.baseline_frames, frame_count)
    initial = np.median(actions[:baseline_count], axis=0)
    final = np.median(actions[-baseline_count:], axis=0)

    if config.start_mode == "gripper":
        if gripper_indices.size == 0:
            raise ValueError("start_mode=gripper requires at least one gripper feature")
        start_signal = (
            np.max(np.abs(actions[:, gripper_indices] - initial[gripper_indices]), axis=1)
            > config.gripper_threshold
        )
    else:
        threshold = (
            config.arm_threshold if config.start_mode == "motion" else config.displacement_threshold
        )
        start_signal = (
            np.max(np.abs(actions[:, arm_indices] - initial[arm_indices]), axis=1) > threshold
        )

    sustained_start = _sustained_mask(start_signal, config.sustain_frames)
    start_indices = np.flatnonzero(sustained_start)
    start_found = start_indices.size > 0
    start_event = int(start_indices[0]) if start_found else 0
    start = int(np.clip(start_event + config.start_offset_frames, 0, frame_count - 1))

    not_settled = _deviation_signal(
        actions,
        final,
        arm_indices,
        gripper_indices,
        config.end_arm_threshold,
        config.end_gripper_threshold,
    )
    sustained_not_settled = _sustained_mask(not_settled, config.sustain_frames)
    unsettled_indices = np.flatnonzero(sustained_not_settled)
    settle_found = unsettled_indices.size > 0
    final_settle_event = int(unsettled_indices[-1] + 1) if settle_found else frame_count
    end = min(frame_count, final_settle_event + config.end_postroll_frames)

    if end - start < config.min_frames:
        end = min(frame_count, start + config.min_frames)
    if end - start < config.min_frames:
        start = max(0, end - config.min_frames)

    return MotionTrimWindow(
        start=start,
        end=end,
        start_event=start_event,
        final_settle_event=final_settle_event,
        start_event_found=start_found,
        final_settle_found=settle_found,
    )
