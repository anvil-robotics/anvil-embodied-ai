"""Data extraction from MCAP files"""

from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
from mcap.exceptions import McapError
from mcap.reader import make_reader as make_mcap_reader

from ..config import ActionSource, DataConfig
from ..exceptions import DataExtractionError
from ..utils.image_utils import decode_compressed_image, resize_image
from .constants import (
    ARM_PREFIX_TO_NAME,
    JOINT_NAME_SEPARATOR,
    LEADER_PREFIX,
    OBSERVATION_PREFIX,
    QUEST_JOINT_ORDER,
    quest_command_topic,
)
from .reader import McapReader

# Map source prefix -> role label, used only inside _buffer_joint_state to
# dispatch joints into the observation vs. command buffer.
_ROLE_BY_PREFIX = {OBSERVATION_PREFIX: "observation", LEADER_PREFIX: "action"}

# Canonical joint order for quest Float64MultiArray output (alphabetical).
_QUEST_CANONICAL_JOINT_NAMES: list[str] = sorted(QUEST_JOINT_ORDER)
_QUEST_REORDER: np.ndarray = np.array(
    [QUEST_JOINT_ORDER.index(name) for name in _QUEST_CANONICAL_JOINT_NAMES],
    dtype=np.intp,
)

# Buffer entry shapes:
#   observation: (ts, pos, vel, eff) — full kinematic state
#   command:     (ts, pos)           — target positions only
ObsEntry = tuple[float, np.ndarray, np.ndarray, np.ndarray]
CmdEntry = tuple[float, np.ndarray]


def parse_joint_name(joint_name: str) -> tuple[str, str, str] | None:
    """
    Parse `{source}_{arm}_{joint_id}` into (role, arm, joint_id).

    `source` must be one of `follower` (-> role "observation") or `leader`
    (-> role "action"). `arm` must be `l` (-> "left") or `r` (-> "right").

    Returns None if the name doesn't follow the convention — callers should
    skip such joints rather than fail.

    Examples:
        "follower_r_joint1"        -> ("observation", "right", "joint1")
        "leader_l_finger_joint1"   -> ("action", "left", "finger_joint1")
    """
    parts = joint_name.split(JOINT_NAME_SEPARATOR, 2)
    if len(parts) < 3:
        return None
    source, arm_prefix, joint_id = parts
    role = _ROLE_BY_PREFIX.get(source)
    arm = ARM_PREFIX_TO_NAME.get(arm_prefix)
    if role is None or arm is None:
        return None
    return role, arm, joint_id


def message_timestamp(message) -> float:
    """
    POSIX seconds for a message, preferring the ROS `header.stamp` over MCAP log_time.

    `header.stamp` is closer to when the data was actually captured (camera shutter
    fired, joint state was sampled); MCAP `log_time` is when the recorder received
    the message, which can lag by 10-20 ms for JPEG-compressed images. We fall back
    to log_time for messages without a header (e.g. std_msgs/Float64MultiArray).
    """
    stamp = getattr(getattr(message.ros_msg, "header", None), "stamp", None)
    if stamp is not None:
        return stamp.sec + stamp.nanosec * 1e-9
    return message.log_time_ns * 1e-9


class BufferedStreamExtractor:
    """
    Memory-efficient streaming extractor for MCAP conversion.

    Uses a sliding window buffer for time alignment while keeping memory bounded.
    Extracts both images and joint states (full teleop data).

    Time base: the first camera in `config.camera_topic_mapping` is the
    `main_cam` and drives the cursor + yield cadence. Other cameras and all
    joint streams are nearest-neighbor matched to its timestamps.

    Algorithm:
        1. Fill buffer to half_buffer, then start processing from cursor=0
        2. Buffer grows from half_buffer -> full_buffer (expanding lookahead)
        3. At full_buffer, switch to FIFO (steady state with +/-half_buffer lookahead)
        4. Flush remaining with shrinking lookahead

    Dual-Rate Buffering:
        - Camera buffer: ~30 Hz -> 150 frames in 5 sec
        - Joint state buffer: 100-250 Hz -> 500-1250 samples in 5 sec
        - Joint state samples aligned to camera timestamps via nearest-neighbor

    Example:
        extractor = BufferedStreamExtractor(config, buffer_seconds=5.0)
        for frame in extractor.extract_frames("recording.mcap", task="teleop"):
            dataset.add_frame(frame)
    """

    def __init__(
        self,
        config: DataConfig,
        buffer_seconds: float = 5.0,
        quiet: bool = False,
        progress_callback: Callable[[int], None] | None = None,
    ):
        """
        Initialize buffered stream extractor.

        Args:
            config: Data configuration (includes `frequency`, topics, camera mapping)
            buffer_seconds: Total buffer window size in seconds (default: 5.0)
            quiet: If True, suppress all print output (default: False)
            progress_callback: Called with frames_yielded count after each frame
        """
        self.config = config
        self.frequency = config.frequency
        self.frame_interval = 1.0 / self.frequency  # seconds between output frames (for subsampling)
        self.buffer_seconds = buffer_seconds
        self.half_buffer = int(buffer_seconds * self.frequency / 2)  # 75 frames for 5s @ 30Hz
        self.full_buffer = self.half_buffer * 2  # 150 frames
        self.target_size = tuple(config.image_resolution)  # (width, height)
        self.quiet = quiet
        self.progress_callback = progress_callback

        # For quest_teleop: map each command topic to its arm.
        self._quest_topic_to_arm: dict[str, str] = (
            {quest_command_topic(arm): arm for arm in config.arms}
            if config.action_source is ActionSource.quest_teleop
            else {}
        )

    @staticmethod
    def _check_quest_topics_present(mcap_path: Path, expected: set[str]) -> None:
        """Raise DataExtractionError when none of the quest command topics are in the MCAP.

        Uses the MCAP footer index (O(1)) — does not scan all messages.
        """
        try:
            with open(mcap_path, "rb") as _f:
                _summary = make_mcap_reader(_f).get_summary()
        except (McapError, OSError):
            return  # Cannot read summary — let streaming fail naturally

        if _summary is None:
            return

        available = {ch.topic for ch in _summary.channels.values()}
        if expected and expected.isdisjoint(available):
            raise DataExtractionError(
                f"\n[ACTION SOURCE ERROR] action_source=quest_teleop, but none of the "
                f"expected command topics were found in this MCAP.\n"
                f"  Expected: {sorted(expected)}\n"
                f"  Available in MCAP: {sorted(available)}\n\n"
                "Fix options:\n"
                "  1. Set action_source: future_observations in config to derive actions "
                "from observations.\n"
                "  2. Record a new session that captures the action-command topics.\n"
                "  3. Check the `arms` field in your config matches what was recorded.\n"
            )

    def extract_frames(
        self,
        mcap_path: Path,
        task: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator with bounded memory usage, yielding complete time-aligned frames.

        Args:
            mcap_path: Path to MCAP file
            task: Task description stored in every frame (e.g. "fold the towel").
                  LeRobot requires this; "" passes validation but provides no
                  task-conditioning signal.

        Yields:
            Dictionary with aligned frame data:
            {
                'observation.images.{camera}': np.ndarray (H, W, C),
                'observation.state': np.ndarray,
                'observation.velocity': np.ndarray,
                'observation.effort': np.ndarray,
                'action': np.ndarray,
                'task': str,
            }
        """
        if not self.quiet:
            print(f"[BufferedStream] Reading MCAP: {mcap_path}")
            print(f"[BufferedStream] Buffer: {self.full_buffer} frames ({self.half_buffer}x2)")

        reader = McapReader(str(mcap_path))
        topic_to_cam = dict(self.config.camera_topic_mapping)

        # Per-camera image buffers, keyed by camera name.
        camera_buffers: dict[str, deque] = {
            cam: deque() for cam in self.config.camera_topic_mapping.values()
        }
        # Per-arm joint buffers, split by purpose (see ObsEntry / CmdEntry above).
        joint_observation_buffers: dict[str, deque] = {}
        joint_command_buffers: dict[str, deque] = {}

        # Build subscription topic list
        all_topics = [self.config.camera_topic_mapping.keys()] + [self.config.robot_state_topic]
        if self.config.action_source is ActionSource.quest_teleop:
            expected_cmd_topics = set(self._quest_topic_to_arm.keys())
            all_topics.extend(expected_cmd_topics)
            self._check_quest_topics_present(mcap_path, expected_cmd_topics)
        elif self.config.action_source is ActionSource.future_observations:
            assert self.config.action_n_step is not None
            n = self.config.action_n_step
            ms = int(n / self.frequency * 1000)
            if not self.quiet:
                print(
                    f"\n[ACTION SOURCE] action_source=future_observations — "
                    f"action[t] = observation[t + {n}] (~{ms} ms lookahead at {self.frequency} Hz)\n"
                )

        # We arbitrarily pick one of the cameras to designate the "main cam", which will key off observation frames
        main_cam = next(iter(self.config.camera_topic_mapping.values()))

        # cursor is the index (in main_cam buffer) of the next frame to consider
        # Yielding requires us having half_buffer frames AHEAD of cursor, ensuring we have enough lookahead data
        # to find the closest
        cursor = 0
        frames_yielded = 0
        frames_dropped = 0
        next_yield_ts: float | None = None  # subsampling target; anchored on first yield attempt

        for message in reader.read_messages(topics=all_topics):
            topic = message.channel.topic

            if topic == self.config.robot_state_topic:
                self._buffer_joint_state(message, joint_observation_buffers, joint_command_buffers)
                continue

            if topic in self._quest_topic_to_arm:
                arm = self._quest_topic_to_arm[topic]
                self._buffer_action_command(message, arm, joint_command_buffers)
                continue

            if topic not in topic_to_cam:
                continue
            cam_name = topic_to_cam[topic]
            ros_msg = message.ros_msg
            img = decode_compressed_image(ros_msg.data, ros_msg.format)
            camera_buffers[cam_name].append((message_timestamp(message), img))

            main_buffer_len = len(camera_buffers[main_cam])

            # Lookahead not ready yet: need half_buffer frames past cursor
            if main_buffer_len < self.half_buffer + cursor:
                continue

            frame, next_yield_ts, attempted = self._try_yield_at_cursor(
                cursor,
                camera_buffers,
                joint_observation_buffers,
                joint_command_buffers,
                main_cam,
                task,
                next_yield_ts,
            )
            if frame is not None:
                yield frame
                frames_yielded += 1
            elif attempted:
                frames_dropped += 1

            cursor += 1

            # Report progress every cursor advance (including subsampled/dropped frames).
            if self.progress_callback:
                self.progress_callback(frames_yielded)
            elif not self.quiet and frames_yielded % 100 == 0:
                print(f"[BufferedStream] Processed {frames_yielded} frames...")

            # Steady state: buffer holds full_buffer frames (~±half_buffer around cursor).
            # Popleft + decrement cursor keeps cursor pointing at the same logical frame.
            if main_buffer_len > self.full_buffer:
                for buf in camera_buffers.values():
                    if len(buf) > 0:
                        buf.popleft()
                if camera_buffers[main_cam]:
                    new_oldest_ts = camera_buffers[main_cam][0][0]
                    self._sync_joint_buffers(
                        joint_observation_buffers,
                        joint_command_buffers,
                        new_oldest_ts,
                    )
                cursor -= 1

        # Flush: walk remaining frames with shrinking lookahead.
        if not self.quiet:
            print("[BufferedStream] Flushing remaining buffer...")
        while cursor < len(camera_buffers[main_cam]):
            frame, next_yield_ts, attempted = self._try_yield_at_cursor(
                cursor,
                camera_buffers,
                joint_observation_buffers,
                joint_command_buffers,
                main_cam,
                task,
                next_yield_ts,
            )
            if frame is not None:
                yield frame
                frames_yielded += 1
                if self.progress_callback:
                    self.progress_callback(frames_yielded)
            elif attempted:
                frames_dropped += 1
            cursor += 1

        if not self.quiet:
            if frames_dropped:
                print(
                    f"[BufferedStream] [OK] Extracted {frames_yielded} frames total "
                    f"({frames_dropped} dropped during alignment)"
                )
            else:
                print(f"[BufferedStream] [OK] Extracted {frames_yielded} frames total")

        if frames_yielded == 0:
            self._log_empty_diagnostics(
                camera_buffers,
                joint_observation_buffers,
                joint_command_buffers,
                frames_dropped,
            )

    def _try_yield_at_cursor(
        self,
        cursor: int,
        camera_buffers: dict[str, deque],
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
        main_cam: str,
        task: str,
        next_yield_ts: float | None,
    ) -> tuple[dict[str, Any] | None, float | None, bool]:
        """
        Attempt to produce a yieldable frame at `cursor`.

        Returns (frame_or_None, advanced_next_yield_ts, alignment_attempted):
          - frame_or_None: aligned frame, or None if subsampled away / alignment failed
          - advanced_next_yield_ts: subsampling target; advances iff a frame is produced
          - alignment_attempted: True if alignment ran (caller uses this to count drops)
        """
        frame_ts: float = camera_buffers[main_cam][cursor][0]

        # Anchor subsampling clock to the first processed frame so output cadence is exactly self.frequency.
        yield_target = next_yield_ts if next_yield_ts is not None else frame_ts

        if frame_ts < yield_target:
            return None, yield_target, False

        frame = self._align_frame_at_cursor(
            camera_buffers,
            joint_observation_buffers,
            joint_command_buffers,
            cursor,
            main_cam,
            task,
        )
        if frame is None:
            return None, yield_target, True
        return frame, yield_target + self.frame_interval, True

    def _align_frame_at_cursor(
        self,
        camera_buffers: dict[str, deque],
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
        cursor: int,
        main_cam: str,
        task: str,
    ) -> dict[str, Any] | None:
        """
        Align frame at cursor position using entire buffer for matching.

        Returns aligned frame dictionary, or None if any required data is missing.
        """
        main_ts, main_img = camera_buffers[main_cam][cursor]
        frame: dict[str, Any] = {
            f"observation.images.{main_cam}": resize_image(main_img, self.target_size)
        }

        # Nearest-neighbor match each non-main camera to main_ts.
        for cam_name, buffer in camera_buffers.items():
            if cam_name == main_cam:
                continue
            if len(buffer) == 0:
                return None
            nearest_idx = self._find_nearest_in_buffer(buffer, main_ts)
            if nearest_idx is None:
                return None
            _, img = buffer[nearest_idx]
            frame[f"observation.images.{cam_name}"] = resize_image(img, self.target_size)

        if joint_observation_buffers or joint_command_buffers:
            joint_aligned = self._align_joint_states(
                joint_observation_buffers, joint_command_buffers, main_ts
            )
            if joint_aligned is None:
                return None
            frame.update(joint_aligned)

        frame["task"] = task
        return frame

    def _align_joint_states(
        self,
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
        target_ts: float,
    ) -> dict[str, Any] | None:
        """
        Align joint streams to `target_ts`. Concatenates per-arm features in sorted
        arm order (left then right).

        Observation entries are (ts, pos, vel, eff); command entries are (ts, pos).
        `observation.state/velocity/effort` come from the observation buffers at
        `target_ts`. `action` comes from:
          - command buffers at target_ts, when action_source is quest_teleop or leader
          - observation buffers at target_ts + N*frame_interval, when action_source
            is future_observations (action[t] = observation[t + N])

        Returns None if any required arm is missing data.
        """
        arms = sorted(self.config.arms)

        # Observation lookup at target_ts: always needed (state/velocity/effort, and
        # also the source for action when action_source == future_observations).
        obs_at_target: dict[str, ObsEntry] = {}
        for arm in arms:
            buf = joint_observation_buffers.get(arm)
            if not buf:
                return None
            idx = self._find_nearest_in_buffer(buf, target_ts)
            if idx is None:
                return None
            obs_at_target[arm] = buf[idx]

        # Action lookup, branched on action_source.
        action_positions: dict[str, np.ndarray] = {}
        if self.config.action_source is ActionSource.future_observations:
            assert self.config.action_n_step is not None
            action_ts = target_ts + self.frame_interval * self.config.action_n_step
            for arm in arms:
                buf = joint_observation_buffers[arm]  # non-empty per loop above
                idx = self._find_nearest_in_buffer(buf, action_ts)
                if idx is None:
                    return None
                _, pos, _, _ = buf[idx]
                action_positions[arm] = pos
        else:
            for arm in arms:
                buf = joint_command_buffers.get(arm)
                if not buf:
                    return None
                idx = self._find_nearest_in_buffer(buf, target_ts)
                if idx is None:
                    return None
                _, pos = buf[idx]
                action_positions[arm] = pos

        return {
            "observation.state": np.concatenate([obs_at_target[a][1] for a in arms]),
            "observation.velocity": np.concatenate([obs_at_target[a][2] for a in arms]),
            "observation.effort": np.concatenate([obs_at_target[a][3] for a in arms]),
            "action": np.concatenate([action_positions[a] for a in arms]),
        }

    def _find_nearest_in_buffer(
        self,
        buffer: deque,
        target_ts: float,
    ) -> int | None:
        """
        Find index of frame with nearest timestamp in buffer.

        Args:
            buffer: Deque of (timestamp, ...) tuples (first element is timestamp)
            target_ts: Target timestamp to match

        Returns:
            Index of nearest frame, or None if buffer is empty
        """
        if not buffer:
            return None

        # Linear search for nearest (buffer is small, typically 150-1000 items)
        min_diff = float("inf")
        nearest_idx = 0

        for i, item in enumerate(buffer):
            ts = item[0]  # First element is always timestamp
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i

        return nearest_idx

    def _buffer_joint_state(
        self,
        message,
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
    ) -> None:
        """
        Parse a JointState message and append per-arm samples.

        Follower joints route to `joint_observation_buffers` with (ts, pos, vel, eff).
        Leader joints (used when action_source == leader) route to
        `joint_command_buffers` with (ts, pos). Joint values within each arm are
        sorted alphabetically by joint_id for deterministic canonical order.
        """
        ros_msg = message.ros_msg
        timestamp = message_timestamp(message)

        # Group by (role, arm) so we can sort each group's parallel arrays together.
        grouped: dict[tuple[str, str], dict] = {}
        for i, joint_name in enumerate(ros_msg.name):
            parsed = parse_joint_name(joint_name)
            if parsed is None:
                continue
            role, arm, joint_id = parsed

            key = (role, arm)
            g = grouped.setdefault(
                key, {"joint_ids": [], "position": [], "velocity": [], "effort": []}
            )
            g["joint_ids"].append(joint_id)
            if ros_msg.position and i < len(ros_msg.position):
                g["position"].append(ros_msg.position[i])
            if ros_msg.velocity and i < len(ros_msg.velocity):
                g["velocity"].append(ros_msg.velocity[i])
            if ros_msg.effort and i < len(ros_msg.effort):
                g["effort"].append(ros_msg.effort[i])

        for g in grouped.values():
            order = sorted(range(len(g["joint_ids"])), key=lambda idx: g["joint_ids"][idx])
            for field in ("joint_ids", "position", "velocity", "effort"):
                g[field] = [g[field][idx] for idx in order] if g[field] else []

        for (role, arm), g in grouped.items():
            pos = np.array(g["position"], dtype=np.float32)
            if role == "observation":
                vel = np.array(g["velocity"], dtype=np.float32)
                eff = np.array(g["effort"], dtype=np.float32)
                joint_observation_buffers.setdefault(arm, deque()).append(
                    (timestamp, pos, vel, eff)
                )
            else:
                joint_command_buffers.setdefault(arm, deque()).append((timestamp, pos))

    def _buffer_action_command(
        self,
        message,
        arm: str,
        joint_command_buffers: dict[str, deque],
    ) -> None:
        """
        Parse a Float64MultiArray action command and append to `joint_command_buffers`.

        Values arrive in QUEST_JOINT_ORDER and are reordered to canonical
        (alphabetical) joint order to match observation grouping.
        """
        timestamp = message_timestamp(message)
        positions = np.array(message.ros_msg.data, dtype=np.float32)[_QUEST_REORDER]
        joint_command_buffers.setdefault(arm, deque()).append((timestamp, positions))

    def _sync_joint_buffers(
        self,
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
        oldest_camera_ts: float,
    ) -> None:
        """Drop joint samples older than the oldest remaining camera frame."""
        for buf in joint_observation_buffers.values():
            while buf and buf[0][0] < oldest_camera_ts:
                buf.popleft()
        for buf in joint_command_buffers.values():
            while buf and buf[0][0] < oldest_camera_ts:
                buf.popleft()

    def _log_empty_diagnostics(
        self,
        camera_buffers: dict[str, deque],
        joint_observation_buffers: dict[str, deque],
        joint_command_buffers: dict[str, deque],
        frames_dropped: int,
    ) -> None:
        """Print diagnostics when extraction produced 0 frames."""
        cam_counts = {cam: len(buf) for cam, buf in camera_buffers.items()}
        obs_counts = {arm: len(buf) for arm, buf in joint_observation_buffers.items()}
        cmd_counts = {arm: len(buf) for arm, buf in joint_command_buffers.items()}

        print(
            f"[BufferedStream] WARNING: 0 frames produced "
            f"({frames_dropped} dropped during alignment) — diagnostics:"
        )
        print(f"  Camera buffers:        {cam_counts}")
        print(f"  Joint obs buffers:     {obs_counts or '(empty)'}")
        print(f"  Joint command buffers: {cmd_counts or '(empty)'}")

        if not cam_counts or all(c == 0 for c in cam_counts.values()):
            print("  -> No camera images found. Check these topics exist in the MCAP:")
            for t in self.config.camera_topic_mapping:
                print(f"       {t}")
        if not obs_counts:
            print(
                f"  -> No joint state data. Check robot_state_topic: "
                f"{self.config.robot_state_topic}"
            )
        elif self.config.action_source is ActionSource.quest_teleop and not cmd_counts:
            print("  -> No action data from quest command topics:")
            for t in self._quest_topic_to_arm:
                print(f"       {t}")
        elif self.config.action_source is ActionSource.leader and not cmd_counts:
            print("  -> No 'leader_*' joints found in /joint_states.")
