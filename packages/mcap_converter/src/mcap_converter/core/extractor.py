"""Data extraction from MCAP files"""

import logging
from collections import deque
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from ..config.schema import DEFAULT_DATA_CONFIG, DataConfig, JointNamePattern
from ..exceptions import DataExtractionError
from ..utils.image_utils import decode_compressed_image, decode_image
from .reader import McapReader

log = logging.getLogger(__name__)

# =============================================================================
# Shared Utilities
# =============================================================================


def parse_joint_name(
    joint_name: str,
    pattern: JointNamePattern,
) -> Optional[Tuple[str, str, str]]:
    """
    Parse joint name to extract role, robot, and joint_id.

    This is a shared utility used by both DataExtractor and BufferedStreamExtractor.

    Args:
        joint_name: Full joint name (e.g., "leader_r_joint1")
        pattern: JointNamePattern configuration

    Returns:
        Tuple of (role, robot, joint_id):
            - role: "observation" or "action"
            - robot: Robot prefix (e.g., "right", "left") or empty string
            - joint_id: Remaining joint identifier (e.g., "joint1")

    Raises:
        DataExtractionError: If joint name doesn't match expected pattern
    """
    sep = pattern.separator

    role = None
    robot = ""
    joint_id = joint_name  # fallback
    remaining = ""

    # Find role prefix (first match)
    for prefix, role_name in pattern.role_prefix.items():
        if joint_name.startswith(prefix + sep):
            role = role_name
            remaining = joint_name[len(prefix) + len(sep) :]
            break

    if role is None:
        raise DataExtractionError(
            f"Cannot determine role for joint '{joint_name}'. "
            f"Expected prefix from: {list(pattern.role_prefix.keys())}"
        )

    # Find robot prefix (optional, first part after role)
    parts = remaining.split(sep, 1)
    if parts and parts[0] in pattern.robot_prefix:
        robot = pattern.robot_prefix[parts[0]]
        joint_id = parts[1] if len(parts) > 1 else parts[0]
    elif parts and pattern.robot_prefix and len(parts) > 1:
        # arms map is configured but this arm identifier is not in it — skip joint
        return None
    else:
        joint_id = remaining

    return role, robot, joint_id


def message_timestamp(message) -> float:
    """
    POSIX seconds for a message, preferring the ROS `header.stamp` over MCAP log_time.
    Falls back to log_time for messages without a header (e.g. std_msgs/Float64MultiArray).
    """
    ros_msg = message.ros_msg
    if hasattr(ros_msg, "header"):
        stamp = ros_msg.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9
    return message.log_time_ns * 1e-9


# =============================================================================
# DataExtractor - Batch Mode (loads entire episode into memory)
# =============================================================================


class DataExtractor:
    """DEPRECATED: batch-mode extractor. Use BufferedStreamExtractor instead.

    This class was not updated for the unified DataConfig schema
    (action_topics is now arm-keyed, not topic-keyed). It remains for
    backward compatibility only and should not be used in new code.

    Extract joint states and images from MCAP files

    Supports new single-topic architecture where all joints are in one JointState
    message, differentiated by joint names.

    Joint Name Convention:
        {role}_{robot}_{joint_id}
        Examples: "leader_r_joint1", "follower_l_joint3"

    Example:
        config = DataConfig()
        extractor = DataExtractor(config)
        data = extractor.extract_episode("recording.mcap")

        # Single robot:
        print(data['joint_states_observation']['position'])
        print(data['joint_states_action']['position'])

        # Multi-robot:
        print(data['joint_states_right_observation']['position'])
        print(data['joint_states_left_action']['position'])
    """

    def __init__(self, config: DataConfig = DEFAULT_DATA_CONFIG):
        """
        Initialize data extractor

        Args:
            config: Data configuration specifying topics and features
        """
        self.config = config
        # Cache for action topic reorder permutations: {topic: np.ndarray}
        self._action_reorder_cache: Dict[str, np.ndarray] = {}

    def _parse_joint_name(self, joint_name: str) -> Optional[Tuple[str, str, str]]:
        """Parse joint name using shared utility."""
        return parse_joint_name(joint_name, self.config.joint_name_pattern)

    def _get_joint_state_key(self, role: str, robot: str) -> str:
        """
        Generate key for extracted data dictionary.

        Args:
            role: "observation" or "action"
            robot: Robot prefix (e.g., "right") or empty string

        Returns:
            Key string (e.g., "joint_states_right_observation" or "joint_states_action")
        """
        if robot:
            return f"joint_states_{robot}_{role}"
        return f"joint_states_{role}"

    def extract_episode(self, mcap_path: str) -> Dict[str, Any]:
        """
        Extract all data from one MCAP episode

        Args:
            mcap_path: Path to MCAP file

        Returns:
            Dictionary with extracted data:
            {
                'joint_states_observation': {  # or 'joint_states_right_observation' for multi-robot
                    'timestamp': np.ndarray,
                    'joint_names': List[str],
                    'position': np.ndarray,
                    'velocity': np.ndarray,
                    'effort': np.ndarray,
                },
                'joint_states_action': {
                    ... same structure ...
                },
                'head': {  # camera name from config
                    'timestamp': np.ndarray,
                    'image_data': np.ndarray,
                    'encoding': str,
                    'height': int,
                    'width': int,
                },
                ...
            }
        """
        print(f"Reading MCAP file: {mcap_path}")

        # Initialize data structure
        extracted_data = self._initialize_data_structure()

        # Read messages
        reader = McapReader(mcap_path)

        # Build list of interested topics
        interested_topics = [self.config.robot_state_topic] + self.config.camera_topics

        # Also include compressed image topics (append /compressed to each camera topic)
        compressed_topics = [t + "/compressed" for t in self.config.camera_topics]
        interested_topics.extend(compressed_topics)

        # Build mapping from compressed topic to camera name
        self._compressed_topic_mapping = {
            t + "/compressed": self.config.camera_topic_mapping[t]
            for t in self.config.camera_topics
        }

        # Include action command topics if configured (quest teleop mode).
        # Use action_command_topics (topic-keyed) not action_topics (arm-keyed).
        if self.config.action_command_topics:
            interested_topics.extend(self.config.action_command_topics.keys())

        # Also include legacy topics if configured
        if self.config.robot_state_topics:
            interested_topics.extend(self.config.robot_state_topics)

        for message in reader.read_messages(topics=interested_topics):
            topic = message.channel.topic

            # Handle single robot state topic (new architecture)
            if topic == self.config.robot_state_topic:
                self._extract_joint_state_single_topic(message, extracted_data)
            # Handle action command topics (quest teleop mode)
            elif self.config.action_command_topics and topic in self.config.action_command_topics:
                self._extract_action_command(message, topic, extracted_data)
            # Handle legacy multi-topic architecture
            elif self.config.robot_state_topics and topic in self.config.robot_state_topics:
                self._extract_joint_state_legacy(message, extracted_data)
            # Handle camera topics (detect CompressedImage vs Image by attribute)
            elif topic in self.config.camera_topics or topic in self._compressed_topic_mapping:
                if hasattr(message.ros_msg, "format"):
                    self._extract_compressed_image(message, extracted_data)
                else:
                    self._extract_image(message, extracted_data)

        # Convert lists to numpy arrays
        self._convert_to_arrays(extracted_data)

        # Print summary
        self._print_extraction_summary(extracted_data)

        return extracted_data

    def _initialize_data_structure(self) -> Dict[str, Any]:
        """Initialize empty data structure"""
        extracted_data = {}

        # Initialize camera data
        for topic in self.config.camera_topics:
            cam_name = self.config.camera_topic_mapping[topic]
            extracted_data[cam_name] = {
                "timestamp": [],
                "image_data": [],
                "encoding": None,
                "height": None,
                "width": None,
            }

        # Note: Joint state structures are created dynamically during extraction
        # because we don't know the robot prefixes until we see the joint names

        return extracted_data

    def _extract_joint_state_single_topic(self, message, extracted_data: Dict):
        """
        Extract joint state from single JointState message containing all joints.

        Parses joint names to determine role and robot, then groups data accordingly.
        Joints are sorted by joint_id for deterministic canonical ordering.
        """
        ros_msg = message.ros_msg
        time_s = message_timestamp(message)

        # Group joints by (role, robot)
        grouped_data: Dict[Tuple[str, str], Dict] = {}

        for i, joint_name in enumerate(ros_msg.name):
            try:
                result = self._parse_joint_name(joint_name)
            except DataExtractionError as e:
                print(f"Warning: {e}")
                continue
            if result is None:
                continue
            role, robot, joint_id = result

            key = (role, robot)
            if key not in grouped_data:
                grouped_data[key] = {
                    "joint_names": [],
                    "position": [],
                    "velocity": [],
                    "effort": [],
                }

            grouped_data[key]["joint_names"].append(joint_id)

            if ros_msg.position and i < len(ros_msg.position):
                grouped_data[key]["position"].append(ros_msg.position[i])
            if ros_msg.velocity and i < len(ros_msg.velocity):
                grouped_data[key]["velocity"].append(ros_msg.velocity[i])
            if ros_msg.effort and i < len(ros_msg.effort):
                grouped_data[key]["effort"].append(ros_msg.effort[i])

        # Sort each group by joint_id for canonical ordering
        for key, data in grouped_data.items():
            sort_indices = sorted(
                range(len(data["joint_names"])),
                key=lambda idx: data["joint_names"][idx],
            )
            data["joint_names"] = [data["joint_names"][idx] for idx in sort_indices]
            data["position"] = [data["position"][idx] for idx in sort_indices]
            data["velocity"] = [data["velocity"][idx] for idx in sort_indices]
            data["effort"] = [data["effort"][idx] for idx in sort_indices]

        # Store in extracted_data with structured keys
        for (role, robot), data in grouped_data.items():
            key = self._get_joint_state_key(role, robot)

            if key not in extracted_data:
                extracted_data[key] = {
                    "timestamp": [],
                    "joint_names": data["joint_names"],  # Set once (assumes consistent ordering)
                    "position": [],
                    "velocity": [],
                    "effort": [],
                }

            extracted_data[key]["timestamp"].append(time_s)
            extracted_data[key]["position"].append(data["position"])
            extracted_data[key]["velocity"].append(data["velocity"])
            extracted_data[key]["effort"].append(data["effort"])

    def _extract_action_command(self, message, topic: str, extracted_data: Dict):
        """
        Extract action from a command topic (quest teleop mode).

        Command topics publish std_msgs/Float64MultiArray with joint positions.
        Float64MultiArray has no header — message_timestamp falls back to log_time.
        Positions are reordered to canonical (sorted) joint order using joint_order
        from the ActionTopicConfig.

        Args:
            message: MCAP message (Float64MultiArray)
            topic: The ROS topic name
            extracted_data: Data dictionary to populate
        """
        ros_msg = message.ros_msg
        time_s = message_timestamp(message)

        # Get action topic config
        topic_cfg = self.config.action_topics[topic]
        robot = topic_cfg.arm
        key = self._get_joint_state_key("action", robot)

        # Extract position data from Float64MultiArray.data
        positions = list(ros_msg.data)

        # Compute reorder permutation on first message (cached per topic)
        if topic not in self._action_reorder_cache:
            if topic_cfg.joint_order:
                # Sort joint_order alphabetically to match canonical observation order
                self._action_reorder_cache[topic] = np.array(
                    sorted(
                        range(len(topic_cfg.joint_order)),
                        key=lambda idx: topic_cfg.joint_order[idx],
                    ),
                    dtype=np.intp,
                )
            else:
                # No joint_order specified — identity permutation (no reorder)
                self._action_reorder_cache[topic] = np.arange(len(positions), dtype=np.intp)

        reorder = self._action_reorder_cache[topic]

        if key not in extracted_data:
            # Use canonical (sorted) joint names
            if topic_cfg.joint_order:
                joint_names = sorted(topic_cfg.joint_order)
            else:
                joint_names = [f"joint{i}" for i in range(len(positions))]
            extracted_data[key] = {
                "timestamp": [],
                "joint_names": joint_names,
                "position": [],
                "velocity": [],
                "effort": [],
            }

        # Reorder positions to canonical order
        pos_array = np.array(positions, dtype=np.float64)
        extracted_data[key]["timestamp"].append(time_s)
        extracted_data[key]["position"].append(pos_array[reorder].tolist())
        # Float64MultiArray only contains position commands
        extracted_data[key]["velocity"].append([0.0] * len(positions))
        extracted_data[key]["effort"].append([0.0] * len(positions))

    def _extract_joint_state_legacy(self, message, extracted_data: Dict):
        """
        Extract joint state from legacy multi-topic architecture.

        For backward compatibility with robot_state_topics configuration.
        """
        # Determine if follower (observation) or leader (action)
        if message.channel.topic == self.config.robot_state_topics[0]:
            joint_key = "joint_states_observation"
        else:
            joint_key = "joint_states_action"

        # Initialize if not exists
        if joint_key not in extracted_data:
            extracted_data[joint_key] = {
                "timestamp": [],
                "joint_names": [],
            }

        # Extract data
        ros_msg = message.ros_msg
        time_ns = (ros_msg.header.stamp.sec * 1e9 + ros_msg.header.stamp.nanosec) / 1e9
        extracted_data[joint_key]["timestamp"].append(time_ns)

        # Store joint names (once) - JointState uses 'name' not 'joint_names'
        if not extracted_data[joint_key]["joint_names"]:
            extracted_data[joint_key]["joint_names"] = list(ros_msg.name)

        # Extract position, velocity, effort directly from JointState arrays
        for field_name in ["position", "velocity", "effort"]:
            field_data = getattr(ros_msg, field_name, None)
            if field_data is not None and len(field_data) > 0:
                if field_name not in extracted_data[joint_key]:
                    extracted_data[joint_key][field_name] = []
                extracted_data[joint_key][field_name].append(list(field_data))

    def _extract_image(self, message, extracted_data: Dict):
        """Extract image from ROS Image message"""
        ros_msg = message.ros_msg
        time_s = message_timestamp(message)

        cam_name = self.config.camera_topic_mapping[message.channel.topic]
        extracted_data[cam_name]["timestamp"].append(time_s)

        # Decode image
        img_data = decode_image(ros_msg.data, ros_msg.encoding, ros_msg.height, ros_msg.width)
        extracted_data[cam_name]["image_data"].append(img_data)

        # Store metadata (once)
        if extracted_data[cam_name]["encoding"] is None:
            extracted_data[cam_name]["encoding"] = ros_msg.encoding
            extracted_data[cam_name]["height"] = ros_msg.height
            extracted_data[cam_name]["width"] = ros_msg.width

    def _extract_compressed_image(self, message, extracted_data: Dict):
        """Extract image from ROS CompressedImage message"""
        ros_msg = message.ros_msg
        time_s = message_timestamp(message)

        topic = message.channel.topic
        # Look up camera name from either the main mapping or the auto-generated compressed mapping
        if topic in self.config.camera_topic_mapping:
            cam_name = self.config.camera_topic_mapping[topic]
        else:
            cam_name = self._compressed_topic_mapping[topic]
        extracted_data[cam_name]["timestamp"].append(time_s)

        # Decode compressed image
        # CompressedImage format field contains the compression format (e.g., "jpeg", "png")
        img_data = decode_compressed_image(ros_msg.data, ros_msg.format)
        extracted_data[cam_name]["image_data"].append(img_data)

        # Store metadata (once) - get dimensions from decoded image
        if extracted_data[cam_name]["encoding"] is None:
            extracted_data[cam_name]["encoding"] = ros_msg.format
            extracted_data[cam_name]["height"] = img_data.shape[0]
            extracted_data[cam_name]["width"] = img_data.shape[1]

    def _is_joint_state_key(self, key: str) -> bool:
        """Check if a key is a joint state key."""
        return key.startswith("joint_states_")

    def _convert_to_arrays(self, extracted_data: Dict):
        """Convert lists to numpy arrays"""
        # Convert images
        for key, value in extracted_data.items():
            if key in self.config.camera_topic_mapping.values():
                if len(value["image_data"]) > 0:
                    value["image_data"] = np.array(value["image_data"], dtype=np.uint8)
                else:
                    value["image_data"] = np.empty((0,), dtype=np.uint8)

        # Convert timestamps (relative to first timestamp)
        # Find global first timestamp
        first_ts = None
        for key, value in extracted_data.items():
            if "timestamp" in value and len(value["timestamp"]) > 0:
                ts = value["timestamp"][0]
                if first_ts is None or ts < first_ts:
                    first_ts = ts

        if first_ts is None:
            first_ts = 0.0

        # Apply relative timestamps
        for key, value in extracted_data.items():
            if "timestamp" in value:
                ts_list = value["timestamp"]
                if len(ts_list) > 0:
                    ts = np.array(ts_list, dtype=np.float64)
                    ts = ts - first_ts
                    value["timestamp"] = ts.astype(np.float32)
                else:
                    value["timestamp"] = np.empty((0,), dtype=np.float32)

        # Convert joint states
        for key, value in extracted_data.items():
            if self._is_joint_state_key(key):
                for field_name, field_values in value.items():
                    if field_name in ["timestamp", "joint_names"]:
                        continue
                    if isinstance(field_values, list) and len(field_values) > 0:
                        value[field_name] = np.array(field_values, dtype=np.float32)
                    elif isinstance(field_values, list):
                        value[field_name] = np.empty((0,), dtype=np.float32)

    def _print_extraction_summary(self, extracted_data: Dict):
        """Print extraction summary"""
        # Print joint state summaries
        for key, value in extracted_data.items():
            if self._is_joint_state_key(key):
                count = len(value["timestamp"])
                joint_count = len(value.get("joint_names", []))
                print(f"[OK] Extracted {count} samples for {key} ({joint_count} joints)")

        # Print camera summaries
        for cam_name in self.config.camera_topic_mapping.values():
            if cam_name in extracted_data:
                count = len(extracted_data[cam_name]["image_data"])
                print(f"[OK] Extracted {count} images ({cam_name})")


# =============================================================================
# BufferedStreamExtractor - Streaming Mode (memory-bounded)
# =============================================================================


class BufferedStreamExtractor:
    """
    Memory-efficient streaming extractor for MCAP conversion.

    Uses a sliding window buffer for time alignment while keeping memory bounded.
    Extracts both images and joint states (full teleop data).

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
        extractor = BufferedStreamExtractor(config, buffer_seconds=5.0, fps=30)
        for frame in extractor.extract_frames("recording.mcap", task="teleop"):
            dataset.add_frame(frame)
    """

    def __init__(
        self,
        config: DataConfig,
        buffer_seconds: float = 5.0,
        fps: int = 30,
        quiet: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None,
        cli_act_from_obs: bool = False,
    ):
        """
        Initialize buffered stream extractor.

        Args:
            config: Data configuration specifying topics and camera mapping
            buffer_seconds: Total buffer window size in seconds (default: 5.0)
            fps: Frame rate for buffer size calculation (default: 30)
            quiet: If True, suppress all print output (default: False)
            progress_callback: Called with frames_yielded count after each frame
            cli_act_from_obs: When True, force ``action[t] = observation.state[t]``
                regardless of whether ``action_topics`` are configured (joint mode
                only; EE mode is always effectively act-from-obs).
        """
        self.config = config
        self.fps = fps
        self.frame_interval = 1.0 / fps  # seconds between output frames (for subsampling)
        self.buffer_seconds = buffer_seconds
        self.half_buffer = int(buffer_seconds * fps / 2)  # 75 frames for 5s @ 30fps
        self.full_buffer = self.half_buffer * 2  # 150 frames
        self.target_size = tuple(config.image_resolution)  # (width, height)
        self.quiet = quiet
        self.progress_callback = progress_callback
        self._cli_act_from_obs = cli_act_from_obs

        # Joint name pattern for parsing (reuse from DataExtractor)
        self._joint_pattern = config.joint_name_pattern

        # Cache for action topic reorder permutations: {topic: np.ndarray}
        self._action_reorder_cache: Dict[str, np.ndarray] = {}

        # Last known action position per robot ("" for single-arm, "left"/"right"
        # for bimanual) — used to forward-fill action frames when an arm is
        # disengaged (no new command published). Updated in
        # _buffer_action_command() and is NOT evicted by the sliding buffer
        # window, so a command published long ago can still be reused.
        self._last_known_action: Dict[str, np.ndarray] = {}

        # Per-robot, per-episode gap-fill counters, keyed by robot then by
        # one of "exact" | "hold_last" | "fallback_to_observation" | "dropped".
        # Reset implicitly per episode because BufferedStreamExtractor is
        # constructed fresh for each MCAP file (see convert.py).
        self._action_fill_stats: Dict[str, Dict[str, int]] = {}

    @property
    def act_from_obs(self) -> bool:
        """Whether ``action[t] = observation.state[t]`` (no command-topic source).

        True when the CLI flag forces it or when action_topics is empty. In EE
        mode this is structurally true because action_topics is always empty,
        but the EE align still encodes action with rot6d (not a literal copy
        of the quaternion-encoded state).
        """
        return self._cli_act_from_obs or not bool(self.config.action_command_topics)

    @staticmethod
    def _check_action_topics_present(mcap_path: str, action_topic_set: set) -> None:
        """Raise DataExtractionError when none of the configured action topics are in the MCAP.

        Uses the MCAP footer index (O(1)) — does not scan all messages.
        """
        from mcap.reader import make_reader as _make_mcap_reader

        try:
            with open(mcap_path, "rb") as _f:
                _summary = _make_mcap_reader(_f).get_summary()
        except Exception:
            return  # Cannot read summary — let streaming fail naturally

        if _summary is None:
            return

        available = {ch.topic for ch in _summary.channels.values()}
        if action_topic_set and action_topic_set.isdisjoint(available):
            raise DataExtractionError(
                f"\n[ACTION SOURCE ERROR] action_from_observation=false, but none of the "
                f"configured action topics were found in this MCAP.\n"
                f"  Expected (from config): {sorted(action_topic_set)}\n"
                f"  Available in MCAP:      {sorted(available)}\n\n"
                "Fix options:\n"
                "  1. Set action_from_observation: true in config to derive actions from "
                "observations.\n"
                "  2. Record a new session that captures the action-command topics.\n"
                "  3. Use a different --config that matches this recording's topic layout.\n"
            )

    def extract_frames(
        self,
        mcap_path: str,
        task: str = "teleop",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator yielding time-aligned frames with bounded memory.

        Args:
            mcap_path: Path to MCAP file
            task: Task name for each frame (default: "teleop")

        Yields:
            Dictionary with aligned frame data:
            {
                'observation.images.{camera}': np.ndarray (H, W, C),
                'observation.state': np.ndarray,        # or '{robot}.observation.state'
                'observation.velocity': np.ndarray,     # optional
                'observation.effort': np.ndarray,       # optional
                'action': np.ndarray,                   # or '{robot}.action'
                'task': str,
            }
        """
        from ..utils.image_utils import resize_image

        if not self.quiet:
            print(f"[BufferedStream] Reading MCAP: {mcap_path}")
            print(f"[BufferedStream] Buffer: {self.full_buffer} frames ({self.half_buffer}x2)")

        reader = McapReader(mcap_path)

        # Build camera topic lists
        camera_topics = list(self.config.camera_topics)
        compressed_topics = [t + "/compressed" for t in camera_topics]
        all_camera_topics = camera_topics + compressed_topics

        # Mapping from topic to camera name
        topic_to_cam = dict(self.config.camera_topic_mapping)
        for t in camera_topics:
            topic_to_cam[t + "/compressed"] = self.config.camera_topic_mapping[t]

        # Per-camera buffers: {cam_name: deque of (timestamp, image)}
        camera_buffers: Dict[str, deque] = {
            cam: deque() for cam in self.config.camera_topic_mapping.values()
        }

        # Per-mode signal buffers (only one of these is populated per run).
        # Joint mode: {(role, robot): {'buffer': deque, 'joint_names': list}}
        joint_buffers = self._init_joint_buffers()
        # EE mode: {arm_id: deque of (ts, pos_xyz np.ndarray, quat_xyzw np.ndarray, gripper float)}
        ee_buffers: Dict[str, deque] = {}
        # Reverse map topic -> [arm_id, ...] for EE dispatch (a topic may serve
        # multiple arms in pathological configs; the loop fans out the message
        # to every arm that listed this topic).
        ee_topic_to_arms: Dict[str, List[str]] = {}

        # Build topic list for reading (cameras + joint or ee signals)
        all_topics = list(all_camera_topics)
        action_topic_set: set = set()

        if self.config.is_ee:
            for arm_id, topic in self.config.observation_topics.items():
                ee_topic_to_arms.setdefault(topic, []).append(arm_id)
                ee_buffers[arm_id] = deque()
            for t in ee_topic_to_arms:
                if t not in all_topics:
                    all_topics.append(t)
        else:
            # Joint mode: shared /joint_states topic for all arms.
            rst = self.config.robot_state_topic
            if rst:
                all_topics.append(rst)
            if self.act_from_obs:
                if not self.quiet:
                    if self._cli_act_from_obs and self.config.action_command_topics:
                        print(
                            "\n[ACTION SOURCE] --act-from-obs override: "
                            "action[t] = observation.state[t]; command topics ignored.\n"
                        )
                    else:
                        print(
                            "\n[ACTION SOURCE] action_topics empty: "
                            "action[t] = observation.state[t]\n"
                        )
            else:
                action_topic_set = set(self.config.action_command_topics.keys())
                all_topics.extend(action_topic_set)
                # Fast-fail: verify at least one action topic is recorded in this MCAP.
                # Reads only the file footer index (O(1)) — no message scanning.
                self._check_action_topics_present(mcap_path, action_topic_set)

        # Get main camera (first one in config)
        main_cam = list(self.config.camera_topic_mapping.values())[0]

        cursor = 0  # Index of frame to process next
        frames_yielded = 0
        next_yield_ts = None  # Next target timestamp for subsampling
        # EE delta-encoding only: the immediately-preceding frame's absolute, ALWAYS
        # quaternion-encoded state (never observation.state as actually written to disk,
        # which may be rot6d/axis_angle — see _align_ee_signals), carried forward as the
        # anchor for the NEXT frame's baked delta. None on the first frame of the episode
        # (self-anchor — see _align_ee_signals). Reset per extract_frames call, i.e. per
        # episode.
        prev_ee_state_quat: Optional[np.ndarray] = None

        for message in reader.read_messages(topics=all_topics):
            topic = message.channel.topic

            # EE mode: one topic feeds one or more arms.
            if self.config.is_ee:
                arms_for_topic = ee_topic_to_arms.get(topic)
                if arms_for_topic is not None:
                    self._buffer_ee_pose(message, arms_for_topic, ee_buffers)
                    continue
            else:
                # Joint state messages
                if topic == self.config.robot_state_topic:
                    self._buffer_joint_state(message, joint_buffers)
                    continue
                # Joint action command messages
                if topic in action_topic_set:
                    self._buffer_action_command(message, topic, joint_buffers)
                    continue

            # Handle camera messages
            if topic not in topic_to_cam:
                continue
            cam_name = topic_to_cam[topic]

            time_s = message_timestamp(message)

            # Decode image — detect CompressedImage vs Image by attribute
            ros_msg = message.ros_msg
            if hasattr(ros_msg, "format"):
                img = decode_compressed_image(ros_msg.data, ros_msg.format)
            else:
                img = decode_image(ros_msg.data, ros_msg.encoding, ros_msg.height, ros_msg.width)

            # Add to buffer
            camera_buffers[cam_name].append((time_s, img))

            # Start processing when main camera buffer reaches threshold
            main_buffer_len = len(camera_buffers[main_cam])

            # Condition: buffer has at least half_buffer frames ahead of cursor
            if main_buffer_len >= self.half_buffer + cursor:
                frame_ts = camera_buffers[main_cam][cursor][0]

                # Initialize subsampling anchor on first frame
                if next_yield_ts is None:
                    next_yield_ts = frame_ts

                # Subsampling: only yield if this frame is at or past the next target timestamp
                if frame_ts >= next_yield_ts:
                    frame, state_quat = self._align_frame_at_cursor(
                        camera_buffers, joint_buffers, ee_buffers,
                        cursor, main_cam, task, resize_image,
                        prev_ee_state_quat=prev_ee_state_quat,
                    )
                    if frame is not None:
                        yield frame
                        frames_yielded += 1
                        next_yield_ts += self.frame_interval
                        if self.config.is_ee:
                            prev_ee_state_quat = state_quat

                cursor += 1

                # Always report progress after each cursor advance (including skipped frames)
                if self.progress_callback:
                    self.progress_callback(frames_yielded)
                elif not self.quiet and frames_yielded % 100 == 0:
                    print(f"[BufferedStream] Processed {frames_yielded} frames...")

                # Once buffer reaches full size, remove oldest to maintain size
                if main_buffer_len > self.full_buffer:
                    for buffer in camera_buffers.values():
                        if len(buffer) > 0:
                            buffer.popleft()

                    # Sync signal buffers to remove old samples
                    if camera_buffers[main_cam]:
                        new_oldest_ts = camera_buffers[main_cam][0][0]
                        if self.config.is_ee:
                            self._sync_ee_buffers(ee_buffers, new_oldest_ts)
                        else:
                            self._sync_joint_buffers(joint_buffers, new_oldest_ts)

                    cursor -= 1  # Adjust cursor after removal

        # Flush: process remaining frames in buffer
        if not self.quiet:
            print("[BufferedStream] Flushing remaining buffer...")
        while cursor < len(camera_buffers[main_cam]):
            frame_ts = camera_buffers[main_cam][cursor][0]
            if next_yield_ts is None or frame_ts >= next_yield_ts:
                frame, state_quat = self._align_frame_at_cursor(
                    camera_buffers, joint_buffers, ee_buffers,
                    cursor, main_cam, task, resize_image,
                    prev_ee_state_quat=prev_ee_state_quat,
                )
                if frame is not None:
                    yield frame
                    frames_yielded += 1
                    if self.config.is_ee:
                        prev_ee_state_quat = state_quat
                    if next_yield_ts is not None:
                        next_yield_ts += self.frame_interval
                    if self.progress_callback:
                        self.progress_callback(frames_yielded)
            cursor += 1

        if not self.quiet:
            print(f"[BufferedStream] [OK] Extracted {frames_yielded} frames total")

        if frames_yielded == 0:
            # Diagnostic: show why no frames were produced
            cam_counts = {cam: len(buf) for cam, buf in camera_buffers.items()}
            print(f"[BufferedStream] WARNING: 0 frames produced — diagnostics:")
            print(f"  Camera buffers: {cam_counts}")
            if not cam_counts or all(c == 0 for c in cam_counts.values()):
                print(f"  -> No camera images found. Check that these topics exist in the MCAP:")
                for t in self.config.camera_topics:
                    print(f"       {t}")

            if self.config.is_ee:
                ee_counts = {arm: len(buf) for arm, buf in ee_buffers.items()}
                print(f"  EE buffers:    {ee_counts if ee_counts else '(empty — no /ee_pose data received)'}")
                if not ee_counts or all(c == 0 for c in ee_counts.values()):
                    print(f"  -> No EE pose data. Check observation_topics:")
                    for arm_id, topic in self.config.observation_topics.items():
                        print(f"       {arm_id} -> {topic}")
            else:
                joint_keys = {
                    f"{role}:{robot or 'default'}": len(d["buffer"])
                    for (role, robot), d in joint_buffers.items()
                }
                print(f"  Joint buffers: {joint_keys if joint_keys else '(empty — no joint data received)'}")
                if not joint_keys:
                    print(f"  -> No joint state data. Check robot_state_topic: {self.config.robot_state_topic}")
                    cmd_topics = self.config.action_command_topics
                    if cmd_topics:
                        print(f"  -> No action data. Check action command topics: {list(cmd_topics.keys())}")
                elif not any(k.startswith("action:") for k in joint_keys) and not self.act_from_obs:
                    cmd_topics = self.config.action_command_topics
                    if cmd_topics:
                        print(f"  -> No action data received from action command topics:")
                        for t in cmd_topics:
                            print(f"       {t}")
                    else:
                        print(f"  -> No action data parsed from joint_states (no leader prefix matched).")

    def _align_frame_at_cursor(
        self,
        camera_buffers: Dict[str, deque],
        joint_buffers: Dict[Tuple[str, str], Dict],
        ee_buffers: Dict[str, deque],
        cursor: int,
        main_cam: str,
        task: str,
        resize_func,
        prev_ee_state_quat: Optional[Any] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Align frame at cursor position using entire buffer for matching.

        Args:
            camera_buffers: Per-camera buffers
            joint_buffers: Joint-mode signal buffers (keyed by (role, robot))
            ee_buffers: EE-mode signal buffers (keyed by arm_id)
            cursor: Index of frame to align in main camera buffer
            main_cam: Name of main camera
            task: Task name
            resize_func: Function to resize images
            prev_ee_state_quat: EE delta-encoding only — the immediately-preceding
                frame's absolute, quaternion-encoded state, forwarded to
                ``_align_ee_signals`` as the delta anchor. ``None`` on an
                episode's first frame (self-anchor). Unused outside EE mode.

        Returns:
            ``(frame, state_quat)`` — ``frame`` is the aligned frame dictionary, or
            ``None`` if not all data available. ``state_quat`` is the quaternion-encoded
            EE state for this frame (EE mode only, else ``None``) — not part of the frame
            dict itself, since it's not a dataset feature; callers thread it back in as
            the next call's ``prev_ee_state_quat``.
        """
        # Get main camera frame at cursor
        main_ts, main_img = camera_buffers[main_cam][cursor]

        # Resize main camera image
        resized_main = resize_func(main_img, self.target_size)
        frame = {f"observation.images.{main_cam}": resized_main}

        # Find nearest match for each other camera
        for cam_name, buffer in camera_buffers.items():
            if cam_name == main_cam:
                continue

            # If any camera has no data, skip this frame
            if len(buffer) == 0:
                return None, None

            # Search entire buffer for nearest timestamp match
            nearest_idx = self._find_nearest_in_buffer(buffer, main_ts)
            if nearest_idx is not None:
                _, img = buffer[nearest_idx]
                resized_img = resize_func(img, self.target_size)
                frame[f"observation.images.{cam_name}"] = resized_img
            else:
                return None, None

        # Align non-camera signals — mode-specific
        state_quat: Optional[np.ndarray] = None
        if self.config.is_ee:
            result = self._align_ee_signals(
                ee_buffers, main_ts, prev_state_quat=prev_ee_state_quat
            )
            if result is None:
                return None, None
            signals, state_quat = result
        elif joint_buffers:
            signals = self._align_joint_states(
                joint_buffers, main_ts, act_from_obs=self.act_from_obs
            )
        else:
            signals = None

        if signals is None:
            return None, None
        frame.update(signals)

        frame["task"] = task
        return frame, state_quat

    def _align_joint_states(
        self,
        joint_buffers: Dict[Tuple[str, str], Dict],
        target_ts: float,
        act_from_obs: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Align joint states to target timestamp.

        For multi-robot setups (left/right), concatenates features into single arrays:
        - observation.state = [left_joints..., right_joints...]
        - action = [left_joints..., right_joints...]

        Args:
            joint_buffers: Joint state buffers keyed by (role, robot)
            target_ts: Target timestamp for observation alignment
            act_from_obs: When True, ``action[t] = observation.state[t]`` per arm
                — used when ``action_topics`` is empty or ``--act-from-obs`` is
                set. The future window is applied at train time by
                ``delta_timestamps``.

        Returns:
            Dictionary with aligned joint state features, or None if data missing
        """
        # Collect data by role, then concatenate robots in sorted order
        obs_data = {}  # {robot: {pos, vel, eff}}
        action_data = {}  # {robot: {pos}}

        # Pass 1: observation — required, no fallback (unchanged behavior).
        # Action fallback (pass 2 below) needs obs_data fully populated first,
        # which is why this is a separate pass rather than one combined loop.
        for (role, robot), data in joint_buffers.items():
            if role != "observation":
                continue
            buffer = data["buffer"]

            if len(buffer) == 0:
                return None  # Required observation data missing

            nearest_idx = self._find_nearest_in_buffer(buffer, target_ts)
            if nearest_idx is None:
                return None

            ts, pos, vel, eff = buffer[nearest_idx]
            obs_data[robot] = {
                "pos": pos.copy(),
                "vel": vel.copy() if vel.size > 0 else None,
                "eff": eff.copy() if eff.size > 0 else None,
            }

        # Pass 2: action — forward-fill instead of dropping the whole frame
        # when the arm is disengaged. See _resolve_action_position docstring
        # for the fallback order.
        for (role, robot), data in joint_buffers.items():
            if role != "action":
                continue
            pos, fill_kind = self._resolve_action_position(
                robot, data["buffer"], target_ts, obs_data
            )
            self._record_action_fill(robot, fill_kind)
            if pos is None:
                return None  # No action and no observation fallback available
            action_data[robot] = {"pos": pos}

        # Unified act-from-obs rule: when no action source is configured (or
        # the CLI forces it), set action[t] = observation.state[t] per arm.
        # The future window is later applied by LeRobot's delta_timestamps at
        # training time.
        if act_from_obs and obs_data:
            action_data = {robot: {"pos": v["pos"].copy()} for robot, v in obs_data.items()}

        # Check if multi-robot (has named robots like 'left', 'right')
        robots = sorted([r for r in set(obs_data.keys()) | set(action_data.keys()) if r])

        if robots:
            # Multi-robot: require ALL robots to have both observation and action data
            # to ensure consistent output shape (e.g., 16 = 8 left + 8 right)
            for r in robots:
                if r not in obs_data:
                    return None  # Observation data not yet available for this arm
                if r not in action_data:
                    return None  # Action data not yet available for this arm

            # Multi-robot: concatenate in sorted order (left, right)
            result = {}

            # Concatenate observation state
            obs_positions = [obs_data[r]["pos"] for r in robots]
            if obs_positions:
                result["observation.state"] = np.concatenate(obs_positions)

            # Concatenate observation velocity (only if enabled in config)
            if "velocity" in self.config.observation_feature_mapping.others:
                obs_velocities = [
                    obs_data[r]["vel"] for r in robots if obs_data[r]["vel"] is not None
                ]
                if obs_velocities:
                    result["observation.velocity"] = np.concatenate(obs_velocities)

            # Concatenate observation effort (only if enabled in config)
            if "effort" in self.config.observation_feature_mapping.others:
                obs_efforts = [obs_data[r]["eff"] for r in robots if obs_data[r]["eff"] is not None]
                if obs_efforts:
                    result["observation.effort"] = np.concatenate(obs_efforts)

            # Concatenate action
            action_positions = [action_data[r]["pos"] for r in robots]
            if action_positions:
                result["action"] = np.concatenate(action_positions)

            return result
        else:
            # Single robot: use original naming (no prefix)
            result = {}

            if "" in obs_data:
                result["observation.state"] = obs_data[""]["pos"]
                if "velocity" in self.config.observation_feature_mapping.others:
                    if obs_data[""]["vel"] is not None:
                        result["observation.velocity"] = obs_data[""]["vel"]
                if "effort" in self.config.observation_feature_mapping.others:
                    if obs_data[""]["eff"] is not None:
                        result["observation.effort"] = obs_data[""]["eff"]

            if "" in action_data:
                result["action"] = action_data[""]["pos"]

            return result

    def _find_nearest_in_buffer(
        self,
        buffer: deque,
        target_ts: float,
    ) -> Optional[int]:
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

    def _resolve_action_position(
        self,
        robot: str,
        buffer: deque,
        target_ts: float,
        obs_data: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Resolve the action position for one robot at target_ts, forward-filling
        when the arm is disengaged (no live command in the sliding buffer).

        Fallback order:
        1. Nearest command in the live buffer ("exact").
        2. Last command ever published for this robot, regardless of how long
           ago ("hold_last") — the arm physically holds its last commanded
           position when idle, so this reflects reality.
        3. The robot's current measured joint position from obs_data
           ("fallback_to_observation") — used when the robot has never
           published a command yet in this episode (e.g. an idle arm at the
           very start of a recording).
        4. None ("dropped") — should not happen in practice since
           observation data is dense, but kept as a safety net.

        Returns:
            (position, fill_kind) where position is a copy of the resolved
            array, or None if fill_kind == "dropped".
        """
        if buffer:
            nearest_idx = self._find_nearest_in_buffer(buffer, target_ts)
            _, pos, _, _ = buffer[nearest_idx]
            return pos.copy(), "exact"

        last_known = self._last_known_action.get(robot)
        if last_known is not None:
            return last_known.copy(), "hold_last"

        if robot in obs_data:
            return obs_data[robot]["pos"].copy(), "fallback_to_observation"

        return None, "dropped"

    def _record_action_fill(self, robot: str, fill_kind: str) -> None:
        """Increment the per-robot, per-episode gap-fill counter."""
        stats = self._action_fill_stats.setdefault(
            robot, {"exact": 0, "hold_last": 0, "fallback_to_observation": 0, "dropped": 0}
        )
        stats[fill_kind] += 1

    def get_action_fill_stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-robot action gap-fill counters accumulated this episode."""
        return self._action_fill_stats

    def _parse_joint_name(self, joint_name: str) -> Optional[Tuple[str, str, str]]:
        """Parse joint name using shared utility."""
        return parse_joint_name(joint_name, self._joint_pattern)

    def _init_joint_buffers(self) -> Dict[Tuple[str, str], Dict]:
        """
        Create the initial joint_buffers dict for a new extraction run.

        Pre-seeds an empty ("action", robot) entry for every robot configured
        in action_topics, so _align_joint_states's action pass always visits
        every configured robot from the first frame onward — even before that
        robot has published its first command. Without this, a robot with no
        live command yet would have no key in joint_buffers at all, and
        _resolve_action_position's "fallback_to_observation" tier would never
        be reached: the frame would instead be silently dropped by the
        multi-robot consistency check further down in _align_joint_states.

        Observation keys are NOT pre-seeded here — they're always created
        densely and immediately by _buffer_joint_state from the continuous
        /joint_states stream, so there's no equivalent gap for them.
        """
        joint_buffers: Dict[Tuple[str, str], Dict] = {}
        for topic_cfg in self.config.action_command_topics.values():
            key = ("action", topic_cfg.arm)
            joint_names = sorted(topic_cfg.joint_order) if topic_cfg.joint_order else []
            joint_buffers[key] = {"buffer": deque(), "joint_names": joint_names}
        return joint_buffers

    def _buffer_joint_state(
        self,
        message,
        joint_buffers: Dict[Tuple[str, str], Dict],
    ) -> None:
        """
        Parse joint state message and add to appropriate buffers.

        Joints are sorted by joint_id for deterministic canonical ordering.

        Args:
            message: MCAP message with ROS JointState
            joint_buffers: Dict keyed by (role, robot) containing buffer and metadata
        """
        ros_msg = message.ros_msg
        timestamp = message_timestamp(message)

        # Group joints by (role, robot)
        grouped: Dict[Tuple[str, str], Dict] = {}

        for i, joint_name in enumerate(ros_msg.name):
            try:
                result = self._parse_joint_name(joint_name)
            except DataExtractionError:
                continue  # Skip unparseable joints
            if result is None:
                continue  # arm not in configured arms map, skip
            role, robot, joint_id = result

            key = (role, robot)
            if key not in grouped:
                grouped[key] = {
                    "joint_ids": [],
                    "position": [],
                    "velocity": [],
                    "effort": [],
                }

            grouped[key]["joint_ids"].append(joint_id)

            if ros_msg.position and i < len(ros_msg.position):
                grouped[key]["position"].append(ros_msg.position[i])
            if ros_msg.velocity and i < len(ros_msg.velocity):
                grouped[key]["velocity"].append(ros_msg.velocity[i])
            if ros_msg.effort and i < len(ros_msg.effort):
                grouped[key]["effort"].append(ros_msg.effort[i])

        # Sort each group by joint_id for canonical ordering
        for key, data in grouped.items():
            sort_indices = sorted(
                range(len(data["joint_ids"])),
                key=lambda idx: data["joint_ids"][idx],
            )
            data["joint_ids"] = [data["joint_ids"][idx] for idx in sort_indices]
            data["position"] = [data["position"][idx] for idx in sort_indices]
            data["velocity"] = [data["velocity"][idx] for idx in sort_indices]
            data["effort"] = [data["effort"][idx] for idx in sort_indices]

        # Add to buffers
        for key, data in grouped.items():
            if key not in joint_buffers:
                joint_buffers[key] = {
                    "buffer": deque(),
                    "joint_names": data["joint_ids"],  # Store sorted joint names once
                }

            # Create arrays
            pos = (
                np.array(data["position"], dtype=np.float32)
                if data["position"]
                else np.array([], dtype=np.float32)
            )
            vel = (
                np.array(data["velocity"], dtype=np.float32)
                if data["velocity"]
                else np.array([], dtype=np.float32)
            )
            eff = (
                np.array(data["effort"], dtype=np.float32)
                if data["effort"]
                else np.array([], dtype=np.float32)
            )

            # Append as tuple: (timestamp, position, velocity, effort)
            joint_buffers[key]["buffer"].append((timestamp, pos, vel, eff))

    def _buffer_action_command(
        self,
        message,
        topic: str,
        joint_buffers: Dict[Tuple[str, str], Dict],
    ) -> None:
        """
        Parse action command message (Float64MultiArray) and add to joint buffers.

        Used in quest teleop mode where actions come from separate command topics
        instead of from leader joints in the JointState topic.
        Positions are reordered to canonical (sorted) joint order using joint_order
        from the ActionTopicConfig.

        Args:
            message: MCAP message with Float64MultiArray
            topic: The ROS topic name
            joint_buffers: Dict keyed by (role, robot) containing buffer and metadata
        """
        ros_msg = message.ros_msg
        # Float64MultiArray has no header — message_timestamp falls back to log_time.
        timestamp = message_timestamp(message)

        # Get action topic config
        topic_cfg = self.config.action_command_topics[topic]
        robot = topic_cfg.arm
        key = ("action", robot)

        # Extract position data from Float64MultiArray.data
        positions = list(ros_msg.data)

        # Compute reorder permutation on first message (cached per topic)
        if topic not in self._action_reorder_cache:
            if topic_cfg.joint_order:
                # Sort joint_order alphabetically to match canonical observation order
                self._action_reorder_cache[topic] = np.array(
                    sorted(
                        range(len(topic_cfg.joint_order)),
                        key=lambda idx: topic_cfg.joint_order[idx],
                    ),
                    dtype=np.intp,
                )
            else:
                # No joint_order specified — identity permutation (no reorder)
                self._action_reorder_cache[topic] = np.arange(len(positions), dtype=np.intp)

        reorder = self._action_reorder_cache[topic]

        if key not in joint_buffers:
            # Use canonical (sorted) joint names
            if topic_cfg.joint_order:
                joint_names = sorted(topic_cfg.joint_order)
            else:
                joint_names = [f"joint{i}" for i in range(len(positions))]
            joint_buffers[key] = {
                "buffer": deque(),
                "joint_names": joint_names,
            }

        # Reorder positions to canonical order
        pos = np.array(positions, dtype=np.float32)[reorder]
        vel = np.array([], dtype=np.float32)
        eff = np.array([], dtype=np.float32)

        self._last_known_action[robot] = pos.copy()

        # Append as tuple: (timestamp, position, velocity, effort)
        joint_buffers[key]["buffer"].append((timestamp, pos, vel, eff))

    def _sync_joint_buffers(
        self,
        joint_buffers: Dict[Tuple[str, str], Dict],
        oldest_camera_ts: float,
    ) -> None:
        """
        Remove joint state samples older than the oldest camera frame.

        This keeps joint state buffers synchronized with camera buffer time window.

        Args:
            joint_buffers: Joint state buffers to sync
            oldest_camera_ts: Timestamp of oldest remaining camera frame
        """
        for key, data in joint_buffers.items():
            buffer = data["buffer"]
            while buffer and buffer[0][0] < oldest_camera_ts:
                buffer.popleft()

    def _get_joint_names(
        self,
        joint_buffers: Dict[Tuple[str, str], Dict],
    ) -> Dict[str, List[str]]:
        """
        Extract joint names from buffers for dataset feature creation.

        Returns:
            Dict mapping robot prefix to joint names (e.g., {"right": ["joint1", ...], "left": [...]})
        """
        joint_names = {}
        for (role, robot), data in joint_buffers.items():
            if role == "observation":  # Use observation to get joint names (same for action)
                prefix = robot if robot else ""
                if prefix not in joint_names:
                    joint_names[prefix] = data["joint_names"]
        return joint_names

    # ------------------------------------------------------------------
    # EE-mode helpers
    # ------------------------------------------------------------------

    def _buffer_ee_pose(
        self,
        message,
        arms_for_topic: List[str],
        ee_buffers: Dict[str, deque],
    ) -> None:
        """Decode a CommandedEEPose message and append to every arm it serves.

        We buffer the raw ``(pos, quat, gripper)`` tuple once and defer encoding
        (state vs action) to align time, so the same decode is reused for both
        the 8-dim quaternion state and the 10-dim rot6d action.
        """
        ros_msg = message.ros_msg
        timestamp = message.log_time.timestamp()

        pos = np.array(
            [
                ros_msg.pose.position.x,
                ros_msg.pose.position.y,
                ros_msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        quat = np.array(
            [
                ros_msg.pose.orientation.x,
                ros_msg.pose.orientation.y,
                ros_msg.pose.orientation.z,
                ros_msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not hasattr(ros_msg, "gripper"):
            log.warning(
                "[EE] Message on topic '%s' has no 'gripper' field — defaulting to 0.0. "
                "Check that the topic publishes anvil_msgs/msg/CommandedEEPose.",
                arms_for_topic[0] if arms_for_topic else "?",
            )
        gripper = float(getattr(ros_msg, "gripper", 0.0))

        sample = (timestamp, pos, quat, gripper)
        for arm_id in arms_for_topic:
            ee_buffers[arm_id].append(sample)

    def _align_ee_signals(
        self,
        ee_buffers: Dict[str, deque],
        target_ts: float,
        prev_state_quat: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
        """Align per-arm EE pose buffers to the camera frame at ``target_ts``.

        Builds the unified canonical features by concatenating per-arm slices in
        the insertion order of ``config.observation_topics``::

            observation.state  = concat([xyz, <rotation encoding>, gripper] per arm)
            action             = concat([xyz, rot6d, gripper]   per arm)     (10 * n_arms,)

        ``observation.state``'s rotation component is written in
        ``config.observation_encoding`` (quaternion/rot6d/axis_angle — see
        ``config/encodings.py`` for the per-encoding layout table). ``action`` is always
        rot6d regardless — an independent knob from ``observation_encoding``.

        ``action`` is the absolute EE pose (re-encoded to rot6d) by default
        (``config.action_encoding == "absolute"``). When ``config.is_action_delta``,
        ``action`` instead becomes a baked per-frame Delta(n-(n-1)) target:
        ``ee_delta_forward(action_abs, anchor)``, where ``anchor`` is
        ``prev_state_quat`` — the immediately-preceding frame's absolute,
        QUATERNION-encoded state (never the ``observation_encoding``-selected value
        actually written to disk: ``ee_delta_forward``/``n_arms_from_dims`` hardcode an
        8-dim-per-arm quaternion state layout, so the delta anchor must stay quaternion
        regardless of what ``observation.state`` is encoded as on disk) — or, on an
        episode's first frame (``prev_state_quat is None``), the CURRENT frame's own
        quaternion state (self-anchor, yielding a zero delta; matches the identity-delta
        invariant asserted by ``test_identity_state_identity_action_zero_delta`` and its
        ``ee_delta_forward`` analogue — no LIBERO precedent exists for this boundary
        case, since LIBERO's ``native`` never differences two states).

        Returns ``(frame, state_quat)`` where ``state_quat`` is ALWAYS quaternion-encoded
        (regardless of ``observation_encoding``) — it is NOT part of the dataset schema;
        callers thread it back in as the next call's ``prev_state_quat`` rather than
        smuggling it into ``frame`` and stripping it out later, so nothing downstream can
        accidentally treat it as a real feature.

        Returns ``None`` if any arm has no buffered pose yet.
        """
        # Lazy imports keep the converter independent of training packages except when
        # EE mode (and, for non-quaternion observation_encoding or delta action_encoding,
        # anvil_shared's transform/rotation modules) is actually used.
        from anvil_shared.rotation import matrix_to_rot6d, quat_to_matrix

        from ..config.encodings import encode_rotation

        state_quat_slices: List[np.ndarray] = []
        state_encoded_slices: List[np.ndarray] = []
        action_slices: List[np.ndarray] = []
        for arm_id in self.config.observation_topics:  # insertion order
            buffer = ee_buffers.get(arm_id)
            if not buffer:
                return None
            idx = self._find_nearest_in_buffer(buffer, target_ts)
            if idx is None:
                return None
            _, pos, quat, gripper = buffer[idx]
            gripper_arr = np.array([gripper], dtype=np.float64)

            state_quat_slices.append(np.concatenate([pos, quat, gripper_arr]))

            rot_encoded = encode_rotation(quat, self.config.observation_encoding)
            state_encoded_slices.append(np.concatenate([pos, rot_encoded, gripper_arr]))

            rot6d = matrix_to_rot6d(quat_to_matrix(quat))  # action stays rot6d always
            action_slices.append(np.concatenate([pos, rot6d, gripper_arr]))

        state_quat = np.concatenate(state_quat_slices)
        state_encoded = np.concatenate(state_encoded_slices)
        action_abs = np.concatenate(action_slices)

        if self.config.is_action_delta:
            # Lazy import: keeps the converter independent of training packages
            # except when this specific encoding is actually requested.
            from anvil_shared.ee_transform import ee_delta_forward

            anchor = state_quat if prev_state_quat is None else prev_state_quat
            action_out = ee_delta_forward(action_abs, anchor)
        else:
            action_out = action_abs

        frame = {
            "observation.state": state_encoded.astype(np.float32),
            "action": action_out.astype(np.float32),
        }
        return frame, state_quat

    def _sync_ee_buffers(
        self,
        ee_buffers: Dict[str, deque],
        oldest_ts: float,
    ) -> None:
        """Drop EE samples older than the oldest camera frame to bound memory."""
        for buffer in ee_buffers.values():
            while buffer and buffer[0][0] < oldest_ts:
                buffer.popleft()
