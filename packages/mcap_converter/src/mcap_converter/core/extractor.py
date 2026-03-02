"""Data extraction from MCAP files"""

from collections import deque
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from ..config.schema import DEFAULT_DATA_CONFIG, DataConfig, JointNamePattern
from ..exceptions import DataExtractionError
from ..utils.image_utils import decode_compressed_image, decode_image
from .reader import McapReader

# =============================================================================
# Shared Utilities
# =============================================================================


def parse_joint_name(
    joint_name: str,
    pattern: JointNamePattern,
) -> Tuple[str, str, str]:
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
    else:
        joint_id = remaining

    return role, robot, joint_id


# =============================================================================
# DataExtractor - Batch Mode (loads entire episode into memory)
# =============================================================================


class DataExtractor:
    """
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

    def _parse_joint_name(self, joint_name: str) -> Tuple[str, str, str]:
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

        # Include action command topics if configured (quest teleop mode)
        if self.config.action_topics:
            interested_topics.extend(self.config.action_topics.keys())

        # Also include legacy topics if configured
        if self.config.robot_state_topics:
            interested_topics.extend(self.config.robot_state_topics)

        for message in reader.read_messages(topics=interested_topics):
            topic = message.channel.topic

            # Handle single robot state topic (new architecture)
            if topic == self.config.robot_state_topic:
                self._extract_joint_state_single_topic(message, extracted_data)
            # Handle action command topics (quest teleop mode)
            elif self.config.action_topics and topic in self.config.action_topics:
                self._extract_action_command(message, topic, extracted_data)
            # Handle legacy multi-topic architecture
            elif self.config.robot_state_topics and topic in self.config.robot_state_topics:
                self._extract_joint_state_legacy(message, extracted_data)
            # Handle camera topics (raw images)
            elif topic in self.config.camera_topics:
                self._extract_image(message, extracted_data)
            # Handle compressed image topics
            elif topic in self._compressed_topic_mapping:
                self._extract_compressed_image(message, extracted_data)

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
        """
        ros_msg = message.ros_msg
        time_ns = (ros_msg.header.stamp.sec * 1e9 + ros_msg.header.stamp.nanosec) / 1e9

        # Group joints by (role, robot)
        grouped_data: Dict[Tuple[str, str], Dict] = {}

        for i, joint_name in enumerate(ros_msg.name):
            try:
                role, robot, joint_id = self._parse_joint_name(joint_name)
            except DataExtractionError as e:
                print(f"Warning: {e}")
                continue

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

            extracted_data[key]["timestamp"].append(time_ns)
            extracted_data[key]["position"].append(data["position"])
            extracted_data[key]["velocity"].append(data["velocity"])
            extracted_data[key]["effort"].append(data["effort"])

    def _extract_action_command(self, message, topic: str, extracted_data: Dict):
        """
        Extract action from a command topic (quest teleop mode).

        Command topics publish std_msgs/Float64MultiArray with joint positions.
        These messages have no header, so we use MCAP log_time for timestamps.

        Args:
            message: MCAP message (Float64MultiArray)
            topic: The ROS topic name
            extracted_data: Data dictionary to populate
        """
        ros_msg = message.ros_msg
        # Float64MultiArray has no header; use MCAP recording timestamp
        time_s = message.log_time / 1e9

        # Map topic to arm identifier (e.g., "left" or "right")
        robot = self.config.action_topics[topic]
        key = self._get_joint_state_key("action", robot)

        # Extract position data from Float64MultiArray.data
        positions = list(ros_msg.data)

        if key not in extracted_data:
            # Generate joint names as positional indices (j0, j1, ...)
            # These will be replaced by observation joint names during alignment
            joint_names = [f"joint{i}" for i in range(len(positions))]
            extracted_data[key] = {
                "timestamp": [],
                "joint_names": joint_names,
                "position": [],
                "velocity": [],
                "effort": [],
            }

        extracted_data[key]["timestamp"].append(time_s)
        extracted_data[key]["position"].append(positions)
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
        time_ns = (ros_msg.header.stamp.sec * 1e9 + ros_msg.header.stamp.nanosec) / 1e9

        cam_name = self.config.camera_topic_mapping[message.channel.topic]
        extracted_data[cam_name]["timestamp"].append(time_ns)

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
        time_ns = (ros_msg.header.stamp.sec * 1e9 + ros_msg.header.stamp.nanosec) / 1e9

        cam_name = self._compressed_topic_mapping[message.channel.topic]
        extracted_data[cam_name]["timestamp"].append(time_ns)

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
    ):
        """
        Initialize buffered stream extractor.

        Args:
            config: Data configuration specifying topics and camera mapping
            buffer_seconds: Total buffer window size in seconds (default: 5.0)
            fps: Frame rate for buffer size calculation (default: 30)
            quiet: If True, suppress all print output (default: False)
            progress_callback: Called with frames_yielded count after each frame
        """
        self.config = config
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.half_buffer = int(buffer_seconds * fps / 2)  # 75 frames for 5s @ 30fps
        self.full_buffer = self.half_buffer * 2  # 150 frames
        self.target_size = tuple(config.image_resolution)  # (width, height)
        self.quiet = quiet
        self.progress_callback = progress_callback

        # Joint name pattern for parsing (reuse from DataExtractor)
        self._joint_pattern = config.joint_name_pattern

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

        # Joint state buffers: {(role, robot): {'buffer': deque, 'joint_names': list}}
        joint_buffers: Dict[Tuple[str, str], Dict] = {}

        # Build topic list for reading (cameras + joint states + action commands)
        all_topics = list(all_camera_topics)
        all_topics.append(self.config.robot_state_topic)

        # Include action command topics if configured (quest teleop mode)
        action_topic_set = set()
        if self.config.action_topics:
            action_topic_set = set(self.config.action_topics.keys())
            all_topics.extend(action_topic_set)

        # Get main camera (first one in config)
        main_cam = list(self.config.camera_topic_mapping.values())[0]

        cursor = 0  # Index of frame to process next
        frames_yielded = 0
        first_ts = None

        for message in reader.read_messages(topics=all_topics):
            topic = message.channel.topic

            # Handle joint state messages
            if topic == self.config.robot_state_topic:
                self._buffer_joint_state(message, joint_buffers)
                continue

            # Handle action command messages (quest teleop mode)
            if topic in action_topic_set:
                self._buffer_action_command(message, topic, joint_buffers)
                continue

            # Handle camera messages
            if topic not in topic_to_cam:
                continue
            cam_name = topic_to_cam[topic]

            # Get timestamp
            ros_msg = message.ros_msg
            time_ns = (ros_msg.header.stamp.sec * 1e9 + ros_msg.header.stamp.nanosec) / 1e9

            # Track first timestamp for relative timing
            if first_ts is None:
                first_ts = time_ns

            # Decode image
            if topic in camera_topics:
                img = decode_image(ros_msg.data, ros_msg.encoding, ros_msg.height, ros_msg.width)
            else:
                img = decode_compressed_image(ros_msg.data, ros_msg.format)

            # Add to buffer
            camera_buffers[cam_name].append((time_ns, img))

            # Start processing when main camera buffer reaches threshold
            main_buffer_len = len(camera_buffers[main_cam])

            # Condition: buffer has at least half_buffer frames ahead of cursor
            if main_buffer_len >= self.half_buffer + cursor:
                frame = self._align_frame_at_cursor(
                    camera_buffers, joint_buffers, cursor, main_cam, task, resize_image
                )
                if frame is not None:
                    yield frame
                    frames_yielded += 1

                    # Progress reporting
                    if self.progress_callback:
                        self.progress_callback(frames_yielded)
                    elif not self.quiet and frames_yielded % 100 == 0:
                        print(f"[BufferedStream] Processed {frames_yielded} frames...")

                cursor += 1

                # Once buffer reaches full size, remove oldest to maintain size
                if main_buffer_len > self.full_buffer:
                    for buffer in camera_buffers.values():
                        if len(buffer) > 0:
                            buffer.popleft()

                    # Sync joint buffers to remove old samples
                    if camera_buffers[main_cam]:
                        new_oldest_ts = camera_buffers[main_cam][0][0]
                        self._sync_joint_buffers(joint_buffers, new_oldest_ts)

                    cursor -= 1  # Adjust cursor after removal

        # Flush: process remaining frames in buffer
        if not self.quiet:
            print("[BufferedStream] Flushing remaining buffer...")
        while cursor < len(camera_buffers[main_cam]):
            frame = self._align_frame_at_cursor(
                camera_buffers, joint_buffers, cursor, main_cam, task, resize_image
            )
            if frame is not None:
                yield frame
                frames_yielded += 1
                if self.progress_callback:
                    self.progress_callback(frames_yielded)
            cursor += 1

        if not self.quiet:
            print(f"[BufferedStream] [OK] Extracted {frames_yielded} frames total")

    def _align_frame_at_cursor(
        self,
        camera_buffers: Dict[str, deque],
        joint_buffers: Dict[Tuple[str, str], Dict],
        cursor: int,
        main_cam: str,
        task: str,
        resize_func,
    ) -> Optional[Dict[str, Any]]:
        """
        Align frame at cursor position using entire buffer for matching.

        Args:
            camera_buffers: Per-camera buffers
            joint_buffers: Joint state buffers (keyed by (role, robot))
            cursor: Index of frame to align in main camera buffer
            main_cam: Name of main camera
            task: Task name
            resize_func: Function to resize images

        Returns:
            Aligned frame dictionary, or None if not all data available
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
                return None

            # Search entire buffer for nearest timestamp match
            nearest_idx = self._find_nearest_in_buffer(buffer, main_ts)
            if nearest_idx is not None:
                _, img = buffer[nearest_idx]
                resized_img = resize_func(img, self.target_size)
                frame[f"observation.images.{cam_name}"] = resized_img
            else:
                return None

        # Align joint states
        if joint_buffers:
            joint_aligned = self._align_joint_states(joint_buffers, main_ts)
            if joint_aligned is None:
                return None  # Skip frame if joint states not available
            frame.update(joint_aligned)

        frame["task"] = task
        return frame

    def _align_joint_states(
        self,
        joint_buffers: Dict[Tuple[str, str], Dict],
        target_ts: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Align joint states to target timestamp.

        For multi-robot setups (left/right), concatenates features into single arrays:
        - observation.state = [left_joints..., right_joints...]
        - action = [left_joints..., right_joints...]

        Args:
            joint_buffers: Joint state buffers keyed by (role, robot)
            target_ts: Target timestamp to align to

        Returns:
            Dictionary with aligned joint state features, or None if data missing
        """
        # Collect data by role, then concatenate robots in sorted order
        obs_data = {}  # {robot: {pos, vel, eff}}
        action_data = {}  # {robot: {pos}}

        for (role, robot), data in joint_buffers.items():
            buffer = data["buffer"]

            if len(buffer) == 0:
                return None  # Required joint data missing

            # Find nearest joint state sample
            nearest_idx = self._find_nearest_in_buffer(buffer, target_ts)
            if nearest_idx is None:
                return None

            ts, pos, vel, eff = buffer[nearest_idx]

            if role == "observation":
                obs_data[robot] = {
                    "pos": pos.copy(),
                    "vel": vel.copy() if vel.size > 0 else None,
                    "eff": eff.copy() if eff.size > 0 else None,
                }
            else:  # action
                action_data[robot] = {"pos": pos.copy()}

        # Check if multi-robot (has named robots like 'left', 'right')
        robots = sorted([r for r in set(obs_data.keys()) | set(action_data.keys()) if r])

        if robots:
            # Multi-robot: concatenate in sorted order (left, right)
            result = {}

            # Concatenate observation state
            obs_positions = [obs_data[r]["pos"] for r in robots if r in obs_data]
            if obs_positions:
                result["observation.state"] = np.concatenate(obs_positions)

            # Concatenate observation velocity (if available)
            obs_velocities = [
                obs_data[r]["vel"]
                for r in robots
                if r in obs_data and obs_data[r]["vel"] is not None
            ]
            if obs_velocities:
                result["observation.velocity"] = np.concatenate(obs_velocities)

            # Concatenate observation effort (if available)
            obs_efforts = [
                obs_data[r]["eff"]
                for r in robots
                if r in obs_data and obs_data[r]["eff"] is not None
            ]
            if obs_efforts:
                result["observation.effort"] = np.concatenate(obs_efforts)

            # Concatenate action
            action_positions = [action_data[r]["pos"] for r in robots if r in action_data]
            if action_positions:
                result["action"] = np.concatenate(action_positions)

            return result
        else:
            # Single robot: use original naming (no prefix)
            result = {}

            if "" in obs_data:
                result["observation.state"] = obs_data[""]["pos"]
                if obs_data[""]["vel"] is not None:
                    result["observation.velocity"] = obs_data[""]["vel"]
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

    def _parse_joint_name(self, joint_name: str) -> Tuple[str, str, str]:
        """Parse joint name using shared utility."""
        return parse_joint_name(joint_name, self._joint_pattern)

    def _buffer_joint_state(
        self,
        message,
        joint_buffers: Dict[Tuple[str, str], Dict],
    ) -> None:
        """
        Parse joint state message and add to appropriate buffers.

        Args:
            message: MCAP message with ROS JointState
            joint_buffers: Dict keyed by (role, robot) containing buffer and metadata
        """
        ros_msg = message.ros_msg
        timestamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec / 1e9

        # Group joints by (role, robot)
        grouped: Dict[Tuple[str, str], Dict] = {}

        for i, joint_name in enumerate(ros_msg.name):
            try:
                role, robot, joint_id = self._parse_joint_name(joint_name)
            except DataExtractionError:
                continue  # Skip unparseable joints

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

        # Add to buffers
        for key, data in grouped.items():
            if key not in joint_buffers:
                joint_buffers[key] = {
                    "buffer": deque(),
                    "joint_names": data["joint_ids"],  # Store joint names once
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

        Args:
            message: MCAP message with Float64MultiArray
            topic: The ROS topic name
            joint_buffers: Dict keyed by (role, robot) containing buffer and metadata
        """
        ros_msg = message.ros_msg
        # Float64MultiArray has no header; use MCAP recording timestamp
        timestamp = message.log_time / 1e9

        # Map topic to arm identifier (e.g., "left" or "right")
        robot = self.config.action_topics[topic]
        key = ("action", robot)

        # Extract position data from Float64MultiArray.data
        positions = list(ros_msg.data)

        if key not in joint_buffers:
            joint_buffers[key] = {
                "buffer": deque(),
                "joint_names": [f"joint{i}" for i in range(len(positions))],
            }

        pos = np.array(positions, dtype=np.float32)
        vel = np.array([], dtype=np.float32)
        eff = np.array([], dtype=np.float32)

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
