"""
Single-Process Strategy (mode: single)

Uses ROS2 callbacks with threading for observation acquisition.
Can achieve ~30 Hz because OpenCV releases the GIL during JPEG decompression.

Use cases:
- Debugging observation pipeline issues
- Simple testing without worker process complexity
- Environments where multiprocessing is not available

The multi-process strategy (mode: mp) provides better isolation (worker crashes
don't affect the main process) but both strategies can achieve similar FPS.
"""

import threading
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CompressedImage, JointState

from ..observation_manager import ObservationManager
from ..image_converter import ImageConverter


class SingleProcessStrategy:
    """
    Single-process strategy using callbacks + threading.

    Runs all processing in the main process with multi-threaded callbacks.
    Can achieve ~30 Hz because OpenCV releases the GIL during JPEG decompression.

    Use this for debugging or when multiprocessing is not available.
    For better isolation, use multi-process mode (mode: mp).
    """

    def __init__(self):
        self._node = None
        self._config = None
        self._camera_names: List[str] = []
        self._camera_mapping: Dict[str, str] = {}
        self._joint_names_config: dict = {}
        self._device: str = "cuda"

        # Components
        self._obs_manager: Optional[ObservationManager] = None
        self._image_converter: Optional[ImageConverter] = None

        # Thread safety
        self._obs_lock = threading.Lock()

        # Callback group for parallel execution
        self._sensor_callback_group: Optional[ReentrantCallbackGroup] = None

        # Topic mapping
        self._ros_to_ml_camera: Dict[str, str] = {}

        # Metrics tracker (set via setup)
        self._metrics = None

        # Status tracking
        self._last_incomplete_reason: str = ""

    def setup(
        self,
        node: Any,
        config: dict,
        camera_mapping: Dict[str, str],
        joint_names_config: dict,
        joint_state_topic: str,
        image_shape: tuple,
        metrics: Any = None,
    ) -> None:
        """Initialize observation manager and setup subscriptions."""
        self._node = node
        self._config = config
        self._camera_mapping = camera_mapping
        self._camera_names = list(camera_mapping.values())
        self._joint_names_config = joint_names_config
        self._device = config.get('device', 'cuda')
        self._metrics = metrics

        # Initialize components
        self._obs_manager = ObservationManager(device=self._device)
        self._image_converter = ImageConverter()

        # Setup callback groups
        self._sensor_callback_group = ReentrantCallbackGroup()

        # Setup ROS2 subscriptions
        self._setup_subscriptions(joint_state_topic)

        self._node.get_logger().info(
            "SingleProcessStrategy initialized. "
            "For better isolation, consider multi-process mode (mode: mp)."
        )

    def _setup_subscriptions(self, joint_state_topic: str) -> None:
        """Setup camera and joint state subscriptions."""
        # QoS profiles
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Joint state subscription
        self._node.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_state_callback,
            joint_qos,
            callback_group=self._sensor_callback_group,
        )
        self._node.get_logger().info(f"Subscribed to: {joint_state_topic}")

        # Camera subscriptions
        for ros_topic, ml_name in self._camera_mapping.items():
            self._ros_to_ml_camera[ros_topic] = ml_name
            is_compressed = ros_topic.endswith('/compressed')
            msg_type = CompressedImage if is_compressed else Image

            if is_compressed:
                callback = lambda msg, t=ros_topic: self._compressed_image_callback(msg, t)
            else:
                callback = lambda msg, t=ros_topic: self._image_callback(msg, t)

            self._node.create_subscription(
                msg_type,
                ros_topic,
                callback,
                qos_profile_sensor_data,
                callback_group=self._sensor_callback_group,
            )
            self._node.get_logger().info(f"Subscribed to: {ros_topic} -> {ml_name}")

    def _joint_state_callback(self, msg: JointState) -> None:
        """Handle joint state updates."""
        # Record metrics
        if self._metrics:
            self._metrics.record_joint_state()

        filtered_msg = self._filter_joint_state(msg)
        if filtered_msg:
            with self._obs_lock:
                self._obs_manager.update_joint_state(filtered_msg)

    def _filter_joint_state(self, msg: JointState) -> Optional[JointState]:
        """Filter joint state to observation joints only."""
        if not self._joint_names_config:
            return msg

        obs_prefix = self._joint_names_config.get('observation_prefix', 'follower')
        sep = self._joint_names_config.get('separator', '_')
        arm_mapping = self._joint_names_config.get('arm_mapping', {'l': 'left', 'r': 'right'})
        joint_order = self._joint_names_config.get(
            'model_joint_order',
            self._joint_names_config.get('joint_order', []),
        )

        ordered_positions = []
        ordered_names = []

        for arm_key in sorted(arm_mapping.keys()):
            for joint_id in joint_order:
                expected_name = f"{obs_prefix}{sep}{arm_key}{sep}{joint_id}"
                if expected_name in msg.name:
                    idx = msg.name.index(expected_name)
                    ordered_positions.append(
                        msg.position[idx] if idx < len(msg.position) else 0.0
                    )
                    ordered_names.append(expected_name)
                else:
                    ordered_positions.append(0.0)
                    ordered_names.append(expected_name)

        filtered = JointState()
        filtered.header = msg.header
        filtered.name = ordered_names
        filtered.position = ordered_positions
        return filtered

    def _image_callback(self, msg: Image, ros_topic: str) -> None:
        """Handle raw camera image updates."""
        try:
            ml_name = self._ros_to_ml_camera.get(ros_topic, ros_topic)
            # Record metrics
            if self._metrics:
                self._metrics.record_image(ml_name)
            np_image = self._image_converter.imgmsg_to_numpy(msg, desired_encoding='rgb8')
            with self._obs_lock:
                self._obs_manager.update_image(ml_name, np_image)
        except Exception as e:
            self._node.get_logger().warn(f"Image conversion failed ({ros_topic}): {e}")

    def _compressed_image_callback(self, msg: CompressedImage, ros_topic: str) -> None:
        """Handle compressed camera image updates."""
        try:
            ml_name = self._ros_to_ml_camera.get(ros_topic, ros_topic)
            # Record metrics
            if self._metrics:
                self._metrics.record_image(ml_name)
            np_image = self._image_converter.compressed_imgmsg_to_numpy(
                msg, desired_encoding='rgb8'
            )
            with self._obs_lock:
                self._obs_manager.update_image(ml_name, np_image)
        except Exception as e:
            self._node.get_logger().warn(
                f"Compressed image conversion failed ({ros_topic}): {e}"
            )

    def get_observation(
        self,
        camera_names: List[str],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get observation if complete."""
        with self._obs_lock:
            if not self._obs_manager.has_complete_observation(camera_names):
                self._last_incomplete_reason = self._get_incomplete_reason_internal()
                return None

            # Get observation (copy data while holding lock)
            observation = self._obs_manager.get_observation(camera_names)

        return observation

    def _get_incomplete_reason_internal(self) -> str:
        """Get reason why observation is incomplete (called under lock)."""
        reasons = []

        if self._obs_manager.latest_joint_state is None:
            reasons.append("no joint state")

        for cam in self._camera_names:
            if cam not in self._obs_manager.latest_images:
                reasons.append(f"missing camera '{cam}'")

        return ", ".join(reasons) if reasons else "unknown"

    def get_current_joint_positions(self) -> Dict[str, float]:
        """Get current joint positions for delta limiting."""
        if self._obs_manager.latest_joint_state is None:
            return {}
        js = self._obs_manager.latest_joint_state
        return {
            name: js.position[i]
            for i, name in enumerate(js.name)
            if i < len(js.position)
        }

    def get_incomplete_reason(self) -> str:
        """Get reason why observation is incomplete."""
        return self._last_incomplete_reason

    def record_metrics(self, metrics_tracker: Any) -> None:
        """Record metrics - handled by callbacks."""
        pass

    def cleanup(self) -> None:
        """No special cleanup needed for single-process strategy."""
        self._node.get_logger().info("SingleProcessStrategy cleanup complete")
