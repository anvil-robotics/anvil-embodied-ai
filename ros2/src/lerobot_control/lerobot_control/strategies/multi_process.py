"""
Multi-Process Inference Strategy

Uses separate worker processes for image acquisition, providing true
parallelism (no GIL contention), process isolation, and crash resilience.

Architecture:
- Image Worker Processes: One per camera, subscribe to topics, decompress JPEG,
  write to shared memory
- Main Process: Read from shared memory, run model inference, publish actions
"""

import multiprocessing as mp
import time
from typing import Any

import torch
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from ..image_worker import run_image_worker
from ..shared_image_buffer import SharedImageBuffer


class MultiProcessStrategy:
    """
    Multi-process strategy using shared memory and worker processes.

    Provides better process isolation - worker crashes don't affect the
    main inference process. This is the default mode (mode: mp).
    """

    def __init__(self):
        self._node = None
        self._config = None
        self._camera_names: list[str] = []
        self._camera_mapping: dict[str, str] = {}
        self._joint_names_config: dict = {}
        self._image_shape: tuple = (480, 640, 3)

        # Shared memory buffer
        self._image_buffer: SharedImageBuffer | None = None

        # Worker processes
        self._worker_processes: list[mp.Process] = []
        self._stop_event: mp.Event | None = None

        # Joint state (handled in main process - lightweight)
        self._joint_positions: dict[str, float] | None = None
        self._joint_velocities: dict[str, float] | None = None
        self._joint_efforts: dict[str, float] | None = None
        self._joint_timestamp: float | None = None

        # Metrics tracker (set via setup)
        self._metrics = None

        # Status tracking
        self._last_incomplete_reason: str = ""

    def setup(
        self,
        node: Any,
        config: dict,
        camera_mapping: dict[str, str],
        joint_names_config: dict,
        joint_state_topic: str,
        image_shape: tuple,
        metrics: Any = None,
        callback_group: Any = None,
        debug_image_dir: str | None = None,
    ) -> None:
        """Initialize shared memory and start worker processes."""
        self._node = node
        self._config = config
        self._camera_mapping = camera_mapping
        self._camera_names = list(camera_mapping.values())
        self._joint_names_config = joint_names_config
        self._image_shape = image_shape
        self._metrics = metrics
        self._callback_group = callback_group
        self._debug_image_dir = debug_image_dir

        # Create shared memory buffers
        self._setup_shared_memory()

        # Start image worker processes
        self._start_workers()

        # Subscribe to joint states (lightweight, runs in main process)
        self._setup_joint_subscription(joint_state_topic)

        self._node.get_logger().info(
            f"MultiProcessStrategy initialized with {len(self._worker_processes)} image workers"
        )

    def _setup_shared_memory(self) -> None:
        """Create shared memory buffers for all cameras."""
        self._node.get_logger().info("Setting up shared memory buffers...")

        self._image_buffer = SharedImageBuffer(
            camera_names=self._camera_names,
            image_shape=self._image_shape,
            create=True,
        )

        self._node.get_logger().info(f"Created shared memory for {len(self._camera_names)} cameras")

    def _start_workers(self) -> None:
        """Start image worker processes."""
        self._node.get_logger().info("Starting image worker processes...")

        # Use 'spawn' context for clean subprocess start
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        self._worker_processes = []

        for topic, camera_name in self._camera_mapping.items():
            p = ctx.Process(
                target=run_image_worker,
                args=(topic, camera_name, self._image_shape),
                kwargs={
                    "stop_event": self._stop_event,
                    "debug_dir": self._debug_image_dir,
                },
                name=f"image_worker_{camera_name}",
            )
            p.start()
            self._worker_processes.append(p)
            self._node.get_logger().info(f"Started worker: {topic} -> {camera_name} (PID: {p.pid})")

        # Give workers time to connect to shared memory
        time.sleep(0.5)

    def _setup_joint_subscription(self, joint_state_topic: str) -> None:
        """Setup joint state subscription (runs in main process)."""
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._node.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_callback,
            joint_qos,
            callback_group=self._callback_group,
        )
        self._node.get_logger().info(f"Subscribed to: {joint_state_topic}")

    def _joint_callback(self, msg: JointState) -> None:
        """Process joint state (lightweight, no GIL issue)."""
        # Record metrics
        if self._metrics:
            self._metrics.record_joint_state()

        self._joint_positions = dict(zip(msg.name, msg.position))
        if msg.velocity:
            self._joint_velocities = dict(zip(msg.name, msg.velocity))
        if msg.effort:
            self._joint_efforts = dict(zip(msg.name, msg.effort))
        self._joint_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def get_observation(
        self,
        camera_names: list[str],
    ) -> dict[str, torch.Tensor] | None:
        """Get observation from shared memory if complete."""
        # Check for complete observation from shared memory
        images = self._image_buffer.read_all_if_ready()

        if images is None:
            # Not all cameras have new frames yet
            missing = []
            for name in camera_names:
                if not self._image_buffer.has_new_frame(name):
                    missing.append(name)
            self._last_incomplete_reason = f"waiting for cameras: {missing}"
            return None

        if self._joint_positions is None:
            self._last_incomplete_reason = "waiting for joint state"
            return None

        # Build observation dict
        observation = self._build_observation(images)
        return observation

    def _build_observation(
        self,
        images: dict[str, tuple],
    ) -> dict[str, torch.Tensor]:
        """Build observation dict from shared memory images and joint state."""
        observation = {}

        # Add images (already decompressed by workers)
        for camera_name, (image, timestamp) in images.items():
            # Convert to tensor and normalize to [0, 1]
            image_tensor = torch.from_numpy(image).float() / 255.0
            # Rearrange to (C, H, W) and add batch dimension
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            observation[f"observation.images.{camera_name}"] = image_tensor

        # Build state observations (position / velocity / effort) based on config
        if self._joint_positions:
            obs_prefix = self._joint_names_config.get("observation_prefix", "follower")
            sep = self._joint_names_config.get("separator", "_")
            arm_mapping = self._joint_names_config.get("arm_mapping", {"l": "left", "r": "right"})
            joint_order = self._joint_names_config.get("model_joint_order", [])
            state_features = self._joint_names_config.get("state_features", ["position"])

            feature_map = {
                "position": (self._joint_positions, "observation.state"),
                "velocity": (self._joint_velocities, "observation.velocity"),
                "effort": (self._joint_efforts, "observation.effort"),
            }

            for feature in state_features:
                if feature not in feature_map:
                    continue
                data_dict, obs_key = feature_map[feature]
                ordered = []
                for arm_key in sorted(arm_mapping.keys()):
                    for joint_id in joint_order:
                        joint_name = f"{obs_prefix}{sep}{arm_key}{sep}{joint_id}"
                        val = data_dict.get(joint_name, 0.0) if data_dict else 0.0
                        ordered.append(val)
                observation[obs_key] = torch.tensor(ordered, dtype=torch.float32).unsqueeze(0)

        return observation

    def get_current_joint_positions(self) -> dict[str, float]:
        """Get current joint positions for delta limiting."""
        if self._joint_positions is None:
            return {}
        return self._joint_positions

    def get_incomplete_reason(self) -> str:
        """Get reason why observation is incomplete."""
        return self._last_incomplete_reason

    def record_metrics(self, metrics_tracker: Any) -> None:
        """Record metrics - joint state is tracked via callback."""
        # Joint state metrics are recorded by main node
        # Image metrics tracked per-camera from shared memory frame counters
        pass

    def get_frame_counters(self) -> dict[str, int]:
        """Get frame counters from shared memory (for stats logging)."""
        if self._image_buffer:
            return self._image_buffer.get_frame_counters()
        return {}

    def cleanup(self) -> None:
        """Stop workers and clean up shared memory."""
        if self._node:
            self._node.get_logger().info("Stopping worker processes...")

        # Signal workers to stop
        if self._stop_event:
            self._stop_event.set()

        # Wait for workers to finish
        for p in self._worker_processes:
            p.join(timeout=2.0)
            if p.is_alive():
                if self._node:
                    self._node.get_logger().warn(f"Force terminating worker {p.name}")
                p.terminate()
                p.join(timeout=1.0)

        if self._node:
            self._node.get_logger().info("All workers stopped")

        # Clean up shared memory
        if self._image_buffer:
            self._image_buffer.unlink()
            self._image_buffer = None
