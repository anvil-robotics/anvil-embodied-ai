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
import threading
import time
from typing import Any

import torch
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from ..ee_obs_sequence_guard import SequenceStalenessGuard
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

        # EE state: ordered flat list per arm (x,y,z,qx,qy,qz,qw,gripper), set in EE mode
        self._ee_state_by_arm: dict[str, list[float]] | None = None
        self._is_ee: bool = False
        # Ordered arm list for building observation.state in EE mode
        self._ee_arm_order: list[str] = []
        # Fake-hardware-only: whether /ee_pose_{arm} is the mock's MockEEPose echo
        # (sequence-guarded) rather than a real controller's plain CommandedEEPose
        # (no sequence — see MockEEPose.msg's docstring for why this is a hard
        # split, not a runtime fallback). Set in setup().
        self._mock_ee_pose_echo: bool = False
        # Detects stale/repeated /ee_pose_{arm} echoes via MockEEPose.sequence.
        # Only ever constructed when _mock_ee_pose_echo is true — see
        # ee_obs_sequence_guard.py's module docstring for why this mechanism has
        # no real-hardware role at all, not even a degraded/fallback one.
        self._ee_seq_guard: SequenceStalenessGuard | None = None
        # EE subscriptions run on a ReentrantCallbackGroup (set via `callback_group`
        # in setup()), so _cb can execute concurrently on different executor threads
        # for different arms (or even overlapping messages on the same arm). Guards
        # the check-then-write (mock path) or plain write (real path) so a race
        # can't let an older message's write land after a newer one's.
        self._ee_state_lock = threading.Lock()

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
        video_dir: str | None = None,
        mock_ee_pose_echo: bool = False,
    ) -> None:
        """Initialize shared memory and start worker processes.

        ``mock_ee_pose_echo``: true only when ``/ee_pose_{arm}`` is the
        fake-hardware mock's echo (see MockEEPose.msg) — real hardware leaves
        this false, and the sequence-staleness-guard machinery is never even
        constructed in that case, not just unused.
        """
        self._node = node
        self._config = config
        self._camera_mapping = camera_mapping
        self._camera_names = list(camera_mapping.values())
        self._joint_names_config = joint_names_config
        self._image_shape = image_shape
        self._metrics = metrics
        self._callback_group = callback_group
        self._debug_image_dir = debug_image_dir
        self._video_dir = video_dir
        self._mock_ee_pose_echo = mock_ee_pose_echo

        # Create shared memory buffers
        self._setup_shared_memory()

        # Start image worker processes
        self._start_workers()

        # EE mode: subscribe to CommandedEEPose topics; otherwise subscribe to JointState
        arms_config: dict = config.get("arms", {})
        ee_arms = {
            name: ac for name, ac in arms_config.items() if "ee_command_topic" in ac
        }
        if ee_arms:
            self._is_ee = True
            self._ee_arm_order = list(ee_arms.keys())
            self._ee_state_by_arm = {}
            if self._mock_ee_pose_echo:
                # Consecutive stale /ee_pose_{arm} reads (per arm) before a fault is
                # logged, and before the guard gives up and falls back entirely (see
                # ee_obs_sequence_guard.py) — configurable, but this whole mechanism
                # is fake-hardware-only; real hardware never constructs a guard at all.
                stale_threshold = config.get("ee_obs_stale_threshold", 10)
                degraded_threshold = config.get("ee_obs_degraded_after_streak", 50)
                self._ee_seq_guard = SequenceStalenessGuard(
                    stale_fault_threshold=stale_threshold,
                    degraded_after_streak=degraded_threshold,
                )
            self._setup_ee_subscriptions(ee_arms)
        else:
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
                    "video_dir": self._video_dir,
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

    def _setup_ee_subscriptions(self, ee_arms: dict) -> None:
        """Subscribe to /ee_pose_{arm} for each EE arm (EE mode).

        Message type is a hard split, not a runtime fallback (see
        MockEEPose.msg's docstring): the mock's echo publishes MockEEPose
        (sequence-guarded); real hardware publishes plain CommandedEEPose (no
        guard at all — that mechanism has no real-hardware role, see
        ee_obs_sequence_guard.py).
        """
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        msg_type = self._mock_ee_pose_echo_msg_type()
        make_cb = self._make_mock_ee_cb if self._mock_ee_pose_echo else self._make_real_ee_cb
        for arm_name, arm_config in ee_arms.items():
            obs_topic = arm_config.get("ee_obs_topic", f"/ee_pose_{arm_name}")
            self._node.create_subscription(
                msg_type,
                obs_topic,
                make_cb(arm_name),
                qos,
                callback_group=self._callback_group,
            )
            self._node.get_logger().info(
                f"Subscribed to EE obs: {obs_topic} (arm={arm_name}, "
                f"type={msg_type.__name__})"
            )

    def _mock_ee_pose_echo_msg_type(self):
        if self._mock_ee_pose_echo:
            from anvil_msgs.msg import MockEEPose
            return MockEEPose
        from anvil_msgs.msg import CommandedEEPose
        return CommandedEEPose

    def _make_mock_ee_cb(self, name: str):
        """Fake-hardware path: MockEEPose, sequence-guarded (see setup())."""
        def _cb(msg) -> None:
            if self._metrics:
                self._metrics.record_joint_state()

            # EE subscriptions run on a ReentrantCallbackGroup, so this body
            # can execute concurrently on multiple executor threads — without
            # a lock, two overlapping calls could both pass the staleness
            # check against the same pre-update last-seen value and then
            # write _ee_state_by_arm out of order, silently regressing the
            # anchor to an older pose despite each check individually being
            # "correct". The lock makes check-then-write atomic.
            with self._ee_state_lock:
                is_stale, streak = self._ee_seq_guard.check(name, msg.sequence)
                if is_stale:
                    # Sequence didn't advance — this echo is a repeat of one we
                    # already consumed (see ee_obs_sequence_guard.py for why
                    # sequence, not header.stamp, is the signal). Keep the
                    # previous anchor rather than overwrite it with a stale
                    # value: simply don't touch _ee_state_by_arm this tick, so
                    # the next read (obs/publish loop) reuses the last-known-good
                    # pose — safer than skipping a publish tick outright, since
                    # it keeps the control loop's cadence intact.
                    self._node.get_logger().warn(
                        f"[ee_mode] stale /ee_pose_{name} read: sequence "
                        f"{msg.sequence} did not advance — reusing previous "
                        f"anchor (consecutive stale={streak})"
                    )
                    if self._ee_seq_guard.is_fault(streak):
                        self._node.get_logger().error(
                            f"[ee_mode] {name}: {streak} consecutive stale "
                            f"/ee_pose_{name} reads — observation feed appears "
                            "degraded or stuck"
                        )
                    if self._ee_seq_guard.just_degraded(streak):
                        # Given up on sequence for this arm — see
                        # ee_obs_sequence_guard.py's module docstring. Fall
                        # through and accept THIS read's pose too (no reason
                        # to also discard it); every future read is accepted
                        # unconditionally from here on.
                        self._node.get_logger().error(
                            f"[ee_mode] {name}: giving up on MockEEPose.sequence "
                            f"after {streak} consecutive stale reads — falling back "
                            "to always-accept (no staleness protection) for the rest "
                            "of this session. This shouldn't happen against the mock "
                            "(fake_hardware_node.py always advances it) — if it does, "
                            "something's wrong with the mock, not this consumer."
                        )
                    else:
                        return

                p = msg.base.pose.position
                o = msg.base.pose.orientation
                self._ee_state_by_arm[name] = [
                    p.x, p.y, p.z,
                    o.x, o.y, o.z, o.w,
                    msg.base.gripper,
                ]
                # TEMP DEBUG (gt_replay ee_delta divergence investigation).
                if getattr(self._node, "_debug", False):
                    self._node.get_logger().info(
                        f"[DEBUG-ANCHOR] ee_pose_cb arm={name} "
                        f"pos={[round(p.x, 5), round(p.y, 5), round(p.z, 5)]} "
                        f"t={time.monotonic():.4f}"
                    )
        return _cb

    def _make_real_ee_cb(self, name: str):
        """Real-hardware path: plain CommandedEEPose, no guard, no sequence.

        Unconditionally overwrites _ee_state_by_arm with whatever arrives —
        exactly the pre-sequence-guard "keep only latest" behavior. Still
        under the lock: the ReentrantCallbackGroup concurrency concern (two
        overlapping calls writing out of order) is orthogonal to the
        sequence-guard question and applies here too.
        """
        def _cb(msg) -> None:
            if self._metrics:
                self._metrics.record_joint_state()
            with self._ee_state_lock:
                p = msg.pose.position
                o = msg.pose.orientation
                self._ee_state_by_arm[name] = [
                    p.x, p.y, p.z,
                    o.x, o.y, o.z, o.w,
                    msg.gripper,
                ]
                if getattr(self._node, "_debug", False):
                    self._node.get_logger().info(
                        f"[DEBUG-ANCHOR] ee_pose_cb arm={name} "
                        f"pos={[round(p.x, 5), round(p.y, 5), round(p.z, 5)]} "
                        f"t={time.monotonic():.4f}"
                    )
        return _cb

    def get_ee_obs_sequence_snapshot(self) -> tuple[int | None, ...] | None:
        """Last-accepted MockEEPose.sequence per arm, in ``_ee_arm_order``.

        ``None`` outside EE mode, AND ``None`` outright on real hardware
        (``_ee_seq_guard`` is never constructed when ``mock_ee_pose_echo`` is
        false — see setup()) — this whole mechanism is fake-hardware-only.
        Lets a caller (the ee_delta publish loop) tell whether the mock's
        echo has genuinely advanced since it last consumed it — distinct from
        "the pose value happens to look the same," which is expected for a
        stationary arm and must not be treated as staleness. Same-tick guard
        as get_observation()'s own _ee_state_by_arm read, but sequence-based
        rather than value-based.

        A per-arm element is ``None`` when that arm has degraded (see
        ee_obs_sequence_guard.py) — its sequence isn't trustworthy anymore, so
        the caller should treat it as "always advanced" rather than gate on it.
        """
        if not self._is_ee or self._ee_seq_guard is None:
            return None
        return tuple(
            None if self._ee_seq_guard.is_degraded(arm)
            else (self._ee_seq_guard.last_accepted(arm) or 0)
            for arm in self._ee_arm_order
        )

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

        if self._is_ee:
            if not self._ee_state_by_arm:
                self._last_incomplete_reason = "waiting for EE pose"
                return None
        else:
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
        """Build observation dict from shared memory images and state."""
        observation = {}

        # Add images (already decompressed by workers)
        for camera_name, (image, timestamp) in images.items():
            # Convert to tensor and normalize to [0, 1]
            image_tensor = torch.from_numpy(image).float() / 255.0
            # Rearrange to (C, H, W) and add batch dimension
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            observation[f"observation.images.{camera_name}"] = image_tensor

        if self._is_ee:
            # EE state: concatenate per-arm [x,y,z,qx,qy,qz,qw,gripper] in arm order
            state_flat: list[float] = []
            for arm_name in self._ee_arm_order:
                state_flat.extend(self._ee_state_by_arm.get(arm_name, [0.0] * 8))
            observation["observation.state"] = torch.tensor(
                state_flat, dtype=torch.float32
            ).unsqueeze(0)
            return observation

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
