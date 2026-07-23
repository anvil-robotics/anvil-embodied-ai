#!/usr/bin/env python3
"""
LeRobot Inference Node for Robot Arms

Multi-process inference node with shared-memory image workers.

Usage:
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p config_file:=/path/to/config.yaml

Subscribes to:
    - Joint states topic (sensor_msgs/JointState)
    - Camera image topics (sensor_msgs/CompressedImage)

Publishes:
    - Forward position controller command topics (std_msgs/Float64MultiArray)
    - /monitor/obs_state, /monitor/raw_output, /monitor/control_cmd  (when monitor_enable:=true)
"""

import json
import math
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .action_limiter import ActionLimiter
from .delta_restore import resolve_action_type, restore_delta_chunk
from .metrics_tracker import MetricsTracker
from .model_loader import ModelLoader, set_deterministic_mode
from .policy_registry import (
    is_language_conditioned,
    resolve_rtc_inference,
    uses_sync_chunk_inference,
)


class LeRobotInferenceNode(Node):
    """
    ROS2 node for LeRobot model inference and robot control.

    Uses multi-process strategy with shared-memory image workers for
    GIL-free JPEG decompression and true parallel camera processing.
    """

    def __init__(self, parameter_overrides: list = None):
        super().__init__("lerobot_inference_node", parameter_overrides=parameter_overrides or [])

        self._subscription_callback_group = ReentrantCallbackGroup()

        self._setup_config()

        self.metrics = MetricsTracker()
        self.strategy = self._create_strategy()
        self.strategy.setup(
            node=self,
            config={"device": self.device, **self.config},
            camera_mapping=self.camera_mapping,
            joint_names_config=self.joint_names_config,
            joint_state_topic=self.joint_state_topic,
            image_shape=self.image_shape,
            metrics=self.metrics,
            callback_group=self._subscription_callback_group,
            debug_image_dir=self._debug_image_dir,
        )

        # Non-RTC action buffer. It is intentionally unbounded: deque(maxlen=10)
        # silently discarded the beginning of chunks longer than ten actions.
        # Sync-prefetch access is protected because inference and publication run
        # in separate threads.
        self._classic_action_deque: deque = deque()
        self._classic_action_lock = threading.Lock()
        # Reference joint state captured at the moment each action chunk was
        # generated (in model/observation order).  All queued steps in the chunk
        # share this reference so delta restoration is consistent with training.
        # _delta_ref_state and _abs_shadow_queue must always be reset together;
        # use _reset_delta_state() in any future reload or episode-boundary path.
        self._delta_ref_state: np.ndarray | None = None
        self._abs_shadow_queue: deque[np.ndarray] = deque()

        self._shutting_down: bool = False
        self._has_published: bool = False

        if not self.echo_topic_only:
            self._setup_model()

            self.action_limiter = ActionLimiter(
                max_delta=self.max_position_delta,
                min_delta_threshold=self.min_position_delta,
                model_joint_order=self.joint_names_config.get("model_joint_order", []),
                controller_joint_order=self.joint_names_config.get("controller_joint_order", []),
                logger=self.get_logger(),
            )

            # Resolve delta exclude indices in model joint order (used by chunk restore)
            _model_order = self.joint_names_config.get("model_joint_order", [])
            self._delta_exclude_indices = [
                _model_order.index(name)
                for name in self.delta_exclude_joints
                if name in _model_order
            ]

            self._setup_publishers()

            # Unified split-timer architecture for all models:
            #   _obs_update:    preprocess (+ inference for non-RTC policies)
            #   _publish_loop:  pop action from queue/deque -> publish
            self._obs_callback_group = MutuallyExclusiveCallbackGroup()
            self._publish_callback_group = MutuallyExclusiveCallbackGroup()

            self._obs_timer = self.create_timer(
                1.0 / self.control_freq,
                self._obs_update,
                callback_group=self._obs_callback_group,
            )
            self._publish_timer = self.create_timer(
                1.0 / self.control_freq,
                self._publish_loop,
                callback_group=self._publish_callback_group,
            )

        self._log_startup()

        # Debug mode: enables ActionSmoothTracker, queue depth stats, Action FPS
        self._smooth_tracker = None
        self._queue_depths: deque[int] = deque(maxlen=300)
        self._rtc_skip_count: int = 0
        self._sync_skip_count: int = 0
        self._sync_replaced_actions: int = 0
        if self._debug and not self.echo_topic_only and hasattr(self, "model"):
            from .action_smooth_tracker import ActionSmoothTracker

            total_action_dim = sum(
                ac.get("action_end", 0) - ac.get("action_start", 0)
                for ac in self.arms_config.values()
            )
            if total_action_dim > 0:
                self._smooth_tracker = ActionSmoothTracker(action_dim=total_action_dim)

        # Stats logging timer (in publish callback group to avoid race on _queue_depths)
        self._stats_log_interval = 5.0
        self._stats_timer = self.create_timer(
            self._stats_log_interval,
            self._log_input_stats,
            callback_group=self._publish_callback_group if not self.echo_topic_only else MutuallyExclusiveCallbackGroup(),
        )

        # Optional one-shot episode limit. Created after model loading so startup
        # time does not reduce the active control window.
        self._max_run_timer = None
        if not self.echo_topic_only and self.max_run_seconds > 0:
            self._max_run_timer = self.create_timer(
                self.max_run_seconds,
                self._on_max_run_elapsed,
                callback_group=self._publish_callback_group,
            )
            self.get_logger().info(
                f"Maximum run duration: {self.max_run_seconds:.1f}s (starts after model load)"
            )

        # Windowed rate tracking
        self._prev_log_time: float | None = None
        self._prev_joint_count: int = 0
        self._prev_control_count: int = 0
        self._prev_inference_count: int = 0
        self._prev_action_output_count: int = 0
        self._prev_frame_counters: dict[str, int] = {}

    def _setup_config(self) -> None:
        """Declare ROS2 params, load YAML, and read all checkpoint metadata."""
        self.declare_parameter("model_path", "")
        self.declare_parameter("config_file", "")
        self.declare_parameter("control_frequency", 30.0)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("deterministic", False)
        self.declare_parameter("deterministic_seed", 42)
        self.declare_parameter("echo_topic_only", False)
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_image_dir", "")
        self.declare_parameter("monitor_enable", False)
        self.declare_parameter("max_run_seconds", 0)

        # Static fields from ROS2 params
        self.echo_topic_only = self.get_parameter("echo_topic_only").value
        self._debug = self.get_parameter("debug").value
        self._monitor_enable: bool = self.get_parameter("monitor_enable").value
        self.max_run_seconds: float = float(self.get_parameter("max_run_seconds").value)
        _debug_image_dir = self.get_parameter("debug_image_dir").value
        self._debug_image_dir: str | None = _debug_image_dir if _debug_image_dir else None
        self.model_path = self.get_parameter("model_path").value
        if not self.model_path and not self.echo_topic_only:
            raise ValueError("model_path parameter is required")

        self.control_freq = self.get_parameter("control_frequency").value
        self.device = self.get_parameter("device").value

        # Load YAML config
        config_file = self.get_parameter("config_file").value
        self.config = self._load_yaml_config(config_file)

        # Fields from YAML config
        safety_config = self.config.get("safety", {})
        self.max_position_delta = safety_config.get("max_position_delta", 0.1)
        self.min_position_delta = safety_config.get("min_position_delta", None)

        self.joint_state_topic = self.config.get("joint_state_topic", "/joint_states")
        _cameras_cfg: dict = self.config.get("cameras", {})
        self.camera_mapping = _cameras_cfg.get("mapping", {})
        self.camera_names = list(self.camera_mapping.values())

        # Build per-camera expected fps dict (camera name → expected fps).
        # Warning threshold = expected * 2/3, independent of control_frequency.
        _global_expected_fps: float = _cameras_cfg.get("fps", 30.0)
        _fps_overrides: dict = _cameras_cfg.get("fps_overrides", {})
        # overrides are keyed by ROS topic; map to camera name via camera_mapping
        self._expected_camera_fps: dict[str, float] = {
            name: _fps_overrides.get(topic, _global_expected_fps)
            for topic, name in self.camera_mapping.items()
        }

        self.arms_config = self.config.get("arms", {})
        self.joint_names_config = self.config.get("joint_names", {})

        # Inference tuning — per model type (resolved after model_type is known)
        self._tuning_config = self.config.get("inference_tuning", {})

        # --- Checkpoint metadata (lightweight JSON reads, no tensor loading) ---
        # Skip in echo_topic_only mode — no checkpoint needed
        meta = {} if self.echo_topic_only else self._read_checkpoint_metadata()

        # image_shape: from config.json input_features — must match training
        # Default (480, 640, 3) is used only in echo_topic_only mode with no checkpoint
        self.image_shape = meta.get("image_shape", (480, 640, 3))

        # model_type: from config.json, YAML overrides if explicitly set
        model_cfg = self.config.get("model", {})
        self.model_type = model_cfg.get("type") or meta.get("model_type")
        rtc_requested = self._tuning_config.get("rtc", {}).get("enabled")
        if rtc_requested is not None and not isinstance(rtc_requested, bool):
            raise ValueError("inference_tuning.rtc.enabled must be true, false, or null")
        if self.echo_topic_only:
            self._rtc_enabled = False
        else:
            self._rtc_enabled = resolve_rtc_inference(
                self.model_type,
                rtc_requested,
            )

        # action_type from anvil_config.json — must match training
        self.action_type: str = resolve_action_type(meta)
        self.use_delta_actions: bool = self.action_type in ("delta_obs_t", "delta_sequential")
        self.delta_exclude_joints: list[str] = meta.get("delta_exclude_joints", [])

        # Resolve delta exclude joint indices (in model output order)
        # Will be finalized after joint_names_config is loaded.
        self._delta_exclude_indices: list[int] = []

        # task_description: anvil_config.json first, YAML overrides if explicitly set
        self.task_description = meta.get("task_description", "")
        if model_cfg.get("task_description"):
            self.task_description = model_cfg["task_description"]


    @property
    def _uses_rtc_inference(self) -> bool:
        """True if the loaded model uses RTC background chunk inference."""
        return getattr(self, "_rtc_enabled", False)

    @property
    def _uses_sync_chunk_inference(self) -> bool:
        """True for non-RTC policies that return a complete action chunk."""
        return uses_sync_chunk_inference(getattr(self, "model_type", None))

    @property
    def _uses_sync_prefetch(self) -> bool:
        """True when a synchronous chunk policy has background prefetch enabled."""
        return (
            not self._uses_rtc_inference
            and self._uses_sync_chunk_inference
            and getattr(self, "_sync_prefetch_enabled", False)
        )

    @property
    def _is_language_conditioned(self) -> bool:
        """True if the loaded model consumes a natural-language task prompt."""
        return is_language_conditioned(getattr(self, "model_type", None))

    def _load_yaml_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        if not config_file:
            self.get_logger().warn("No config_file specified, using defaults")
            return {}

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _read_checkpoint_metadata(self) -> dict:
        """
        Read checkpoint metadata from config.json and anvil_config.json.
        Lightweight — JSON only, no tensor loading.
        Raises RuntimeError if model_path is set but config.json is missing/unreadable.
        """
        if not self.model_path:
            return {}

        checkpoint = Path(self.model_path)

        # Auto-detect pretrained_model subdirectory (mirrors ModelLoader logic)
        pretrained = checkpoint / "pretrained_model"
        if pretrained.exists() and (pretrained / "config.json").exists():
            checkpoint = pretrained

        # Auto-detect HF cache snapshot structure (blobs/ + snapshots/)
        if not (checkpoint / "config.json").exists():
            snapshots = checkpoint / "snapshots"
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if (snap / "config.json").exists():
                        checkpoint = snap
                        break

        # config.json — required
        config_path = checkpoint / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"config.json not found in {checkpoint}")
        cfg = json.loads(config_path.read_text())

        # image shape from input_features (first VISUAL entry)
        image_shape = None
        for feat in cfg.get("input_features", {}).values():
            if feat.get("type") == "VISUAL":
                c, h, w = feat["shape"]   # stored as [C, H, W]
                image_shape = (h, w, c)   # return as (H, W, C) for cv2
                break
        if image_shape is None:
            raise RuntimeError(f"No VISUAL input feature found in {config_path}")

        # Update model_path to resolved checkpoint (for ModelLoader)
        self.model_path = str(checkpoint)

        meta = {
            "image_shape": image_shape,
            "model_type":  cfg.get("type"),
        }

        # anvil_config.json — optional (absent for checkpoints pre-anvil_config)
        anvil_path = checkpoint / "anvil_config.json"
        if anvil_path.exists():
            anvil = json.loads(anvil_path.read_text())
            meta["action_type"] = anvil.get("action_type", "absolute")
            meta["use_delta_actions"] = anvil.get("use_delta_actions", False)
            meta["delta_exclude_joints"] = anvil.get("delta_exclude_joints", [])
            if "task_description" in anvil:
                meta["task_description"] = anvil["task_description"]
        return meta

    def _create_strategy(self):
        """Create multi-process inference strategy."""
        from .strategies.multi_process import MultiProcessStrategy

        return MultiProcessStrategy()

    def _setup_model(self) -> None:
        """Load model weights and processors. All config fields must be set by _setup_config()."""
        if self.get_parameter("deterministic").value:
            seed = self.get_parameter("deterministic_seed").value
            set_deterministic_mode(seed)
            self.get_logger().info(f"Deterministic mode enabled with seed={seed}")
        if self.model_type == "vla_jepa" and self._uses_rtc_inference and self.use_delta_actions:
            raise ValueError(
                "VLA-JEPA RTC currently requires an absolute-action checkpoint; "
                f"checkpoint action_type is '{self.action_type}'"
            )

        # Resolve inference tuning per model type
        tuning = self._tuning_config
        config_overrides = {}

        self._sync_config_yaml = tuning.get("sync", {})
        self._sync_prefetch_enabled = bool(self._sync_config_yaml.get("async_prefetch", False))

        if self._uses_rtc_inference:
            self.rtc_config_yaml = tuning.get("rtc", {})
        elif self.model_type == "diffusion":
            diff = tuning.get("diffusion", {})
            if diff.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = diff["n_action_steps"]
            if diff.get("num_inference_steps") is not None:
                config_overrides["num_inference_steps"] = diff["num_inference_steps"]
        elif self.model_type == "act":
            act = tuning.get("act", {})
            if act.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = act["n_action_steps"]
            if act.get("temporal_ensemble_coeff") is not None:
                config_overrides["temporal_ensemble_coeff"] = act["temporal_ensemble_coeff"]
                if act.get("n_action_steps") is None or act["n_action_steps"] > 1:
                    self.get_logger().warn(
                        "temporal_ensemble requires n_action_steps=1, forcing override"
                    )
                    config_overrides["n_action_steps"] = 1
        else:
            sync = self._sync_config_yaml
            if sync.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = sync["n_action_steps"]

        # Fallback: also check old top-level rtc key for backward compatibility
        if self._uses_rtc_inference and not self.rtc_config_yaml:
            self.rtc_config_yaml = self.config.get("rtc", {})

        self.n_action_steps_override = config_overrides.get("n_action_steps")

        loader = ModelLoader(
            self.model_path,
            self.device,
            self.model_type,
            config_overrides=config_overrides,
            logger=self.get_logger(),
            rtc_config_yaml=getattr(self, "rtc_config_yaml", {}),
            rtc_enabled=self._uses_rtc_inference,
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self._loader = loader

        # Confirm final model_type (ModelLoader auto-detects if None was passed)
        self.model_type = loader.model_type

        # RTC policies: set up ActionQueue and start background inference thread
        if self._uses_rtc_inference:
            self._setup_rtc_inference()
            self._start_inference_thread()
        else:
            # Non-RTC policies share the lightweight latency tracker. Chunked
            # foundation policies may additionally run their forward pass in a
            # background thread so action publication never blocks on inference.
            from lerobot_control.latency_stats import LatencyStats

            self._latency_tracker = LatencyStats(maxlen=100)
            if self._uses_sync_prefetch:
                self._setup_sync_prefetch()
                self._start_sync_prefetch_thread()

        if self._is_language_conditioned and not self.task_description:
            self.get_logger().warn(
                f"{self.model_type} has no task_description - re-train with --task-description "
                "or set model.task_description in the inference YAML."
            )

    def _log_startup(self) -> None:
        """Log unified startup summary after all setup is complete."""
        logger = self.get_logger()
        logger.info("=" * 50)
        logger.info("LeRobot Inference Node")
        logger.info("=" * 50)
        if self.echo_topic_only:
            logger.info("Mode:       Monitor Only (no model, no publishing)")
        else:
            logger.info(f"Model:      {self.model_path}")
            logger.info(f"Type:       {self.model_type or 'unknown'}")
            logger.info(f"Action type: {self.action_type}")
            if self.use_delta_actions and self.delta_exclude_joints:
                logger.info(f"Delta excl: {self.delta_exclude_joints}")
            if self._is_language_conditioned:
                logger.info(f"Task:       '{self.task_description}'")
        logger.info(f"Device:     {self.device}")
        logger.info(f"Frequency:  {self.control_freq} Hz")
        if not self.echo_topic_only:
            max_delta = (
                f"{self.max_position_delta} rad"
                if self.max_position_delta is not None
                else "disabled (delegated to robot controller)"
            )
            logger.info(f"Max delta:  {max_delta}")

        h, w, _ = self.image_shape
        res_note = "auto-detected from checkpoint" if self.model_path else "default"
        logger.info(f"Resolution: {w}x{h}  ({res_note})")

        logger.info(f"Cameras:    {self.camera_names}")
        logger.info(f"Arms:       {list(self.arms_config.keys())}")

        if not self.echo_topic_only and hasattr(self, "model") and hasattr(self.model, "config"):
            config = self.model.config
            chunk_size = getattr(config, "chunk_size", None)
            n_action_steps = getattr(config, "n_action_steps", None)
            cs = str(chunk_size) if chunk_size is not None else "N/A"
            nas = str(n_action_steps) if n_action_steps is not None else "N/A"

            logger.info("┌─ Inference tuning ──────────────────────────────────────┐")
            logger.info(f"│  chunk_size      = {cs:<4} (fixed at training, read-only)   │")
            logger.info(f"│  n_action_steps  = {nas:<4} (override in inference_tuning:)  │")
            logger.info( "│    → jittery / oscillating?  raise n_action_steps       │")
            logger.info( "│    → hesitates / freezes?    lower n_action_steps       │")
            logger.info( "└─────────────────────────────────────────────────────────┘")

            orig = getattr(self._loader, "checkpoint_n_action_steps", None)
            if (
                orig is not None
                and n_action_steps is not None
                and orig != n_action_steps
                and self.n_action_steps_override is not None
            ):
                logger.info(f"  (overridden from checkpoint default: {orig} → {n_action_steps})")

            if getattr(config, "temporal_ensemble_coeff", None) is not None:
                if hasattr(self.model, "temporal_ensembler"):
                    logger.info("Temporal ensembler initialized successfully")
                else:
                    logger.error("temporal_ensemble_coeff is set but ensembler not created!")

        # GPU/CPU memory after model load
        if not self.echo_topic_only and hasattr(self, "model"):
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated(self.device) / 1e6
                logger.info(f"GPU memory (weights): {gpu_mb:.0f} MB")
            try:
                import psutil

                cpu_mb = psutil.Process().memory_info().rss / 1e6
                logger.info(f"CPU RSS after load:   {cpu_mb:.0f} MB")
            except ImportError:
                pass

        if not self.echo_topic_only and self._uses_rtc_inference:
            rtc = self.rtc_config_yaml
            logger.info("┌─ RTC ───────────────────────────────────────────────────┐")
            logger.info("│  Status:              ENABLED                           │")
            logger.info(f"│  execution_horizon  = {rtc.get('execution_horizon', 10):<4}                             │")
            logger.info(f"│  max_guidance_weight= {rtc.get('max_guidance_weight', 10.0):<6}                           │")
            logger.info(f"│  attention_schedule = {rtc.get('prefix_attention_schedule', 'EXP'):<6}                           │")
            logger.info(f"│  queue_threshold    = {rtc.get('queue_trigger_threshold', 30):<4}                             │")
            logger.info("└─────────────────────────────────────────────────────────┘")

        if not self.echo_topic_only and self._uses_sync_prefetch:
            logger.info("┌─ Sync chunk prefetch ────────────────────────────────────┐")
            logger.info("│  Status:              ENABLED                           │")
            logger.info(f"│  refill threshold   = {self._sync_prefetch_threshold:<4} actions                     │")
            logger.info(f"│  replace queued tail= {str(self._sync_replace_pending):<5}                            │")
            logger.info("│  forward pass runs off the control/publish timers       │")
            logger.info("└─────────────────────────────────────────────────────────┘")

    def _setup_publishers(self) -> None:
        """Setup action publishers."""
        self.arm_publishers: dict[str, rclpy.publisher.Publisher] = {}
        for arm_name, arm_config in self.arms_config.items():
            cmd_topic = arm_config.get(
                "command_topic",
                f"/{arm_name}_forward_position_controller/commands",
            )
            self.arm_publishers[arm_name] = self.create_publisher(Float64MultiArray, cmd_topic, 10)
            self.get_logger().info(f"Publishing to: {cmd_topic}")

        if self._monitor_enable:
            self._monitor_obs_pub = self.create_publisher(Float64MultiArray, "/monitor/obs_state", 10)
            self._monitor_raw_pub = self.create_publisher(Float64MultiArray, "/monitor/raw_output", 10)
            self._monitor_cmd_pub = self.create_publisher(Float64MultiArray, "/monitor/control_cmd", 10)
            self.get_logger().info("Monitor topics enabled: /monitor/{obs_state,raw_output,control_cmd}")

    def _setup_rtc_inference(self) -> None:
        """Initialise ActionQueue and LatencyTracker for RTC mode."""
        from lerobot.policies.rtc.action_queue import ActionQueue

        from lerobot_control.latency_stats import LatencyStats

        self._action_queue = ActionQueue(self.model.config.rtc_config)
        self._latency_tracker = LatencyStats(maxlen=100)
        self._rtc_acquire_latency = LatencyStats(maxlen=100)
        self._rtc_preprocess_latency = LatencyStats(maxlen=100)
        self._rtc_model_latency = LatencyStats(maxlen=100)
        self._inference_stop = threading.Event()
        self._rtc_threshold = self.rtc_config_yaml.get("queue_trigger_threshold", 30)
        self._rtc_delay_fallback = self.rtc_config_yaml.get("inference_delay", 4)
        self._rtc_allow_latency_overrun = bool(
            self.rtc_config_yaml.get("allow_latency_overrun", False)
        )
        self._rtc_starvation_warned = False

    def _start_inference_thread(self) -> None:
        """Start the background RTC inference daemon thread."""
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="rtc-inference",
            daemon=True,
        )
        self._inference_thread.start()

    def _inference_loop(self) -> None:
        """Background inference thread for RTC mode.

        Continuously predicts the next action chunk whenever ActionQueue depth
        falls to or below the trigger threshold. Postprocessing happens here
        (before merge) so that control_loop can publish directly from the queue
        without any further processing.
        """
        while not self._inference_stop.is_set():
            # Wait until queue is low enough to warrant a new inference
            if self._action_queue.qsize() > self._rtc_threshold:
                time.sleep(0.005)
                continue

            # Acquire and materialize exactly one newest observation per prediction.
            # Reading shared images here avoids copying and converting three full
            # resolution frames on every 30 Hz observation timer tick.
            t0 = time.monotonic()
            raw_obs = self.strategy.get_observation(self.camera_names)
            acquire_elapsed = time.monotonic() - t0
            if raw_obs is None:
                time.sleep(0.005)
                continue

            # Snapshot queue state before inference for delay validation
            idx_before = self._action_queue.get_action_index()
            prev_actions = self._action_queue.get_left_over()

            # Compute inference delay from latency history
            max_lat = self._latency_tracker.max()
            inference_delay = (
                math.ceil(max_lat * self.control_freq) if max_lat else self._rtc_delay_fallback
            )

            # Run inference — do NOT use torch.inference_mode():
            # RTCProcessor calls torch.enable_grad() internally for guidance gradients.
            # inference_mode() cannot be overridden and would silently zero all gradients.
            try:
                preprocess_start = time.monotonic()
                obs = self._preprocess_policy_observation(raw_obs)
                preprocess_elapsed = time.monotonic() - preprocess_start
                model_start = time.monotonic()
                raw = self.model.predict_action_chunk(
                    obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=self.model.config.rtc_config.execution_horizon,
                )
                model_elapsed = time.monotonic() - model_start
            except Exception as e:
                import traceback
                self.get_logger().error(f"[RTC] predict_action_chunk failed: {e}")
                self.get_logger().error(traceback.format_exc())
                time.sleep(0.005)
                continue

            elapsed = time.monotonic() - t0
            self._latency_tracker.add(elapsed)
            self._rtc_acquire_latency.add(acquire_elapsed)
            self._rtc_preprocess_latency.add(preprocess_elapsed)
            self._rtc_model_latency.add(model_elapsed)
            new_delay = math.ceil(elapsed * self.control_freq)

            # Postprocess in inference thread (official pattern from eval_with_real_robot.py):
            #   original = raw (for RTC guidance of the next chunk)
            #   processed = denormalized (ready for the robot)
            original = raw.squeeze(0).clone()
            chunk_steps = len(original)
            merge_delay = new_delay
            if new_delay >= chunk_steps:
                if self._rtc_allow_latency_overrun:
                    merge_delay = max(chunk_steps - 1, 0)
                    if not self._rtc_starvation_warned:
                        self.get_logger().warning(
                            "[RTC] Latency overrun override active: "
                            f"clamping delay {new_delay} -> {merge_delay} "
                            "to execute the final available action"
                        )
                elif not self._rtc_starvation_warned:
                    self.get_logger().error(
                        "[RTC] Inference latency consumed the complete action chunk "
                        f"(delay={new_delay}, chunk={chunk_steps}); holding the last command"
                    )
                self._rtc_starvation_warned = True
            else:
                self._rtc_starvation_warned = False

            if self.postprocessor:
                processed = self.postprocessor.process_action(raw.squeeze(0))
            else:
                processed = original

            self._action_queue.merge(original, processed, merge_delay, idx_before)
            self.metrics.record_inference()

    def _setup_sync_prefetch(self) -> None:
        """Initialise background prefetch for non-RTC chunk policies.

        These policies expose ``predict_action_chunk`` but their stock
        ``select_action`` implementation does inference only after its private
        queue is empty. Running that call from the observation timer guarantees
        a publication gap whenever inference latency exceeds one control period.
        The prefetch worker instead fills the node-owned action deque while the
        publish timer continues consuming the previous chunk.
        """
        n_action_steps = int(
            getattr(self.model.config, "n_action_steps", None)
            or getattr(self.model.config, "chunk_size", 1)
        )
        configured_threshold = int(
            self._sync_config_yaml.get("prefetch_threshold", n_action_steps)
        )
        if configured_threshold < 0:
            raise ValueError("inference_tuning.sync.prefetch_threshold must be >= 0")

        self._sync_prefetch_threshold = min(configured_threshold, n_action_steps)
        self._sync_replace_pending = bool(
            self._sync_config_yaml.get("replace_pending_actions", True)
        )
        if configured_threshold > n_action_steps:
            self.get_logger().info(
                "Sync prefetch threshold clamped to n_action_steps: "
                f"{configured_threshold} -> {n_action_steps}"
            )

        self._sync_obs_lock = threading.Lock()
        self._sync_model_lock = threading.Lock()
        self._sync_generation = 0
        self._inference_stop = threading.Event()

    def _start_sync_prefetch_thread(self) -> None:
        """Start the background worker for synchronous chunk policies."""
        self._inference_thread = threading.Thread(
            target=self._sync_prefetch_loop,
            name="sync-chunk-prefetch",
            daemon=True,
        )
        self._inference_thread.start()

    def _sync_prefetch_loop(self) -> None:
        """Predict and enqueue complete chunks ahead of the publish cursor."""
        while not self._inference_stop.is_set():
            with self._classic_action_lock:
                queue_depth = len(self._classic_action_deque)
            if queue_depth > self._sync_prefetch_threshold:
                time.sleep(0.005)
                continue

            # Materialize one newest observation per chunk instead of copying and
            # converting all full-resolution frames on every 30 Hz timer tick.
            with self._sync_obs_lock:
                generation = self._sync_generation
            raw_obs = self.strategy.get_observation(self.camera_names)
            if raw_obs is None:
                time.sleep(0.005)
                continue

            try:
                observation = self._preprocess_policy_observation(raw_obs)
                t0 = time.monotonic()
                with self._sync_model_lock, torch.inference_mode():
                    chunk = self.model.predict_action_chunk(observation)
                elapsed = time.monotonic() - t0
                self._latency_tracker.add(elapsed)

                if isinstance(chunk, torch.Tensor) and chunk.dim() == 3:
                    chunk = chunk.squeeze(0)
                if self.postprocessor:
                    chunk = self.postprocessor.process_action(chunk)
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float64)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]

                n_action_steps = int(
                    getattr(self.model.config, "n_action_steps", None) or len(chunk)
                )
                chunk = chunk[:n_action_steps]
                if len(chunk) == 0:
                    raise RuntimeError("predict_action_chunk returned no executable actions")

                if self.use_delta_actions:
                    state = raw_obs.get("observation.state")
                    if state is None:
                        raise RuntimeError(
                            f"{self.action_type} requires observation.state for sync prefetch"
                        )
                    if isinstance(state, torch.Tensor):
                        state = state.detach().cpu().numpy()
                    state = np.asarray(state, dtype=np.float64).reshape(-1)
                    chunk = restore_delta_chunk(
                        chunk,
                        state,
                        self.action_type,
                        self._delta_exclude_indices,
                    )

                # reset_policy may have invalidated this in-flight prediction.
                with self._sync_obs_lock:
                    if generation != self._sync_generation:
                        continue
                with self._classic_action_lock:
                    if self._sync_replace_pending:
                        self._sync_replaced_actions += len(self._classic_action_deque)
                        self._classic_action_deque.clear()
                    self._classic_action_deque.extend(np.copy(action) for action in chunk)
                self.metrics.record_inference()
            except Exception as e:
                import traceback

                self.get_logger().error(f"[sync-prefetch] predict_action_chunk failed: {e}")
                self.get_logger().error(traceback.format_exc())
                time.sleep(0.01)

    def _preprocess_policy_observation(self, observation: dict) -> dict:
        """Preprocess a raw observation, adding a task prompt when required.

        LeRobot processor pipelines expect a flat batch dict. Language-conditioned
        policies consume a top-level "task" list, which the processor routes into
        the policy-specific language/token fields.
        """
        batch = dict(observation)
        if self._is_language_conditioned and self.task_description:
            batch["task"] = [self.task_description]
        if self.preprocessor:
            batch = self.preprocessor(batch)
        return self._move_to_device(batch)

    def _obs_update(self) -> None:
        """Observation update timer (unified for all models).

        RTC observations are acquired directly by the background inference thread.
        Synchronous policies: preprocess, run select_action, push result to deque.
        """
        if self._shutting_down:
            return
        if self._uses_rtc_inference:
            return
        if self._uses_sync_prefetch:
            return
        observation = self.strategy.get_observation(self.camera_names)
        if observation is None:
            return

        try:
            if not self._uses_sync_prefetch:
                # Keep a reference to the raw (unnormalised) observation so we can
                # capture the joint-state baseline when a new chunk is generated.
                _raw_obs = observation

                # True when the model will actually run a forward pass this tick.
                # ACT uses self._action_queue; Diffusion uses self._queues["action"].
                if hasattr(self.model, "_action_queue"):
                    _will_run_forward = len(self.model._action_queue) == 0
                elif hasattr(self.model, "_queues") and self.model._queues is not None:
                    action_q = self.model._queues.get("action")
                    _will_run_forward = action_q is None or len(action_q) == 0
                else:
                    _will_run_forward = True

                # Detect whether a new action chunk is about to be generated.
                # When the queue is empty, select_action will run the model and fill
                # it with n_action_steps new predictions, all computed relative to
                # the current state.  We capture that state as the delta reference.
                _is_new_chunk = self.use_delta_actions and _will_run_forward

                observation = self._preprocess_policy_observation(observation)

                # [DEBUG] Point 1: obs.state after preprocessor (check normalization)
                if self._debug and _is_new_chunk and "observation.state" in observation:
                    _dbg_s = observation["observation.state"]
                    if isinstance(_dbg_s, torch.Tensor):
                        _dbg_s = _dbg_s.cpu().numpy()
                    _dbg_s = np.asarray(_dbg_s).flatten()
                    self.get_logger().info(
                        f"[DEBUG] obs.state (post-preproc): [{', '.join(f'{v:.4f}' for v in _dbg_s)}]"
                    )

                with torch.inference_mode():
                    if _will_run_forward:
                        _t0 = time.monotonic()
                    action = self.model.select_action(observation)
                    if _will_run_forward:
                        self._latency_tracker.add(time.monotonic() - _t0)
                    # Collect remaining normalized queue items BEFORE postprocessing so
                    # the whole chunk can be denormalized together for delta restore.
                    if _is_new_chunk and self.use_delta_actions and hasattr(self.model, "_queues"):
                        _rest_norm = [a.detach().clone() for a in self.model._queues.get("action", [])]
                    else:
                        _rest_norm = None

                # Capture reference state right after chunk generation
                if _is_new_chunk and "observation.state" in _raw_obs:
                    _s = _raw_obs["observation.state"]
                    if hasattr(_s, "numpy"):
                        _s = (_s.squeeze(0).numpy() if _s.dim() > 1 else _s.numpy())
                    elif hasattr(_s, "cpu"):
                        _s = _s.cpu().numpy()
                    self._delta_ref_state = np.asarray(_s, dtype=np.float64).flatten()

                if self.postprocessor:
                    action = self.postprocessor.process_action(action)

                if isinstance(action, torch.Tensor):
                    if action.dim() > 1:
                        action = action.squeeze(0)
                    action = action.cpu().numpy()

                # [DEBUG] Point 3: action after postprocessor, before delta restore
                if self._debug and _is_new_chunk:
                    self.get_logger().info(
                        f"[DEBUG] action (post-postproc): [{', '.join(f'{v:.4f}' for v in action)}]"
                    )

                # Chunk-level delta restore via shadow queue.
                # The model's internal queue stores normalized tensors; we denormalize
                # the full chunk together, restore delta → absolute, then serve absolute
                # values from a shadow queue so we never re-enter normalized space.
                if self.use_delta_actions:
                    if _is_new_chunk and self._delta_ref_state is not None:
                        if _rest_norm is not None:
                            _rest_denorm = [self._denorm_queue_action(a) for a in _rest_norm]
                            _chunk = np.stack([action] + _rest_denorm) if _rest_denorm else action[np.newaxis]
                        else:
                            _chunk = action[np.newaxis]
                        _abs = restore_delta_chunk(_chunk, self._delta_ref_state, self.action_type, self._delta_exclude_indices)
                        self._abs_shadow_queue = deque(_abs[1:])
                        action = _abs[0]
                    elif self._abs_shadow_queue:
                        action = self._abs_shadow_queue.popleft()
                    elif not hasattr(self.model, "_queues") and self._delta_ref_state is not None:
                        action = restore_delta_chunk(action[np.newaxis], self._delta_ref_state, self.action_type, self._delta_exclude_indices)[0]

                with self._classic_action_lock:
                    self._classic_action_deque.append(action)
                if _will_run_forward:
                    self.metrics.record_inference()

        except Exception as e:
            import traceback
            self.get_logger().error(f"Observation/inference error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _publish_loop(self) -> None:
        """Action publish timer (unified for all models).

        RTC: pop from ActionQueue (filled by background inference thread).
        Synchronous policies: pop from deque (filled by _obs_update).
        """
        if self._shutting_down:
            return
        self.metrics.record_control_loop()

        if self._uses_rtc_inference:
            action = self._action_queue.get()
            if self._debug:
                self._queue_depths.append(self._action_queue.qsize())
            if action is None:
                self._rtc_skip_count += 1
                return
            if isinstance(action, torch.Tensor):
                if action.dim() > 1:
                    action = action.squeeze(0)
                action = action.cpu().numpy()
        else:
            with self._classic_action_lock:
                if not self._classic_action_deque:
                    if self._uses_sync_prefetch:
                        self._sync_skip_count += 1
                        if self._debug:
                            self._queue_depths.append(0)
                    return
                action = self._classic_action_deque.popleft()
                if self._uses_sync_prefetch and self._debug:
                    self._queue_depths.append(len(self._classic_action_deque))

        try:
            self._publish_action(action)
        except Exception as e:
            import traceback
            self.get_logger().error(f"Publish error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _reset_delta_state(self) -> None:
        """Reset delta-restore state; call this whenever the model is reloaded."""
        self._delta_ref_state = None
        self._abs_shadow_queue.clear()

    def _denorm_queue_action(self, a: object) -> np.ndarray:
        """Apply postprocessor and convert a queued normalized tensor to a flat numpy array."""
        if self.postprocessor:
            a = self.postprocessor.process_action(a)  # type: ignore[arg-type]
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().flatten()
        return np.asarray(a).flatten()

    def _move_to_device(self, data):
        """Recursively move tensors to the configured device."""
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {key: self._move_to_device(value) for key, value in data.items()}
        if isinstance(data, tuple):
            return tuple(self._move_to_device(value) for value in data)
        if isinstance(data, list):
            return [self._move_to_device(value) for value in data]
        return data

    def _publish_action(self, action: np.ndarray) -> None:
        """Publish action to arm controllers."""
        current_positions = self.strategy.get_current_joint_positions()
        joint_order = self.joint_names_config.get(
            "controller_joint_order",
            self.joint_names_config.get("joint_order", []),
        )

        monitor_obs_parts: list[np.ndarray] = []
        monitor_cmd_parts: list[np.ndarray] = []

        for arm_name, arm_config in self.arms_config.items():
            start_idx = arm_config.get("action_start", 0)
            end_idx = arm_config.get("action_end", len(action))
            ros_prefix = arm_config.get("ros_prefix", arm_name)

            arm_action = action[start_idx:end_idx].copy()

            arm_current = None
            if current_positions:
                arm_current = np.array(
                    [
                        current_positions.get(f"{ros_prefix}_{joint_order[i]}", 0.0)
                        for i in range(len(arm_action))
                    ]
                )

            # Delta restore is done upstream in _obs_update (chunk-level).
            # Actions arriving here are already absolute — just apply safety limits.
            arm_action = self.action_limiter.process(arm_action, arm_current)

            if self._debug:
                formatted = ", ".join(f"{v:.4f}" for v in arm_action)
                self.get_logger().info(f"[DEBUG] cmd [{arm_name}]: [{formatted}]")

            msg = Float64MultiArray()
            msg.data = arm_action.tolist()
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)

            if self._monitor_enable:
                if arm_current is not None:
                    monitor_obs_parts.append(arm_current)
                monitor_cmd_parts.append(arm_action)

        if self._monitor_enable and monitor_cmd_parts:
            self._publish_monitor(
                obs_state=np.concatenate(monitor_obs_parts) if monitor_obs_parts else np.zeros_like(action),
                raw_output=action,
                control_cmd=np.concatenate(monitor_cmd_parts),
            )

        # Debug: track smoothness
        if self._smooth_tracker is not None:
            self._smooth_tracker.record(action)
        self.metrics.record_action_output()
        self._has_published = True

    def _publish_monitor(
        self,
        obs_state: np.ndarray,
        raw_output: np.ndarray,
        control_cmd: np.ndarray,
    ) -> None:
        """Publish monitor topics for real-time inference visualization."""
        obs_msg = Float64MultiArray()
        obs_msg.data = obs_state.tolist()
        self._monitor_obs_pub.publish(obs_msg)

        raw_msg = Float64MultiArray()
        raw_msg.data = raw_output.tolist()
        self._monitor_raw_pub.publish(raw_msg)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = control_cmd.tolist()
        self._monitor_cmd_pub.publish(cmd_msg)

    def _log_input_stats(self) -> None:
        """Periodically log input reception statistics with windowed rates."""
        stats = self.metrics.get_stats()
        if stats["elapsed_sec"] < 1.0:
            return  # Wait for enough data

        # Get frame counters from shared memory workers
        frame_counters: dict[str, int] = self.strategy.get_frame_counters() or {}

        # Compute windowed rates (delta since last log)
        now = time.time()
        if self._prev_log_time is not None:
            dt = max(now - self._prev_log_time, 0.001)
        else:
            dt = stats["elapsed_sec"]

        joint_hz = (stats["joint_count"] - self._prev_joint_count) / dt
        control_hz = (stats["control_loop_count"] - self._prev_control_count) / dt
        inference_delta = stats["inference_count"] - self._prev_inference_count
        inference_hz = inference_delta / dt
        action_output_hz = (stats["action_output_count"] - self._prev_action_output_count) / dt

        camera_hz: dict[str, float] = {}
        camera_delta: dict[str, int] = {}
        for name, count in frame_counters.items():
            prev = self._prev_frame_counters.get(name, 0)
            camera_delta[name] = count - prev
            camera_hz[name] = camera_delta[name] / dt

        # Store snapshot for next window
        self._prev_log_time = now
        self._prev_joint_count = stats["joint_count"]
        self._prev_control_count = stats["control_loop_count"]
        self._prev_inference_count = stats["inference_count"]
        self._prev_action_output_count = stats["action_output_count"]
        self._prev_frame_counters = dict(frame_counters)

        # Find bottleneck camera: compare each camera against its own expected fps,
        # not control_freq (camera target rate is independent of the control loop).
        bottleneck_name = None
        if not self.echo_topic_only and camera_hz:
            slow_cameras = [
                (name, hz)
                for name, hz in camera_hz.items()
                if hz < self._expected_camera_fps.get(name, 30.0) * 2 / 3
            ]
            if slow_cameras:
                bottleneck_name = min(slow_cameras, key=lambda x: x[1])[0]

        # Common header: joint state + cameras
        logger = self.get_logger()
        logger.info(f"-- Stats ({dt:.0f}s) " + "-" * 30)
        logger.info(f"  Joint State  {joint_hz:7.1f} Hz")
        for name in sorted(camera_hz.keys()):
            hz = camera_hz[name]
            delta = camera_delta.get(name, 0)
            marker = "  << bottleneck" if name == bottleneck_name else ""
            logger.info(f"  {name:12s}  {hz:7.1f} Hz  (+{delta} frames){marker}")

        if not self.echo_topic_only:
            if self._uses_rtc_inference:
                self._log_stats_rtc(logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz)
            else:
                self._log_stats_classic(logger, dt, stats, control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz)

    def _log_stats_common(self, logger, inference_hz, action_output_hz, stats) -> None:
        """Log model-agnostic stats shared across all model types."""
        logger.info(f"  Inference FPS{inference_hz:7.1f} Hz  ({stats['inference_count']} total)")
        logger.info(f"  Action FPS   {action_output_hz:7.1f} Hz")
        if hasattr(self, "_latency_tracker"):
            lat_mean = self._latency_tracker.mean()
            lat_std = self._latency_tracker.std()
            lat_p95 = self._latency_tracker.p95() or 0.0
            if lat_mean > 0:
                logger.info(
                    f"  Infer latency mean={lat_mean * 1000:.1f}ms  "
                    f"std={lat_std * 1000:.1f}ms  p95={lat_p95 * 1000:.1f}ms"
                )

    def _log_stats_rtc(self, logger, _dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log RTC-specific stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

        # RTC policies additionally log queue size.
        if hasattr(self, "_latency_tracker"):
            lat_mean = self._latency_tracker.mean()
            if lat_mean > 0 and hasattr(self, "_action_queue"):
                queue_size = self._action_queue.qsize()
                logger.info(f"  RTC queue    {queue_size}")
                acquire_ms = self._rtc_acquire_latency.mean() * 1000
                preprocess_ms = self._rtc_preprocess_latency.mean() * 1000
                model_ms = self._rtc_model_latency.mean() * 1000
                logger.info(
                    f"  RTC stages   acquire={acquire_ms:.1f}ms  "
                    f"preprocess={preprocess_ms:.1f}ms  model={model_ms:.1f}ms"
                )

            # Debug: Action FPS, effective control Hz, queue depth stats, smoothness
            if self._debug and lat_mean > 0:
                cs = getattr(self.model.config, "chunk_size", 0)
                eh = getattr(self.model.config.rtc_config, "execution_horizon", 0)
                action_fps = cs / lat_mean
                eff_ctrl_hz = action_fps * eh / cs if cs > 0 else 0
                logger.info(f"  [DEBUG] Action FPS {action_fps:.1f}  Eff ctrl Hz {eff_ctrl_hz:.1f}")

        if self._debug and self._queue_depths:
            depths = np.array(self._queue_depths)
            skip_pct = self._rtc_skip_count / max(len(self._queue_depths) + self._rtc_skip_count, 1) * 100
            logger.info(f"  [DEBUG] Queue depth min={depths.min()} mean={depths.mean():.0f} max={depths.max()} skip={skip_pct:.1f}%")
            self._queue_depths.clear()
            self._rtc_skip_count = 0

        if self._debug and self._smooth_tracker is not None:
            smooth = self._smooth_tracker.get_stats()
            if smooth:
                logger.info(
                    f"  [DEBUG] Action D mean={smooth['delta_mean']:.4f} "
                    f"std={smooth['delta_std']:.4f} max={smooth['delta_max']:.4f} "
                    f"jerk={smooth['jerk_mean']:.4f}"
                )

        if bottleneck_name is not None:
            exp = self._expected_camera_fps.get(bottleneck_name, 30.0)
            logger.warn(
                f"  '{bottleneck_name}' is slow: {camera_hz[bottleneck_name]:.1f} Hz"
                f" (threshold: {exp * 2 / 3:.0f} Hz, expected: {exp:.0f} Hz)"
            )

    def _log_stats_classic(self, logger, _dt, stats, _control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log synchronous policy stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

        if self._uses_sync_prefetch and hasattr(self, "_latency_tracker"):
            latency = self._latency_tracker.mean()
            if latency > 0:
                n_action_steps = int(
                    getattr(self.model.config, "n_action_steps", None)
                    or getattr(self.model.config, "chunk_size", 0)
                )
                capacity_hz = n_action_steps / latency if n_action_steps else 0.0
                coverage_s = n_action_steps / self.control_freq if self.control_freq else 0.0
                margin_s = coverage_s - latency
                with self._classic_action_lock:
                    queue_size = len(self._classic_action_deque)
                logger.info(f"  Sync queue   {queue_size}")
                logger.info(
                    f"  Chunk supply {capacity_hz:.1f} action/s  "
                    f"coverage={coverage_s * 1000:.0f}ms  margin={margin_s * 1000:+.0f}ms"
                )

        if self._debug and self._uses_sync_prefetch and self._queue_depths:
            depths = np.array(self._queue_depths)
            total_ticks = len(self._queue_depths)
            skip_pct = self._sync_skip_count / max(total_ticks, 1) * 100
            logger.info(
                f"  [DEBUG] Queue depth min={depths.min()} mean={depths.mean():.0f} "
                f"max={depths.max()} starved={skip_pct:.1f}% "
                f"replaced={self._sync_replaced_actions}"
            )
            self._queue_depths.clear()
            self._sync_skip_count = 0
            self._sync_replaced_actions = 0

        if self._debug and self._smooth_tracker is not None:
            smooth = self._smooth_tracker.get_stats()
            if smooth:
                logger.info(
                    f"  [DEBUG] Action D mean={smooth['delta_mean']:.4f} "
                    f"std={smooth['delta_std']:.4f} max={smooth['delta_max']:.4f} "
                    f"jerk={smooth['jerk_mean']:.4f}"
                )

        if bottleneck_name is not None:
            exp = self._expected_camera_fps.get(bottleneck_name, 30.0)
            logger.warn(
                f"  '{bottleneck_name}' is slow: {camera_hz[bottleneck_name]:.1f} Hz"
                f" (threshold: {exp * 2 / 3:.0f} Hz, expected: {exp:.0f} Hz)"
            )

    def _on_max_run_elapsed(self) -> None:
        """Stop command publication and shut down when the episode limit expires."""
        if self._shutting_down:
            return

        self.get_logger().warning(
            f"Maximum run duration reached ({self.max_run_seconds:.1f}s); stopping inference"
        )
        self._shutting_down = True
        for timer_name in ("_obs_timer", "_publish_timer", "_stats_timer", "_max_run_timer"):
            timer = getattr(self, timer_name, None)
            if timer:
                timer.cancel()

        if self._has_published:
            self._publish_hold_position()
            self._has_published = False

        # Deliver SIGINT to the main thread so the existing KeyboardInterrupt/finally
        # path shuts down the executor, workers, shared memory, and ROS context cleanly.
        os.kill(os.getpid(), signal.SIGINT)


    def reset_policy(self) -> None:
        """Reset policy state."""
        if not hasattr(self, "model"):
            return
        self.get_logger().info("Resetting policy state...")
        if self._uses_sync_prefetch:
            with self._sync_obs_lock:
                self._sync_generation += 1
            with self._classic_action_lock:
                self._classic_action_deque.clear()
            with self._sync_model_lock:
                if hasattr(self.model, "reset"):
                    self.model.reset()
            self._reset_delta_state()
        elif hasattr(self.model, "reset"):
            self.model.reset()
        if self._uses_rtc_inference and hasattr(self, "_action_queue"):
            from lerobot.policies.rtc.action_queue import ActionQueue
            self._action_queue = ActionQueue(self.model.config.rtc_config)
        if hasattr(self, "_latency_tracker"):
            self._latency_tracker.reset()
        for tracker_name in (
            "_rtc_acquire_latency",
            "_rtc_preprocess_latency",
            "_rtc_model_latency",
        ):
            if hasattr(self, tracker_name):
                getattr(self, tracker_name).reset()
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def _publish_hold_position(self) -> None:
        """Publish current joint positions to hold the robot in place on shutdown."""
        if not hasattr(self, "arm_publishers"):
            return
        current = self.strategy.get_current_joint_positions()
        if not current:
            return
        joint_order = self.joint_names_config.get(
            "controller_joint_order",
            self.joint_names_config.get("joint_order", []),
        )
        for arm_name, arm_config in self.arms_config.items():
            ros_prefix = arm_config.get("ros_prefix", arm_name)
            start_idx = arm_config.get("action_start", 0)
            end_idx = arm_config.get("action_end", len(joint_order))
            arm_joints = joint_order[start_idx:end_idx]
            positions = [current.get(f"{ros_prefix}_{j}", 0.0) for j in arm_joints]
            msg = Float64MultiArray()
            msg.data = positions
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)
        self.get_logger().info("Shutdown: hold-position command sent to controllers")

    def destroy_node(self) -> None:
        """Cleanup timers, inference thread, strategy, and destroy node."""
        # Block any new publishes first — timers may still fire during executor shutdown
        self._shutting_down = True

        # Cancel timers before stopping the inference thread so no new callbacks
        # are scheduled while we wait for the thread to join.
        for timer_name in ("_obs_timer", "_publish_timer", "_stats_timer", "_max_run_timer"):
            timer = getattr(self, timer_name, None)
            if timer:
                timer.cancel()

        # Stop background RTC inference thread
        if hasattr(self, "_inference_stop"):
            self._inference_stop.set()
        if hasattr(self, "_inference_thread"):
            self._inference_thread.join(timeout=2.0)

        # Hold position before publisher is torn down — only if we actually
        # commanded the robot at least once during this session.
        if not self.echo_topic_only and self._has_published:
            self._publish_hold_position()

        self.strategy.cleanup()
        super().destroy_node()


def main(args=None):
    """Main entry point with single-threaded executor."""
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = LeRobotInferenceNode()

        # Use MultiThreadedExecutor: RTC mode needs 3+ threads
        # (obs timer, publish timer, stats timer, joint subscription)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        node.get_logger().info("Starting inference loop...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
