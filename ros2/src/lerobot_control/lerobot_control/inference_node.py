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
from .ee_runtime import (
    ee_delta_restore_step,
    ee_poses_from_chunk,
    ee_relative_restore_chunk,
    resolve_action_type,
)
from .metrics_tracker import MetricsTracker
from .model_loader import ModelLoader, set_deterministic_mode


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
            video_dir=self._monitor_video_dir if self._monitor_enable else None,
            mock_ee_pose_echo=self._mock_ee_pose_echo,
        )

        # Non-VLA action buffer (ACT/Diffusion put actions here from obs timer)
        self._classic_action_deque: deque = deque(maxlen=10)
        # Reference joint state captured at the moment each action chunk was
        # generated (in model/observation order).  All queued steps in the chunk
        # share this reference so the n-0 chunk-anchor restore is consistent
        # with training.
        # _relative_anchor_state and _abs_shadow_queue must always be reset together;
        # use _reset_relative_anchor_state() in any future reload or episode-boundary path.
        self._relative_anchor_state: np.ndarray | None = None
        self._abs_shadow_queue: deque[np.ndarray] = deque()

        # ee_delta only: freshest observed absolute EE pose (quat layout, 8*n_arms),
        # updated every _obs_update tick regardless of chunk timing. _publish_loop
        # reads this (under _obs_lock, cross-thread) to compose
        # absolute_target = obs_pose ∘ delta fresh at EVERY publish tick — the
        # decoupled delta-mode publish loop. Never restored to absolute in
        # _obs_update; that would defeat the whole point (see claude_docs/
        # ee-delta-flow-plan.md, Item 2).
        self._ee_delta_latest_obs_quat: np.ndarray | None = None
        # Per-arm CommandedEEPose.sequence backing _ee_delta_latest_obs_quat above,
        # captured alongside it (see MultiProcessStrategy.get_ee_obs_sequence_snapshot).
        # _publish_loop compares this against _ee_delta_last_published_seq to tell
        # whether the anchor has genuinely advanced since the last tick it composed
        # against — NOT whether the pose value looks different (a stationary arm's
        # pose is legitimately unchanged; only sequence non-advancement means the
        # mock/controller hasn't caught up to the last published command yet). If it
        # hasn't advanced, that tick is held rather than composed — composing a new
        # delta against an anchor the loop has already consumed once would silently
        # skip one row of the recorded trajectory and never recover (see
        # claude_docs/gt-replayer-correctness-test-plan.md's staleness investigation).
        self._ee_delta_latest_obs_seq: tuple[int, ...] | None = None
        self._ee_delta_last_published_seq: tuple[int, ...] | None = None
        # Created unconditionally: used by the ee_delta obs/publish handoff above
        # for ANY model type (classic or VLA), not just VLA's _latest_obs snapshot
        # (which reuses this same lock — see _setup_vla_inference).
        self._obs_lock = threading.Lock()

        # TEMP DEBUG (gt_replay ee_delta divergence investigation): generation
        # counter bumped every time _ee_delta_latest_obs_quat is refreshed, so
        # _publish_loop can detect/log when it reuses a stale (already-consumed)
        # anchor instead of a freshly-written one. Remove once root cause is fixed.
        self._ee_delta_obs_gen: int = 0
        self._ee_delta_last_published_gen: int | None = None

        self._shutting_down: bool = False
        self._has_published: bool = False

        if not self.echo_topic_only:
            self._setup_model()

            # ee_relative: raw absolute obs history for re-relativizing the full obs window.
            # n_obs_steps lives on model.config (not the module) in lerobot 0.5.1.
            if self.is_ee_relative and not self._is_vla:
                self._ee_n_obs_steps: int = int(
                    getattr(getattr(self.model, "config", None), "n_obs_steps", 2)
                )
                self._ee_raw_obs_buf: deque = deque(maxlen=self._ee_n_obs_steps)
            else:
                self._ee_n_obs_steps = 1
                self._ee_raw_obs_buf = deque(maxlen=1)

            # action_limiter is used for joint_abs mode only
            if not self.is_ee:
                self.action_limiter = ActionLimiter(
                    max_delta=self.max_position_delta,
                    min_delta_threshold=self.min_position_delta,
                    model_joint_order=self.joint_names_config.get("model_joint_order", []),
                    controller_joint_order=self.joint_names_config.get("controller_joint_order", []),
                    logger=self.get_logger(),
                )

            self._setup_publishers()

            # Unified split-timer architecture for all models:
            #   _obs_update:    preprocess (+ inference for non-VLA)
            #   _publish_loop:  pop action from queue/deque → publish
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
        self._vla_skip_count: int = 0
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
        self.declare_parameter("monitor_video_dir", "")
        # Fake-hardware-only: true iff /ee_pose_{arm} is the mock's MockEEPose
        # echo (sequence-guarded), never true for real hardware (plain
        # CommandedEEPose, no guard at all) — see MockEEPose.msg and
        # ee_obs_sequence_guard.py's module docstrings. Set by the deployment
        # (compose file/launch args), not the YAML config, since it's a
        # "which robot am I actually talking to" question, not a model/task
        # tunable — docker-compose.fake-hardware.yml sets this true wherever
        # mock-robot is co-launched; docker-compose.yml (real) leaves it false.
        self.declare_parameter("mock_ee_pose_echo", False)

        # Static fields from ROS2 params
        self.echo_topic_only = self.get_parameter("echo_topic_only").value
        self._debug = self.get_parameter("debug").value
        self._monitor_enable: bool = self.get_parameter("monitor_enable").value
        self._mock_ee_pose_echo: bool = self.get_parameter("mock_ee_pose_echo").value
        _debug_image_dir = self.get_parameter("debug_image_dir").value
        self._debug_image_dir: str | None = _debug_image_dir if _debug_image_dir else None
        _monitor_video_dir = self.get_parameter("monitor_video_dir").value
        self._monitor_video_dir: str | None = _monitor_video_dir if _monitor_video_dir else None
        self.model_path = self.get_parameter("model_path").value
        self._validate_required_params()

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

        # --- Run metadata (checkpoint or, for subclasses, a dataset) ---
        # Skip in echo_topic_only mode — no checkpoint needed
        meta = {} if self.echo_topic_only else self._load_run_metadata()

        # image_shape: from config.json input_features — must match training
        # Default (480, 640, 3) is used only in echo_topic_only mode with no checkpoint
        self.image_shape = meta.get("image_shape", (480, 640, 3))

        # model_type: from config.json, YAML overrides if explicitly set
        model_cfg = self.config.get("model", {})
        self.model_type = model_cfg.get("type") or meta.get("model_type")

        # action_type from anvil_config.json — must match training.
        # resolve_action_type() normalizes the legacy "ee_rel" alias to
        # "ee_relative", so everything below only ever compares against the
        # canonical value.
        self.action_type: str = resolve_action_type(meta)
        self.is_ee: bool = self.action_type in ("ee_abs", "ee_relative", "ee_delta")
        self.is_ee_relative: bool = self.action_type == "ee_relative"
        self.is_ee_abs: bool = self.action_type == "ee_abs"
        # ee_delta: Delta(n->n+1) — decoupled delta-mode publish loop (see
        # _publish_loop). Model output is a DELTA, not a pre-restored absolute;
        # absolute_target = obs_pose ∘ delta is composed fresh at EACH publish
        # tick against the freshest observed pose, not at chunk-generation time.
        self.is_ee_delta: bool = self.action_type == "ee_delta"
        # True only for ee_abs checkpoints trained with rot6d obs (obs_state_dim % 10 == 0).
        # Old ee_abs checkpoints use raw quat obs (obs_state_dim % 8 == 0) — no conversion needed.
        _obs_dim = meta.get("obs_state_dim")
        self.ee_abs_uses_rot6d_obs: bool = (
            self.is_ee_abs and _obs_dim is not None and _obs_dim % 10 == 0
        )

        # task_description: anvil_config.json first, YAML overrides if explicitly set
        self.task_description = meta.get("task_description", "")
        if model_cfg.get("task_description"):
            self.task_description = model_cfg["task_description"]


    @property
    def _is_vla(self) -> bool:
        """True if the loaded model is a VLA (pi0 / pi05 / smolvla)."""
        return getattr(self, "model_type", None) in {"smolvla", "pi0", "pi05"}

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

    def _validate_required_params(self) -> None:
        """Validate that required ROS2 params are set for this node's run source.

        Base implementation requires ``model_path`` (a checkpoint). Overridden by
        subclasses with a different run source — e.g. ``DatasetGtReplayerNode``
        requires ``dataset`` instead.
        """
        if not self.model_path and not self.echo_topic_only:
            raise ValueError("model_path parameter is required")

    def _load_run_metadata(self) -> dict:
        """
        Read checkpoint metadata from config.json and anvil_config.json.
        Lightweight — JSON only, no tensor loading.
        Raises RuntimeError if model_path is set but config.json is missing/unreadable.

        Overridden by subclasses that run from a source other than a checkpoint
        (e.g. ``DatasetGtReplayerNode`` derives the same meta-dict shape from a
        dataset's ``conversion_config.yaml`` / ``meta/info.json``).
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
        obs_state_dim = None
        for key, feat in cfg.get("input_features", {}).items():
            if feat.get("type") == "VISUAL" and image_shape is None:
                c, h, w = feat["shape"]   # stored as [C, H, W]
                image_shape = (h, w, c)   # return as (H, W, C) for cv2
            if key == "observation.state":
                shape = feat.get("shape", [])
                obs_state_dim = shape[0] if shape else None
        if image_shape is None:
            raise RuntimeError(f"No VISUAL input feature found in {config_path}")

        # Update model_path to resolved checkpoint (for ModelLoader)
        self.model_path = str(checkpoint)

        meta = {
            "image_shape":    image_shape,
            "model_type":     cfg.get("type"),
            "obs_state_dim":  obs_state_dim,
        }

        # anvil_config.json — optional (absent for checkpoints pre-anvil_config)
        anvil_path = checkpoint / "anvil_config.json"
        if anvil_path.exists():
            anvil = json.loads(anvil_path.read_text())
            meta["action_type"] = anvil.get("action_type", "joint_abs")
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

        # Resolve inference tuning per model type
        tuning = self._tuning_config
        config_overrides = {}

        if self._is_vla:
            self.rtc_config_yaml = tuning.get("rtc", {})
        elif self.model_type == "diffusion":
            diff = tuning.get("diffusion", {})
            if diff.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = diff["n_action_steps"]
            if diff.get("num_inference_steps") is not None:
                config_overrides["num_inference_steps"] = diff["num_inference_steps"]
        else:  # ACT and others
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

        # Fallback: also check old top-level rtc key for backward compatibility
        if self._is_vla and not self.rtc_config_yaml:
            self.rtc_config_yaml = self.config.get("rtc", {})

        self.n_action_steps_override = config_overrides.get("n_action_steps")

        loader = ModelLoader(
            self.model_path,
            self.device,
            self.model_type,
            config_overrides=config_overrides,
            logger=self.get_logger(),
            rtc_config_yaml=getattr(self, "rtc_config_yaml", {}),
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self._loader = loader

        # Confirm final model_type (ModelLoader auto-detects if None was passed)
        self.model_type = loader.model_type

        # VLA models: set up ActionQueue and start background inference thread
        if self._is_vla:
            self._setup_vla_inference()
            self._start_inference_thread()
        else:
            # Classic (ACT/Diffusion): initialise latency tracker
            from lerobot_control.latency_stats import LatencyStats

            self._latency_tracker = LatencyStats(maxlen=100)

        if self.model_type in {"smolvla", "pi0", "pi05"} and not self.task_description:
            self.get_logger().warn(
                f"{self.model_type} has no task_description — re-train with --task-description "
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
            if self.model_type in {"smolvla", "pi0", "pi05"}:
                logger.info(f"Task:       '{self.task_description}'")
        logger.info(f"Device:     {self.device}")
        logger.info(f"Frequency:  {self.control_freq} Hz")
        if not self.echo_topic_only:
            logger.info(f"Max delta:  {self.max_position_delta} rad")

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

        if not self.echo_topic_only and self._is_vla:
            rtc = self.rtc_config_yaml
            logger.info("┌─ RTC ───────────────────────────────────────────────────┐")
            logger.info("│  Status:              ENABLED                           │")
            logger.info(f"│  execution_horizon  = {rtc.get('execution_horizon', 10):<4}                             │")
            logger.info(f"│  max_guidance_weight= {rtc.get('max_guidance_weight', 10.0):<6}                           │")
            logger.info(f"│  attention_schedule = {rtc.get('prefix_attention_schedule', 'EXP'):<6}                           │")
            logger.info(f"│  queue_threshold    = {rtc.get('queue_trigger_threshold', 30):<4}                             │")
            logger.info("└─────────────────────────────────────────────────────────┘")

    def _setup_publishers(self) -> None:
        """Setup action publishers.

        EE mode (ee_abs / ee_relative): one ``CommandedEEPose`` publisher per arm,
        topic from ``arm_config["ee_command_topic"]`` (defaults to
        ``/commanded_ee_{arm_name}``).

        Joint mode (joint_abs): one ``Float64MultiArray`` publisher per arm,
        topic from ``arm_config["command_topic"]`` (defaults to
        ``/{arm_name}_forward_position_controller/commands``).
        """
        self.arm_publishers: dict[str, rclpy.publisher.Publisher] = {}

        if self.is_ee:
            from anvil_msgs.msg import CommandedEEPose
            for arm_name, arm_config in self.arms_config.items():
                ee_topic = arm_config.get("ee_command_topic", f"/commanded_ee_{arm_name}")
                self.arm_publishers[arm_name] = self.create_publisher(
                    CommandedEEPose, ee_topic, 10
                )
                self.get_logger().info(f"Publishing EE commands to: {ee_topic}")
        else:
            for arm_name, arm_config in self.arms_config.items():
                cmd_topic = arm_config.get(
                    "command_topic",
                    f"/{arm_name}_forward_position_controller/commands",
                )
                self.arm_publishers[arm_name] = self.create_publisher(
                    Float64MultiArray, cmd_topic, 10
                )
                self.get_logger().info(f"Publishing to: {cmd_topic}")

        if self._monitor_enable:
            self._monitor_obs_pub = self.create_publisher(Float64MultiArray, "/monitor/obs_state", 10)
            self._monitor_raw_pub = self.create_publisher(Float64MultiArray, "/monitor/raw_output", 10)
            self._monitor_cmd_pub = self.create_publisher(Float64MultiArray, "/monitor/control_cmd", 10)
            self.get_logger().info("Monitor topics enabled: /monitor/{obs_state,raw_output,control_cmd}")

    def _setup_vla_inference(self) -> None:
        """Initialise ActionQueue and LatencyTracker for VLA / RTC mode."""
        from lerobot.policies.rtc.action_queue import ActionQueue

        from lerobot_control.latency_stats import LatencyStats

        self._action_queue = ActionQueue(self.model.config.rtc_config)
        self._latency_tracker = LatencyStats(maxlen=100)
        self._latest_obs = None
        # _obs_lock is created unconditionally in __init__ (also guards the
        # ee_delta obs/publish handoff); reused here for VLA's _latest_obs.
        self._inference_stop = threading.Event()
        self._rtc_threshold = self.rtc_config_yaml.get("queue_trigger_threshold", 30)
        self._rtc_delay_fallback = self.rtc_config_yaml.get("inference_delay", 4)

    def _start_inference_thread(self) -> None:
        """Start the background RTC inference daemon thread."""
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="rtc-inference",
            daemon=True,
        )
        self._inference_thread.start()

    def _inference_loop(self) -> None:
        """Background inference thread for VLA / RTC mode.

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

            # Read latest preprocessed observation (non-blocking)
            with self._obs_lock:
                obs = self._latest_obs

            if obs is None:
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
            t0 = time.monotonic()
            try:
                raw = self.model.predict_action_chunk(
                    obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=self.model.config.rtc_config.execution_horizon,
                )
            except Exception as e:
                import traceback
                self.get_logger().error(f"[RTC] predict_action_chunk failed: {e}")
                self.get_logger().error(traceback.format_exc())
                time.sleep(0.005)
                continue

            elapsed = time.monotonic() - t0
            self._latency_tracker.add(elapsed)
            new_delay = math.ceil(elapsed * self.control_freq)

            # Postprocess in inference thread (official pattern from eval_with_real_robot.py):
            #   original = raw (for RTC guidance of the next chunk)
            #   processed = denormalized (ready for the robot)
            original = raw.squeeze(0).clone()
            if self.postprocessor:
                processed = self.postprocessor.process_action(raw.squeeze(0))
            else:
                processed = original

            self._action_queue.merge(original, processed, new_delay, idx_before)
            self.metrics.record_inference()

    def _preprocess_vla_observation(self, observation: dict) -> dict:
        """Preprocess a raw observation for VLA models.

        Follows the official lerobot test convention: build a flat batch dict with
        all observation.* keys plus "task" (as a list of strings), then call the
        preprocessor directly as a callable. The pipeline's to_transition / to_output
        converters handle observation splitting, task → complementary_data routing,
        tokenization, normalization, and device placement in one pass.

        Reference: tests/policies/pi0_pi05/test_pi05_rtc.py
        """
        if self.preprocessor:
            # Build flat batch dict: observation keys + task key (list of strings)
            batch = dict(observation)
            if self.task_description:
                batch["task"] = [self.task_description]
            # preprocessor(batch) → batch_to_transition → _forward → transition_to_batch
            # Output is a flat dict with observation.language.tokens etc. at top level
            observation = self.preprocessor(batch)
        return self._move_to_device(observation)

    def _obs_update(self) -> None:
        """Observation update timer (unified for all models).

        VLA: preprocess and update shared snapshot for background inference thread.
        ACT/Diffusion: preprocess, run select_action, push result to deque.
        """
        if self._shutting_down:
            return
        observation = self.strategy.get_observation(self.camera_names)
        if observation is None:
            return

        try:
            if self._is_vla:
                obs = self._preprocess_vla_observation(observation)
                with self._obs_lock:
                    self._latest_obs = obs
            else:
                # Keep a reference to the raw (unnormalised) observation so we can
                # capture the joint-state baseline when a new chunk is generated.
                _raw_obs = observation

                # EE obs conversion: must happen before preprocessor so the
                # normaliser sees the correct (rot6d) obs.state layout.
                _ee_obs_window_rel = None
                if self.is_ee_relative and "observation.state" in _raw_obs:
                    # ee_relative: maintain raw abs obs history and re-relativize the full
                    # window every step (anchor = current EE).
                    _s_raw = _raw_obs["observation.state"]
                    if hasattr(_s_raw, "numpy"):
                        _s_np = (_s_raw.squeeze(0).numpy() if _s_raw.dim() > 1 else _s_raw.numpy())
                    elif hasattr(_s_raw, "cpu"):
                        _s_np = _s_raw.cpu().numpy()
                    else:
                        _s_np = np.asarray(_s_raw)
                    _s_np = _s_np.flatten().astype(np.float64)
                    self._ee_raw_obs_buf.append(_s_np)

                    if len(self._ee_raw_obs_buf) == self._ee_n_obs_steps:
                        observation, _ee_obs_window_rel = self._apply_ee_relative_obs(observation)
                    else:
                        # Warm-up (buffer not yet full): relativize current step to itself
                        # → identity [0,0,0, 1,0,0,0,1,0, gripper].  This gives the correct
                        # 10-dim shape for the preprocessor/model, though the obs window
                        # won't be perfectly relativized until the buffer fills.
                        from anvil_shared.ee_transform import ee_obs_relative_forward as _eorf
                        _rel_id = _eorf(_s_np[np.newaxis], _s_np)[0]
                        observation = dict(observation)
                        observation["observation.state"] = torch.tensor(_rel_id, dtype=torch.float32).unsqueeze(0)

                elif (self.ee_abs_uses_rot6d_obs or self.is_ee_delta) and "observation.state" in _raw_obs:
                    # ee_abs (rot6d checkpoint) or ee_delta: convert obs.state from quat
                    # (8n) to rot6d (10n), absolute — no relativization (ee_delta keeps
                    # observation.state absolute, matching native's convention; only
                    # `action` carries the delta representation, and it's restored fresh
                    # per publish tick in _publish_loop, not here).
                    from anvil_shared.ee_transform import ee_obs_abs_forward as _eobsf
                    _s_raw = _raw_obs["observation.state"]
                    if hasattr(_s_raw, "numpy"):
                        _s_np = (_s_raw.squeeze(0).numpy() if _s_raw.dim() > 1 else _s_raw.numpy())
                    elif hasattr(_s_raw, "cpu"):
                        _s_np = _s_raw.cpu().numpy()
                    else:
                        _s_np = np.asarray(_s_raw)
                    _s_np = _s_np.flatten().astype(np.float64)
                    _abs_rot6d = _eobsf(_s_np)  # (10*n_arms,)
                    observation = dict(observation)
                    observation["observation.state"] = torch.tensor(
                        _abs_rot6d, dtype=torch.float32
                    ).unsqueeze(0)  # (1, 10*n_arms)

                # Capture raw absolute EE obs (quat layout) — for monitor when enabled,
                # and ALWAYS for ee_delta (the decoupled publish loop reads this every
                # tick, under _obs_lock, regardless of monitor status).
                # _raw_obs still points to the original observation dict before any
                # in-place conversion, so this always holds the quat-layout state.
                if self.is_ee and "observation.state" in _raw_obs and (
                    self._monitor_enable
                    or self.is_ee_delta
                    or not getattr(self, "_homing_confirmed", True)
                ):
                    _mon_s = _raw_obs["observation.state"]
                    if hasattr(_mon_s, "numpy"):
                        _mon_np = _mon_s.squeeze(0).numpy() if _mon_s.dim() > 1 else _mon_s.numpy()
                    else:
                        _mon_np = np.asarray(_mon_s)
                    _mon_np = _mon_np.flatten().astype(np.float64)
                    self._last_raw_ee_obs_np: np.ndarray = _mon_np
                    if self.is_ee_delta:
                        with self._obs_lock:
                            self._ee_delta_latest_obs_quat = _mon_np
                            self._ee_delta_latest_obs_seq = self.strategy.get_ee_obs_sequence_snapshot()
                            self._ee_delta_obs_gen += 1
                            if self._debug:
                                self.get_logger().info(
                                    f"[DEBUG-ANCHOR] obs_update wrote gen={self._ee_delta_obs_gen} "
                                    f"seq={self._ee_delta_latest_obs_seq} "
                                    f"full={_mon_np.round(6).tolist()} t={time.monotonic():.4f}"
                                )

                # Pre-replay homing (DatasetGtReplayerNode only — see its
                # _check_homing_arrival): hold here, never advance _produce_action's
                # cursor, until the live pose is confirmed at the episode's start
                # pose. getattr default keeps this a no-op for every other caller.
                if self.is_ee and not getattr(self, "_homing_confirmed", True):
                    self._check_homing_arrival()
                    return

                action = self._produce_action(observation, _ee_obs_window_rel)
                if action is not None:
                    self._classic_action_deque.append(action)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Observation/inference error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _produce_action(
        self, observation: dict, ee_obs_window_rel: np.ndarray | None
    ) -> np.ndarray | None:
        """Run the model forward pass and return the next physical-units action.

        This is the classic (ACT/Diffusion) prediction seam: preprocess → run the
        model → postprocess → restore (ee_relative only). Returns the action in
        physical units, ready to enter ``_classic_action_deque`` — or ``None`` to
        skip this tick (produce no action).

        Overridden by ``DatasetGtReplayerNode`` to return a recorded ground-truth
        action instead of a model prediction, at this exact seam, so the rest of
        the pipeline (deque, decoupled delta-mode publish loop, restore/publish)
        runs completely unmodified.
        """
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
        # the current state.  We capture that state as the n-0 chunk anchor.
        # Only ee_relative needs chunk-level restore; ee_abs/joint_abs do not.
        _needs_restore = self.is_ee_relative
        _is_new_chunk = (
            _needs_restore
            and hasattr(self.model, "_queues")
            and len(self.model._queues.get("action", [])) == 0
        )

        if self.preprocessor:
            observation = self.preprocessor(dict(observation))
        observation = self._move_to_device(observation)

        # ee_relative: pre-fill model's obs queue with normalized historical
        # relative obs (anchored to current EE).  Must run after preprocessor
        # (queue stores normalized tensors) and before select_action.
        if self.is_ee_relative and ee_obs_window_rel is not None and self._ee_n_obs_steps > 1:
            self._prefill_ee_relative_queue(ee_obs_window_rel)

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
            # the whole chunk can be denormalized together for the ee_relative restore.
            if _is_new_chunk and _needs_restore and hasattr(self.model, "_queues"):
                _rest_norm = [a.detach().clone() for a in self.model._queues.get("action", [])]
            else:
                _rest_norm = None

        # Capture reference state right after chunk generation.
        # Use _ee_raw_obs_buf[-1] — the same raw absolute EE pose used as the
        # obs anchor in _apply_ee_relative_obs — so the action-restore frame is
        # explicitly identical to the obs-relativization frame.
        if _is_new_chunk and self._ee_raw_obs_buf:
            self._relative_anchor_state = np.asarray(
                self._ee_raw_obs_buf[-1], dtype=np.float64
            ).flatten()

        if self.postprocessor:
            action = self.postprocessor.process_action(action)

        if isinstance(action, torch.Tensor):
            if action.dim() > 1:
                action = action.squeeze(0)
            action = action.cpu().numpy()

        # [DEBUG] Point 3: action after postprocessor, before n-0 restore
        if self._debug and _is_new_chunk:
            self.get_logger().info(
                f"[DEBUG] action (post-postproc): [{', '.join(f'{v:.4f}' for v in action)}]"
            )

        # Chunk-level ee_relative restore via shadow queue.
        # The model's internal queue stores normalized tensors; we denormalize
        # the full chunk together, restore rel → absolute, then serve absolute
        # values from a shadow queue so we never re-enter normalized space.
        # ee_abs and joint_abs require no restore — model output is already absolute.
        if self.is_ee_relative:
            if _is_new_chunk and self._relative_anchor_state is not None:
                if _rest_norm is not None:
                    _rest_denorm = [self._denorm_queue_action(a) for a in _rest_norm]
                    _chunk = np.stack([action] + _rest_denorm) if _rest_denorm else action[np.newaxis]
                else:
                    _chunk = action[np.newaxis]
                _abs = ee_relative_restore_chunk(_chunk, self._relative_anchor_state)
                self._abs_shadow_queue = deque(_abs[1:])
                action = _abs[0]
            elif self._abs_shadow_queue:
                action = self._abs_shadow_queue.popleft()
            elif not hasattr(self.model, "_queues") and self._relative_anchor_state is not None:
                action = ee_relative_restore_chunk(action[np.newaxis], self._relative_anchor_state)[0]

        if _will_run_forward:
            self.metrics.record_inference()

        return action

    def _publish_loop(self) -> None:
        """Action publish timer (unified for all models).

        VLA: pop from ActionQueue (filled by background inference thread).
        ACT/Diffusion: pop from deque (filled by _obs_update).
        ee_delta: the deque holds DELTAS, not pre-restored absolutes — the
            decoupled delta-mode publish loop composes
            absolute_target = obs_pose ∘ delta FRESH, right here, at every
            publish tick, against whatever the freshest observed pose is at
            that instant (NOT the pose from when the chunk/delta was
            predicted). This is what makes full open-loop chunk execution
            safe without anchor staleness or forward-integration — see
            claude_docs/ee-delta-flow-plan.md, Item 2.
        """
        if self._shutting_down:
            return

        # Pre-replay homing (DatasetGtReplayerNode only): republish the frozen
        # home target every tick instead of popping/composing from the deque,
        # until _obs_update's arrival check confirms it and flips this back.
        if self.is_ee and not getattr(self, "_homing_confirmed", True):
            self._publish_home_target()
            return

        self.metrics.record_control_loop()

        if self._is_vla:
            action = self._action_queue.get()
            if self._debug:
                self._queue_depths.append(self._action_queue.qsize())
            if action is None:
                self._vla_skip_count += 1
                return
            if isinstance(action, torch.Tensor):
                if action.dim() > 1:
                    action = action.squeeze(0)
                action = action.cpu().numpy()
        else:
            if not self._classic_action_deque:
                return
            if self.is_ee_delta:
                with self._obs_lock:
                    _latest_obs = self._ee_delta_latest_obs_quat
                    _latest_seq = self._ee_delta_latest_obs_seq
                    _gen = self._ee_delta_obs_gen
                if _latest_obs is None:
                    # No observation yet (startup warm-up) — skip this tick rather
                    # than publish an un-composed delta; don't drop the queued
                    # delta, it will compose correctly once an obs arrives.
                    return

                _prev_seq = self._ee_delta_last_published_seq
                if (
                    _prev_seq is not None
                    and _latest_seq is not None
                    and not all(
                        # None means that arm's SequenceStalenessGuard has degraded
                        # (peer doesn't populate CommandedEEPose.sequence — see
                        # ee_obs_sequence_guard.py) and given up on sequence-based
                        # gating for it; treat as trivially advanced rather than
                        # holding forever on an arm we can no longer trust this way.
                        cur is None or prev is None or cur > prev
                        for cur, prev in zip(_latest_seq, _prev_seq)
                    )
                ):
                    # The anchor hasn't genuinely advanced (per CommandedEEPose.sequence)
                    # for at least one arm since the tick we last composed against — the
                    # echo for our last published command hasn't arrived/been processed
                    # yet. Hold: leave the delta queued and skip publishing this tick,
                    # rather than compose a fresh delta against an anchor we've already
                    # consumed once. Composing anyway would silently skip one row of the
                    # recorded trajectory with no way to recover (see
                    # claude_docs/gt-replayer-correctness-test-plan.md's staleness
                    # investigation) — holding preserves the 1:1 obs/delta pairing the
                    # decoupled compose design depends on. Retried automatically next
                    # tick once the anchor catches up.
                    self.get_logger().warn(
                        f"[ee_delta] holding publish tick: obs sequence {_latest_seq} has "
                        f"not advanced past {_prev_seq} for all arms — anchor not yet fresh"
                    )
                    return

                _delta_popped = self._classic_action_deque.popleft()
                if self._debug:
                    _stale = (
                        self._ee_delta_last_published_gen is not None
                        and _gen == self._ee_delta_last_published_gen
                    )
                    self.get_logger().info(
                        f"[DEBUG-ANCHOR] publish_loop read gen={_gen} stale_reuse={_stale} "
                        f"seq={_latest_seq} "
                        f"full={_latest_obs.round(6).tolist()} "
                        f"delta={np.asarray(_delta_popped).round(6).tolist()} t={time.monotonic():.4f}"
                    )
                    self._ee_delta_last_published_gen = _gen
                self._ee_delta_last_published_seq = _latest_seq
                action = ee_delta_restore_step(_delta_popped, _latest_obs)
            else:
                action = self._classic_action_deque.popleft()

        try:
            self._publish_action(action)
        except Exception as e:
            import traceback
            self.get_logger().error(f"Publish error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _apply_ee_relative_obs(self, observation: dict) -> tuple:
        """Convert absolute obs.state (8-dim/arm, quat) to relative (10-dim/arm, rot6d).

        Uses the full buffer window anchored to the current EE pose (last buffer entry).
        Returns (modified_observation_dict, obs_window_rel_np) where obs_window_rel_np
        is (n_obs_steps, 10*n_arms) — used later to pre-fill the model's obs queue.
        """
        from anvil_shared.ee_transform import ee_obs_relative_forward
        anchor = self._ee_raw_obs_buf[-1]                          # (8*n_arms,) absolute
        obs_window_np = np.stack(self._ee_raw_obs_buf)             # (n_obs_steps, 8*n_arms)
        obs_rel_np = ee_obs_relative_forward(obs_window_np, anchor)     # (n_obs_steps, 10*n_arms)
        observation = dict(observation)  # shallow copy — don't mutate caller's dict
        # Current step relative to itself → identity; preserve (1, 10n) batch dim from topic
        observation["observation.state"] = torch.tensor(obs_rel_np[-1], dtype=torch.float32).unsqueeze(0)
        return observation, obs_rel_np

    def _prefill_ee_relative_queue(self, obs_window_rel_np: np.ndarray) -> None:
        """Pre-fill the model's internal obs queue with historical normalized relative obs.

        This ensures the full obs window [t-(n-1), ..., t-1] is correctly relativized
        to the current anchor before select_action pushes the identity step t.
        """
        if not (hasattr(self.model, "_queues") and "observation.state" in self.model._queues):
            return

        queue = self.model._queues["observation.state"]
        queue.clear()

        try:
            model_device = next(iter(self.model.parameters())).device
        except StopIteration:
            model_device = torch.device(self.device)

        for i in range(len(obs_window_rel_np) - 1):
            # Keep (1, 10n) batch dim so queue entries match what populate_queues pushes.
            # predict_action_chunk does torch.stack(queue, dim=1) which requires uniform shapes.
            obs_t = torch.tensor(obs_window_rel_np[i], dtype=torch.float32).unsqueeze(0)  # (1,10n)
            if self.preprocessor:
                try:
                    norm = self.preprocessor({"observation.state": obs_t})
                    obs_t = norm["observation.state"]   # stays (1, 10n) — no squeeze
                except Exception as _exc:
                    self.get_logger().warn(
                        f"[ee_relative] prefill normalization failed at step {i}: {_exc} "
                        "— appending unnormalized obs (output will be incorrect)"
                    )
            queue.append(obs_t.to(model_device))

    def _reset_relative_anchor_state(self) -> None:
        """Reset n-0 chunk-anchor restore state; call whenever the model is reloaded."""
        self._relative_anchor_state = None
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
        """Publish action to arm controllers.

        Routes to EE or joint publishing path based on ``self.action_type``.
        """
        if self.is_ee:
            self._publish_ee_action(action)
        else:
            self._publish_joint_action(action)
        # Debug: track smoothness
        if self._smooth_tracker is not None:
            self._smooth_tracker.record(action)
        self.metrics.record_action_output()
        self._has_published = True

    def _publish_joint_action(self, action: np.ndarray) -> None:
        """Publish joint absolute actions as Float64MultiArray (joint_abs mode)."""
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

            # ee_relative restore is done upstream in _obs_update (chunk-level).
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

    def _publish_ee_action(self, action: np.ndarray) -> None:
        """Publish EE actions as CommandedEEPose messages (ee_abs / ee_relative mode).

        ``action`` is already in absolute rot6d space when it arrives here
        (ee_relative restore happens upstream in _obs_update).

        For each arm:
          - Slices ``action[action_start:action_end]`` (10 dims per arm)
          - Converts rot6d → quaternion via ``ee_poses_from_chunk``
          - Builds a ``CommandedEEPose`` and publishes to ``ee_command_topic``

        Gripper post-processing: the raw model prediction is shifted so the
        closed end becomes the origin, scaled by ``gripper_factor`` (default
        1.0; set <1 to squeeze toward closed for more grip force), shifted
        back, then clamped to [gripper_min, gripper_max] (defaults
        [-0.003, 0.05] m — same as the robot's own hardware clamp).
        All three parameters are readable per-arm from the inference config.
        """
        from anvil_msgs.msg import CommandedEEPose
        from geometry_msgs.msg import Point, Pose, Quaternion
        from std_msgs.msg import Header

        now = self.get_clock().now().to_msg()
        frame_id = self.config.get("frame_id", "world")

        monitor_cmd_parts: list[np.ndarray] = []
        arm_list = list(self.arms_config.items())

        for arm_idx, (arm_name, arm_config) in enumerate(arm_list):
            start_idx = arm_config.get("action_start", 0)
            end_idx = arm_config.get("action_end", start_idx + 10)

            # Per-arm gripper parameters — configurable in inference_ee.yaml arm block.
            g_min = arm_config.get("gripper_min", -0.003)   # hardware closed limit (m)
            g_max = arm_config.get("gripper_max", 0.05)     # hardware open limit (m)
            # gripper_factor < 1.0 squeezes toward closed (more grip force).
            # Formula: shift origin to closed end → scale → shift back → clamp.
            g_factor = arm_config.get("gripper_factor", 1.0)

            arm_action_abs = action[start_idx:end_idx]  # (10,) absolute rot6d

            # Convert single-step action to pose dict
            poses = ee_poses_from_chunk(arm_action_abs[np.newaxis, :], n_arms=1)
            pose_dict = poses[0][0]  # arm_index 0 within the single-arm slice

            pos = pose_dict["pos"]
            quat_xyzw = pose_dict["quat_xyzw"]
            raw_grip = pose_dict["gripper"]
            # Shift origin to closed end → scale → shift back → clamp.
            # factor < 1 → closer to closed (more grip force); 1.0 → no change.
            squeezed = (raw_grip - g_min) * g_factor + g_min
            gripper = float(np.clip(squeezed, g_min, g_max))

            if self._debug:
                self.get_logger().info(
                    f"[DEBUG] EE cmd [{arm_name}]: "
                    f"pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}] "
                    f"quat=[{quat_xyzw[0]:.4f},{quat_xyzw[1]:.4f},"
                    f"{quat_xyzw[2]:.4f},{quat_xyzw[3]:.4f}] "
                    f"gripper={gripper:.4f} (raw={raw_grip:.4f} factor={g_factor:.2f})"
                )

            msg = CommandedEEPose()
            msg.header = Header()
            msg.header.stamp = now
            msg.header.frame_id = frame_id
            msg.pose = Pose()
            msg.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            msg.pose.orientation = Quaternion(
                x=float(quat_xyzw[0]),
                y=float(quat_xyzw[1]),
                z=float(quat_xyzw[2]),
                w=float(quat_xyzw[3]),
            )
            msg.gripper = gripper

            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)

            if self._monitor_enable:
                monitor_cmd_parts.append(arm_action_abs)

        if self._monitor_enable and monitor_cmd_parts:
            # Convert last raw EE obs (quat 8n) → rot6d (10n) to match command space.
            _mon_obs = np.zeros_like(action)
            if hasattr(self, "_last_raw_ee_obs_np"):
                from anvil_shared.ee_transform import ee_obs_abs_forward as _eobsf
                _mon_obs = _eobsf(self._last_raw_ee_obs_np).astype(np.float32)
            self._publish_monitor(
                obs_state=_mon_obs,
                raw_output=action,
                control_cmd=np.concatenate(monitor_cmd_parts),
            )

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
            if self._is_vla:
                self._log_stats_vla(logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz)
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

    def _log_stats_vla(self, logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log VLA (RTC) specific stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

        # VLA: additionally log queue size
        if hasattr(self, "_latency_tracker"):
            lat_mean = self._latency_tracker.mean()
            if lat_mean > 0 and hasattr(self, "_action_queue"):
                queue_size = self._action_queue.qsize()
                logger.info(f"  VLA queue    {queue_size}")

            # Debug: Action FPS, Eff ctrl Hz, queue depth stats, smoothness
            if self._debug and lat_mean > 0:
                cs = getattr(self.model.config, "chunk_size", 0)
                eh = getattr(self.model.config.rtc_config, "execution_horizon", 0)
                action_fps = cs / lat_mean
                eff_ctrl_hz = action_fps * eh / cs if cs > 0 else 0
                logger.info(f"  [DEBUG] Action FPS {action_fps:.1f}  Eff ctrl Hz {eff_ctrl_hz:.1f}")

        if self._debug and self._queue_depths:
            depths = np.array(self._queue_depths)
            skip_pct = self._vla_skip_count / max(len(self._queue_depths) + self._vla_skip_count, 1) * 100
            logger.info(f"  [DEBUG] Queue depth min={depths.min()} mean={depths.mean():.0f} max={depths.max()} skip={skip_pct:.1f}%")
            self._queue_depths.clear()
            self._vla_skip_count = 0

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

    def _log_stats_classic(self, logger, dt, stats, control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log non-VLA (ACT/Diffusion) stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

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

    def reset_policy(self) -> None:
        """Reset policy state."""
        if not hasattr(self, "model"):
            return
        self.get_logger().info("Resetting policy state...")
        if hasattr(self.model, "reset"):
            self.model.reset()
        if self._is_vla and hasattr(self, "_action_queue"):
            from lerobot.policies.rtc.action_queue import ActionQueue
            self._action_queue = ActionQueue(self.model.config.rtc_config)
            with self._obs_lock:
                self._latest_obs = None
        if hasattr(self, "_latency_tracker"):
            self._latency_tracker.reset()
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def _publish_hold_position(self) -> None:
        """Publish current joint positions to hold the robot in place on shutdown.

        Skipped in EE mode (ee_abs / ee_relative) — the robot controller (anvil-workcell)
        retains the last commanded pose autonomously.  Sending a zero Float64MultiArray
        would be interpreted as joint commands, which is wrong for EE mode.
        """
        if not hasattr(self, "arm_publishers"):
            return
        if self.is_ee:
            self.get_logger().info("Shutdown: EE mode — hold-position skipped (robot retains last pose)")
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
        for timer_name in ("_obs_timer", "_publish_timer", "_stats_timer"):
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

    # Docker/docker-compose sends SIGTERM on stop (not SIGINT) — Python's default
    # SIGTERM action is immediate termination with no `finally` block, which was
    # skipping strategy.cleanup() and leaving image-worker video writers
    # unreleased (corrupted mp4s, missing moov atom).
    #
    # Call rclpy.shutdown() directly (matches inference_monitor_node.py's existing
    # SIGINT/SIGTERM handler) rather than raising KeyboardInterrupt: executor.spin()'s
    # own loop is `while self._context.ok(): spin_once()`, so invalidating the
    # context makes it return normally on its next check — no reliance on a Python
    # exception being raised cleanly out of a blocked C-extension wait call.
    # The log line below is deliberate: if a video-writer bug ever recurs, checking
    # whether this line appears tells us immediately whether the handler fired at
    # all, versus fired but ran out of time before the container was SIGKILLed.
    def _sigterm_handler(signum, frame):
        # node may still be None if SIGTERM arrives while LeRobotInferenceNode()
        # is under construction (e.g. still loading the model).
        if node is not None:
            node.get_logger().info("[shutdown] SIGTERM received, shutting down...")
        else:
            print("[shutdown] SIGTERM received during startup, shutting down...")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        node = LeRobotInferenceNode()

        # Use MultiThreadedExecutor: VLA mode needs 3+ threads
        # (obs timer, publish timer, stats timer, joint subscription)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        node.get_logger().info("Starting inference loop...")
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
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
