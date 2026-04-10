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
"""

import json
import math
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
        )

        # Non-VLA action buffer (ACT/Diffusion put actions here from obs timer)
        self._classic_action_deque: deque = deque(maxlen=10)

        if not self.monitor_only:
            self._setup_model()

            self.action_limiter = ActionLimiter(
                max_delta=self.max_position_delta,
                model_joint_order=self.joint_names_config.get("model_joint_order", []),
                controller_joint_order=self.joint_names_config.get("controller_joint_order", []),
                use_delta_actions=self.use_delta_actions,
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
        if self._debug and not self.monitor_only and hasattr(self, "model"):
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
            callback_group=self._publish_callback_group if not self.monitor_only else MutuallyExclusiveCallbackGroup(),
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
        self.declare_parameter("monitor_only", False)
        self.declare_parameter("debug", False)

        # Static fields from ROS2 params
        self.monitor_only = self.get_parameter("monitor_only").value
        self._debug = self.get_parameter("debug").value
        self.model_path = self.get_parameter("model_path").value
        if not self.model_path and not self.monitor_only:
            raise ValueError("model_path parameter is required")

        self.control_freq = self.get_parameter("control_frequency").value
        self.device = self.get_parameter("device").value

        # Load YAML config
        config_file = self.get_parameter("config_file").value
        self.config = self._load_yaml_config(config_file)

        # Fields from YAML config
        safety_config = self.config.get("safety", {})
        self.max_position_delta = safety_config.get("max_position_delta", 0.1)

        self.joint_state_topic = self.config.get("joint_state_topic", "/joint_states")
        self.camera_mapping = self.config.get("camera_mapping", {})
        self.camera_names = list(self.camera_mapping.values())
        self.arms_config = self.config.get("arms", {})
        self.joint_names_config = self.config.get("joint_names", {})

        # Inference tuning knobs (null = use checkpoint defaults)
        tuning_config = self.config.get("inference_tuning", {})
        self.n_action_steps_override = tuning_config.get("n_action_steps", None)
        self.temporal_ensemble_coeff = tuning_config.get("temporal_ensemble_coeff", None)

        # RTC config — passed to ModelLoader for VLA models; ignored for ACT/Diffusion
        self.rtc_config_yaml = self.config.get("rtc", {})

        # --- Checkpoint metadata (lightweight JSON reads, no tensor loading) ---
        meta = self._read_checkpoint_metadata()

        # image_shape: from config.json input_features — must match training
        # Default (480, 640, 3) is used only in monitor_only mode with no checkpoint
        self.image_shape = meta.get("image_shape", (480, 640, 3))

        # model_type: from config.json, YAML overrides if explicitly set
        model_cfg = self.config.get("model", {})
        self.model_type = model_cfg.get("type") or meta.get("model_type")

        # use_delta_actions: from anvil_config.json — must match training, no YAML override
        self.use_delta_actions = meta.get("use_delta_actions", False)

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
            meta["use_delta_actions"] = anvil.get("use_delta_actions", False)
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

        # Build inference tuning overrides
        config_overrides = {}
        if self.n_action_steps_override is not None:
            config_overrides["n_action_steps"] = self.n_action_steps_override
        if self.temporal_ensemble_coeff is not None:
            config_overrides["temporal_ensemble_coeff"] = self.temporal_ensemble_coeff
            if self.n_action_steps_override is None or self.n_action_steps_override > 1:
                self.get_logger().warn(
                    "temporal_ensemble requires n_action_steps=1, forcing override"
                )
                config_overrides["n_action_steps"] = 1

        loader = ModelLoader(
            self.model_path,
            self.device,
            self.model_type,
            config_overrides=config_overrides,
            logger=self.get_logger(),
            rtc_config_yaml=self.rtc_config_yaml,
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self._loader = loader

        # Confirm final model_type (ModelLoader auto-detects if None was passed)
        self.model_type = loader.model_type

        # VLA models: set up ActionQueue and start background inference thread
        if self._is_vla:
            self._setup_vla_inference()
            self._start_inference_thread()

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
        if self.monitor_only:
            logger.info("Mode:       Monitor Only (no model, no publishing)")
        else:
            logger.info(f"Model:      {self.model_path}")
            logger.info(f"Type:       {self.model_type or 'unknown'}")
            logger.info(f"Delta acts: {self.use_delta_actions}")
            if self.model_type in {"smolvla", "pi0", "pi05"}:
                logger.info(f"Task:       '{self.task_description}'")
        logger.info(f"Device:     {self.device}")
        logger.info(f"Frequency:  {self.control_freq} Hz")
        if not self.monitor_only:
            logger.info(f"Max delta:  {self.max_position_delta} rad")

        h, w, _ = self.image_shape
        res_note = "auto-detected from checkpoint" if self.model_path else "default"
        logger.info(f"Resolution: {w}x{h}  ({res_note})")

        logger.info(f"Cameras:    {self.camera_names}")
        logger.info(f"Arms:       {list(self.arms_config.keys())}")

        if not self.monitor_only and hasattr(self, "model") and hasattr(self.model, "config"):
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
        if not self.monitor_only and hasattr(self, "model"):
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated(self.device) / 1e6
                logger.info(f"GPU memory (weights): {gpu_mb:.0f} MB")
            try:
                import psutil

                cpu_mb = psutil.Process().memory_info().rss / 1e6
                logger.info(f"CPU RSS after load:   {cpu_mb:.0f} MB")
            except ImportError:
                pass

        if not self.monitor_only and self._is_vla:
            rtc = self.rtc_config_yaml
            logger.info("┌─ RTC ───────────────────────────────────────────────────┐")
            logger.info("│  Status:              ENABLED                           │")
            logger.info(f"│  execution_horizon  = {rtc.get('execution_horizon', 10):<4}                             │")
            logger.info(f"│  max_guidance_weight= {rtc.get('max_guidance_weight', 10.0):<6}                           │")
            logger.info(f"│  attention_schedule = {rtc.get('prefix_attention_schedule', 'EXP'):<6}                           │")
            logger.info(f"│  queue_threshold    = {rtc.get('queue_trigger_threshold', 30):<4}                             │")
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

    def _setup_vla_inference(self) -> None:
        """Initialise ActionQueue and LatencyTracker for VLA / RTC mode."""
        from lerobot.policies.rtc.action_queue import ActionQueue
        from lerobot.policies.rtc.latency_tracker import LatencyTracker

        self._action_queue = ActionQueue(self.model.config.rtc_config)
        self._latency_tracker = LatencyTracker(maxlen=100)
        self._latest_obs = None
        self._obs_lock = threading.Lock()
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
        observation = self.strategy.get_observation(self.camera_names)
        if observation is None:
            return

        try:
            if self._is_vla:
                obs = self._preprocess_vla_observation(observation)
                with self._obs_lock:
                    self._latest_obs = obs
            else:
                if self.preprocessor:
                    observation = self.preprocessor(dict(observation))
                observation = self._move_to_device(observation)

                with torch.inference_mode():
                    action = self.model.select_action(observation)

                if self.postprocessor:
                    action = self.postprocessor.process_action(action)

                if isinstance(action, torch.Tensor):
                    if action.dim() > 1:
                        action = action.squeeze(0)
                    action = action.cpu().numpy()

                self._classic_action_deque.append(action)
                self.metrics.record_inference()

        except Exception as e:
            import traceback
            self.get_logger().error(f"Observation/inference error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _publish_loop(self) -> None:
        """Action publish timer (unified for all models).

        VLA: pop from ActionQueue (filled by background inference thread).
        ACT/Diffusion: pop from deque (filled by _obs_update).
        """
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
            action = self._classic_action_deque.popleft()

        try:
            self._publish_action(action)
        except Exception as e:
            import traceback
            self.get_logger().error(f"Publish error: {e}")
            self.get_logger().error(traceback.format_exc())

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

            arm_action = self.action_limiter.process(arm_action, arm_current)

            msg = Float64MultiArray()
            msg.data = arm_action.tolist()
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)

        # Debug: track smoothness
        if self._smooth_tracker is not None:
            self._smooth_tracker.record(action)
        self.metrics.record_action_output()

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

        # Find bottleneck camera (only relevant when not monitor_only)
        bottleneck_name = None
        if not self.monitor_only and camera_hz:
            slowest = min(camera_hz.items(), key=lambda x: x[1])
            if slowest[1] < self.control_freq:
                bottleneck_name = slowest[0]

        # Common header: joint state + cameras
        logger = self.get_logger()
        logger.info(f"-- Stats ({dt:.0f}s) " + "-" * 30)
        logger.info(f"  Joint State  {joint_hz:7.1f} Hz")
        for name in sorted(camera_hz.keys()):
            hz = camera_hz[name]
            delta = camera_delta.get(name, 0)
            marker = "  << bottleneck" if name == bottleneck_name else ""
            logger.info(f"  {name:12s}  {hz:7.1f} Hz  (+{delta} frames){marker}")

        if not self.monitor_only:
            if self._is_vla:
                self._log_stats_vla(logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz)
            else:
                self._log_stats_classic(logger, dt, stats, control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz)

    def _log_stats_vla(self, logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log VLA (RTC) specific stats."""
        logger.info(f"  Action output{action_output_hz:7.1f} Hz")
        logger.info(f"  Inference    {inference_hz:7.1f} Hz  ({stats['inference_count']} total)")

        # VLA latency + queue size (always)
        if hasattr(self, "_latency_tracker"):
            vals = self._latency_tracker._values
            lat_mean = float(np.mean(vals)) if vals else 0.0
            lat_std = float(np.std(vals)) if vals else 0.0
            lat_p95 = self._latency_tracker.p95() or 0.0
            queue_size = self._action_queue.qsize() if hasattr(self, "_action_queue") else 0
            logger.info(
                f"  VLA latency  mean={lat_mean * 1000:.1f}ms  "
                f"std={lat_std * 1000:.1f}ms  p95={lat_p95 * 1000:.1f}ms  queue={queue_size}"
            )

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
            logger.warn(f"  '{bottleneck_name}' limits to {camera_hz[bottleneck_name]:.1f} Hz (target: {self.control_freq:.0f} Hz)")

    def _log_stats_classic(self, logger, dt, stats, control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log non-VLA (ACT/Diffusion) stats."""
        logger.info(f"  Action output{action_output_hz:7.1f} Hz")
        logger.info(f"  Inference    {inference_hz:7.1f} Hz  ({stats['inference_count']} total)")

        if self._debug and self._smooth_tracker is not None:
            smooth = self._smooth_tracker.get_stats()
            if smooth:
                logger.info(
                    f"  [DEBUG] Action D mean={smooth['delta_mean']:.4f} "
                    f"std={smooth['delta_std']:.4f} max={smooth['delta_max']:.4f} "
                    f"jerk={smooth['jerk_mean']:.4f}"
                )

        if bottleneck_name is not None:
            logger.warn(f"  '{bottleneck_name}' limits to {camera_hz[bottleneck_name]:.1f} Hz (target: {self.control_freq:.0f} Hz)")

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
            self._latency_tracker.reset()
            with self._obs_lock:
                self._latest_obs = None
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def destroy_node(self) -> None:
        """Cleanup timers, inference thread, strategy, and destroy node."""
        # Stop background RTC inference thread before cancelling timers
        if hasattr(self, "_inference_stop"):
            self._inference_stop.set()
        if hasattr(self, "_inference_thread"):
            self._inference_thread.join(timeout=2.0)
        for timer_name in ("control_timer", "_obs_timer", "_publish_timer", "_stats_timer"):
            timer = getattr(self, timer_name, None)
            if timer:
                timer.cancel()
        self.strategy.cleanup()
        super().destroy_node()


def main(args=None):
    """Main entry point with single-threaded executor."""
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = LeRobotInferenceNode()

        # Use MultiThreadedExecutor: VLA mode needs 3+ threads
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
