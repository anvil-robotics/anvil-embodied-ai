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
from pathlib import Path

import time

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

        self._control_callback_group = MutuallyExclusiveCallbackGroup()
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

            self.control_timer = self.create_timer(
                1.0 / self.control_freq,
                self.control_loop,
                callback_group=self._control_callback_group,
            )

        self._log_startup()

        # Stats logging timer (always active — used for monitor_only mode too)
        self._stats_log_interval = 5.0
        self._stats_timer = self.create_timer(
            self._stats_log_interval,
            self._log_input_stats,
            callback_group=self._control_callback_group,
        )

        # Windowed rate tracking — store previous snapshot for delta computation
        self._prev_log_time: float | None = None
        self._prev_joint_count: int = 0
        self._prev_control_count: int = 0
        self._prev_inference_count: int = 0
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

        # Static fields from ROS2 params
        self.monitor_only = self.get_parameter("monitor_only").value
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
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self._loader = loader

        # Confirm final model_type (ModelLoader auto-detects if None was passed)
        self.model_type = loader.model_type

        if self.model_type in {"smolvla", "pi0", "pi0_fast", "groot", "xvla"} and not self.task_description:
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
            if self.model_type in {"smolvla", "pi0", "pi0_fast", "groot", "xvla"}:
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

    def control_loop(self) -> None:
        """Main control loop - strategy-agnostic."""
        self.metrics.record_control_loop()

        # Get observation via strategy
        observation = self.strategy.get_observation(self.camera_names)
        if observation is None:
            return

        try:
            # Preprocess observation
            if self.preprocessor:
                if self.model_type in {"smolvla", "pi0", "pi0_fast", "groot", "xvla"} and self.task_description:
                    # VLA-family policies need 'task' in complementary_data for tokenization
                    # Use full transition processing instead of just process_observation
                    from lerobot.processor.converters import create_transition
                    from lerobot.processor.core import TransitionKey

                    transition = create_transition(
                        observation=observation, complementary_data={"task": self.task_description}
                    )
                    processed = self.preprocessor._forward(transition)
                    observation = processed[TransitionKey.OBSERVATION]
                else:
                    observation = self.preprocessor.process_observation(observation)

            # Ensure observation tensors are on the same device as the model
            observation = self._move_to_device(observation)

            # Run inference
            with torch.inference_mode():
                action = self.model.select_action(observation)

            # Postprocess if pipeline available
            if self.postprocessor:
                action = self.postprocessor.process_action(action)

            # Convert to numpy
            if isinstance(action, torch.Tensor):
                if action.dim() > 1:
                    action = action.squeeze(0)
                action = action.cpu().numpy()

            self.metrics.record_inference()

            self._publish_action(action)

        except Exception as e:
            import traceback

            self.get_logger().error(f"Inference error: {e}")
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

            # Extract arm's portion of action
            arm_action = action[start_idx:end_idx].copy()

            # Get current positions for this arm
            arm_current = None
            if current_positions:
                arm_current = np.array(
                    [
                        current_positions.get(f"{ros_prefix}_{joint_order[i]}", 0.0)
                        for i in range(len(arm_action))
                    ]
                )

            # Process action (reorder + delta limit)
            arm_action = self.action_limiter.process(arm_action, arm_current)

            # Publish
            msg = Float64MultiArray()
            msg.data = arm_action.tolist()
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)

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
        self._prev_frame_counters = dict(frame_counters)

        # Find bottleneck camera (only relevant when not monitor_only)
        bottleneck_name = None
        if not self.monitor_only and camera_hz:
            slowest = min(camera_hz.items(), key=lambda x: x[1])
            if slowest[1] < self.control_freq:
                bottleneck_name = slowest[0]

        # Log unified stats block
        self.get_logger().info(f"-- Stats ({dt:.0f}s) " + "-" * 30)
        self.get_logger().info(f"  Joint State  {joint_hz:7.1f} Hz")
        for name in sorted(camera_hz.keys()):
            hz = camera_hz[name]
            delta = camera_delta.get(name, 0)
            marker = "  << bottleneck" if name == bottleneck_name else ""
            self.get_logger().info(f"  {name:12s}  {hz:7.1f} Hz  (+{delta} frames){marker}")

        if not self.monitor_only:
            self.get_logger().info(f"  Control Loop {control_hz:7.1f} Hz")
            self.get_logger().info(
                f"  Inference    {inference_hz:7.1f} Hz  ({stats['inference_count']} total)"
            )

            # Inline bottleneck warnings
            if bottleneck_name is not None:
                self.get_logger().warn(
                    f"  '{bottleneck_name}' limits inference to "
                    f"{camera_hz[bottleneck_name]:.1f} Hz (target: {self.control_freq:.0f} Hz)"
                )
            if control_hz > inference_hz * 1.5 and stats["inference_count"] > 0:
                skip_rate = (
                    (stats["control_loop_count"] - stats["inference_count"])
                    / max(stats["control_loop_count"], 1)
                    * 100
                )
                self.get_logger().warn(f"  {skip_rate:.0f}% control loops skipped (no complete obs)")
                incomplete_reason = self.strategy.get_incomplete_reason()
                if incomplete_reason:
                    self.get_logger().warn(f"  Last reason: {incomplete_reason}")

    def reset_policy(self) -> None:
        """Reset policy state."""
        if not hasattr(self, "model"):
            return
        self.get_logger().info("Resetting policy state...")
        if hasattr(self.model, "reset"):
            self.model.reset()
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def destroy_node(self) -> None:
        """Cleanup timers, strategy, and destroy node."""
        if hasattr(self, "control_timer"):
            self.control_timer.cancel()
        if hasattr(self, "_stats_timer"):
            self._stats_timer.cancel()
        self.strategy.cleanup()
        super().destroy_node()


def main(args=None):
    """Main entry point with single-threaded executor."""
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = LeRobotInferenceNode()

        # Use MultiThreadedExecutor so subscription callbacks (joint state)
        # can run concurrently with the control loop timer
        executor = MultiThreadedExecutor(num_threads=2)
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
