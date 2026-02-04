#!/usr/bin/env python3
"""
LeRobot Inference Node for Robot Arms

Unified inference node with pluggable strategies for observation acquisition:
- mp (default): Multi-process architecture with better isolation
- single: Single-process with threading (simpler, good for debugging)

Usage:
    # Multi-process mode (default)
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p config_file:=/path/to/config.yaml

    # Single-process mode (simpler, good for debugging)
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p mode:=single

Subscribes to:
    - Joint states topic (sensor_msgs/JointState)
    - Camera image topics (sensor_msgs/Image or CompressedImage)

Publishes:
    - Forward position controller command topics (std_msgs/Float64MultiArray)
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Any

from std_msgs.msg import Float64MultiArray

from .model_loader import ModelLoader, set_deterministic_mode
from .action_limiter import ActionLimiter
from .metrics_tracker import MetricsTracker


class LeRobotInferenceNode(Node):
    """
    ROS2 node for LeRobot model inference and robot control.

    Uses pluggable strategies for observation acquisition:
    - mp (default): Multi-process with shared memory, better isolation
    - single: Single-process with threading, simpler for debugging
    """

    def __init__(self, parameter_overrides: list = None):
        super().__init__('lerobot_inference_node', parameter_overrides=parameter_overrides or [])

        # Declare parameters
        self._declare_parameters()

        # Callback group for control loop
        self._control_callback_group = MutuallyExclusiveCallbackGroup()

        # Load configuration
        self.config = self._load_config(self.get_parameter('config_file').value)
        self._apply_config()

        # Log configuration
        self._log_config()

        # Initialize metrics
        self.metrics = MetricsTracker()

        # Create strategy based on mode parameter
        self.strategy = self._create_strategy()
        self.strategy.setup(
            node=self,
            config={'device': self.device, **self.config},
            camera_mapping=self.camera_mapping,
            joint_names_config=self.joint_names_config,
            joint_state_topic=self.joint_state_topic,
            image_shape=self.image_shape,
            metrics=self.metrics,
        )

        # Load model with processors
        self._load_model()

        # Setup action limiter
        self.action_limiter = ActionLimiter(
            max_delta=self.max_position_delta,
            model_joint_order=self.joint_names_config.get('model_joint_order', []),
            controller_joint_order=self.joint_names_config.get('controller_joint_order', []),
            use_delta_actions=self.use_delta_actions,
            logger=self.get_logger(),
        )

        # Setup publishers
        self._setup_publishers()

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop,
            callback_group=self._control_callback_group,
        )

        # Stats logging timer
        self._stats_log_interval = 5.0
        self._stats_timer = self.create_timer(
            self._stats_log_interval,
            self._log_input_stats,
            callback_group=self._control_callback_group,
        )

        mode_name = self.get_parameter('mode').value
        self.get_logger().info("=" * 50)
        self.get_logger().info("Node initialized - Ready for inference")
        self.get_logger().info(f"Mode: {mode_name.upper()}")
        self.get_logger().info(f"Target frequency: {self.control_freq} Hz")
        self.get_logger().info("=" * 50)

    def _declare_parameters(self) -> None:
        """Declare ROS2 parameters."""
        self.declare_parameter('model_path', '')
        self.declare_parameter('config_file', '')
        self.declare_parameter('control_frequency', 30.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('model_type', 'act')
        self.declare_parameter('max_position_delta', 0.1)
        self.declare_parameter('deterministic', False)
        self.declare_parameter('deterministic_seed', 42)
        self.declare_parameter('throttle_duration', 1.0)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('image_width', 640)
        # Mode: 'mp' (default) or 'single'
        self.declare_parameter('mode', 'mp')

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        if not config_file:
            self.get_logger().warn("No config_file specified, using defaults")
            return {}

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _apply_config(self) -> None:
        """Apply configuration from parameters and config file."""
        self.model_path = self.get_parameter('model_path').value
        if not self.model_path:
            raise ValueError("model_path parameter is required")

        self.control_freq = self.get_parameter('control_frequency').value
        self.device = self.get_parameter('device').value
        self.throttle_duration = self.get_parameter('throttle_duration').value

        # Model type: config file takes precedence, ROS2 param as fallback
        model_config = self.config.get('model', {})
        self.model_type = model_config.get('type', self.get_parameter('model_type').value)

        # Image dimensions
        self.image_height = self.get_parameter('image_height').value
        self.image_width = self.get_parameter('image_width').value
        self.image_shape = (self.image_height, self.image_width, 3)

        # Safety parameters from config file
        safety_config = self.config.get('safety', {})
        self.max_position_delta = safety_config.get('max_position_delta', 0.1)

        # Model parameters (use_delta_actions, task_description)
        self.use_delta_actions = model_config.get('use_delta_actions', False)
        self.task_description = model_config.get('task_description', '')

        # Topics and mapping
        self.joint_state_topic = self.config.get('joint_state_topic', '/joint_states')
        self.camera_mapping = self.config.get('camera_mapping', {})
        self.camera_names = list(self.camera_mapping.values())
        self.arms_config = self.config.get('arms', {})
        self.joint_names_config = self.config.get('joint_names', {})

        # Model runtime config overrides (model_config already read above)
        self.chunk_size_override = model_config.get('chunk_size', None)
        self.n_action_steps_override = model_config.get('n_action_steps', None)
        self.temporal_ensemble_coeff = model_config.get('temporal_ensemble_coeff', None)

    def _log_config(self) -> None:
        """Log configuration summary."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("LeRobot Inference Node")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Frequency: {self.control_freq} Hz")
        self.get_logger().info(f"Max delta: {self.max_position_delta} rad")
        self.get_logger().info(f"Image shape: {self.image_shape}")
        self.get_logger().info(f"Cameras: {self.camera_names}")
        self.get_logger().info(f"Arms: {list(self.arms_config.keys())}")

    def _create_strategy(self) -> Any:
        """
        Factory: select strategy based on mode parameter.

        mp (default): Multi-process with shared memory, better process isolation
        single: Single-process with threading, simpler architecture for debugging
        """
        mode = self.get_parameter('mode').value

        if mode == 'single':
            self.get_logger().info(
                "Using single-process mode (simpler, good for debugging)"
            )
            from .strategies.single_process import SingleProcessStrategy
            return SingleProcessStrategy()
        else:
            # Default: multi-process
            self.get_logger().info(
                "Using multi-process mode (better isolation)"
            )
            from .strategies.multi_process import MultiProcessStrategy
            return MultiProcessStrategy()

    def _build_config_overrides(self) -> dict:
        """Build config overrides dict from yaml config."""
        overrides = {}

        if self.chunk_size_override is not None:
            overrides['chunk_size'] = self.chunk_size_override

        if self.n_action_steps_override is not None:
            overrides['n_action_steps'] = self.n_action_steps_override

        if self.temporal_ensemble_coeff is not None:
            overrides['temporal_ensemble_coeff'] = self.temporal_ensemble_coeff
            # Temporal ensemble requires n_action_steps=1
            if self.n_action_steps_override is None or self.n_action_steps_override > 1:
                self.get_logger().warn(
                    "temporal_ensemble requires n_action_steps=1, forcing override"
                )
                overrides['n_action_steps'] = 1

        return overrides

    def _load_model(self) -> None:
        """Load model and processors."""
        # Enable deterministic mode if requested
        if self.get_parameter('deterministic').value:
            seed = self.get_parameter('deterministic_seed').value
            set_deterministic_mode(seed)
            self.get_logger().info(f"Deterministic mode enabled with seed={seed}")

        # Build config overrides
        config_overrides = self._build_config_overrides()

        # Load model - overrides are applied after loading in ModelLoader
        loader = ModelLoader(
            self.model_path,
            self.device,
            self.model_type,
            config_overrides=config_overrides,
            logger=self.get_logger(),
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self.get_logger().info("Model loaded with processors")

        # Log effective config
        self._log_effective_model_config()

        # Log SmolVLA language instruction (preprocessor handles tokenization)
        if self.model_type == "smolvla" and self.task_description:
            self.get_logger().info(f"Task description: '{self.task_description}'")

    def _log_effective_model_config(self) -> None:
        """Log the effective model configuration after overrides."""
        if not hasattr(self.model, 'config'):
            return

        config = self.model.config
        self.get_logger().info(
            f"Effective model config: chunk_size={getattr(config, 'chunk_size', 'N/A')}, "
            f"n_action_steps={getattr(config, 'n_action_steps', 'N/A')}, "
            f"temporal_ensemble_coeff={getattr(config, 'temporal_ensemble_coeff', 'N/A')}"
        )

        # Verify temporal ensembler was created if coeff is set
        if getattr(config, 'temporal_ensemble_coeff', None) is not None:
            if hasattr(self.model, 'temporal_ensembler'):
                self.get_logger().info("Temporal ensembler initialized successfully")
            else:
                self.get_logger().error(
                    "temporal_ensemble_coeff is set but ensembler not created!"
                )

    def _setup_publishers(self) -> None:
        """Setup action publishers."""
        self.arm_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        for arm_name, arm_config in self.arms_config.items():
            cmd_topic = arm_config.get(
                'command_topic',
                f'/{arm_name}_forward_position_controller/commands',
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
                if self.model_type == "smolvla" and self.task_description:
                    # SmolVLA needs 'task' in complementary_data for tokenization
                    # Use full transition processing instead of just process_observation
                    from lerobot.processor.converters import create_transition
                    from lerobot.processor.core import TransitionKey
                    transition = create_transition(
                        observation=observation,
                        complementary_data={'task': self.task_description}
                    )
                    processed = self.preprocessor._forward(transition)
                    observation = processed[TransitionKey.OBSERVATION]
                else:
                    observation = self.preprocessor.process_observation(observation)

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

            # Publish action
            self._publish_action(action)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Inference error: {e}")
            self.get_logger().error(traceback.format_exc())

    def _publish_action(self, action: np.ndarray) -> None:
        """Publish action to arm controllers."""
        current_positions = self.strategy.get_current_joint_positions()
        joint_order = self.joint_names_config.get(
            'controller_joint_order',
            self.joint_names_config.get('joint_order', []),
        )

        for arm_name, arm_config in self.arms_config.items():
            start_idx = arm_config.get('action_start', 0)
            end_idx = arm_config.get('action_end', len(action))
            ros_prefix = arm_config.get('ros_prefix', arm_name)

            # Extract arm's portion of action
            arm_action = action[start_idx:end_idx].copy()

            # Get current positions for this arm
            arm_current = None
            if current_positions:
                arm_current = np.array([
                    current_positions.get(f"{ros_prefix}_{joint_order[i]}", 0.0)
                    for i in range(len(arm_action))
                ])

            # Process action (reorder + delta limit)
            arm_action = self.action_limiter.process(arm_action, arm_current)

            # Publish
            msg = Float64MultiArray()
            msg.data = arm_action.tolist()
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)

    def _log_input_stats(self) -> None:
        """Periodically log input reception statistics."""
        stats = self.metrics.get_stats()
        if stats['elapsed_sec'] < 1.0:
            return  # Wait for enough data

        self.get_logger().info("=" * 40)
        self.get_logger().info("INPUT RECEPTION STATS")
        self.get_logger().info("-" * 40)

        # Joint state FPS
        self.get_logger().info(f"Joint State:    {stats['joint_fps']:6.1f} Hz")

        # Per-camera FPS
        for camera in sorted(stats['image_fps'].keys()):
            fps = stats['image_fps'][camera]
            self.get_logger().info(f"Camera {camera:8s}: {fps:6.1f} Hz")

        self.get_logger().info("-" * 40)
        self.get_logger().info(f"Control Loop:   {stats['control_loop_fps']:6.1f} Hz")
        self.get_logger().info(
            f"Inference:      {stats['inference_fps']:6.1f} Hz "
            f"({stats['inference_count']} total)"
        )

        # Log frame counters from strategy if available
        if hasattr(self.strategy, 'get_frame_counters'):
            counters = self.strategy.get_frame_counters()
            if counters:
                self.get_logger().info("-" * 40)
                self.get_logger().info("Shared Memory Frame Counters:")
                for name, count in counters.items():
                    self.get_logger().info(f"  {name}: {count}")

        # Log bottleneck warning
        if stats['inference_fps'] < self.control_freq * 0.5:
            self._log_bottleneck_analysis(stats)

    def _log_bottleneck_analysis(self, stats: dict) -> None:
        """Analyze and log potential bottlenecks."""
        self.get_logger().warn("-" * 40)
        self.get_logger().warn("BOTTLENECK ANALYSIS")

        # Find the slowest camera
        if stats['image_fps']:
            slowest_camera = min(stats['image_fps'].items(), key=lambda x: x[1])
            min_camera_fps = slowest_camera[1]

            if min_camera_fps < self.control_freq:
                self.get_logger().warn(
                    f"Camera '{slowest_camera[0]}' is the bottleneck: "
                    f"{min_camera_fps:.1f} Hz < {self.control_freq:.0f} Hz target"
                )
                self.get_logger().warn(
                    "Inference can only run when ALL cameras have data. "
                    "Max inference rate = min(camera FPS rates)"
                )

        # Check if control loop ran but inference didn't
        if stats['control_loop_fps'] > stats['inference_fps'] * 1.5:
            skip_rate = (
                (stats['control_loop_count'] - stats['inference_count'])
                / max(stats['control_loop_count'], 1) * 100
            )
            self.get_logger().warn(
                f"Control loop ran {stats['control_loop_count']}x but only "
                f"{stats['inference_count']}x had complete obs ({skip_rate:.0f}% skipped)"
            )

        # Log last incomplete reason
        incomplete_reason = self.strategy.get_incomplete_reason()
        if incomplete_reason:
            self.get_logger().warn(f"Last incomplete reason: {incomplete_reason}")

    def reset_policy(self) -> None:
        """Reset policy state."""
        self.get_logger().info("Resetting policy state...")
        if hasattr(self.model, 'reset'):
            self.model.reset()
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def destroy_node(self) -> None:
        """Cleanup strategy and destroy node."""
        self.strategy.cleanup()
        super().destroy_node()


def main(args=None):
    """Main entry point with single-threaded executor."""
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = LeRobotInferenceNode()

        # Use SingleThreadedExecutor for reliable timer callbacks
        # Multi-process strategy handles parallelism via worker processes
        executor = SingleThreadedExecutor()
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


if __name__ == '__main__':
    main()
