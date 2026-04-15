"""LeRobot Control - ROS2 inference for trained LeRobot models

Multi-process inference node with shared-memory image workers.

Usage:
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p config_file:=/path/to/config.yaml
"""

__version__ = "0.5.0"

from .model_loader import ModelLoader, reset_model_state, set_deterministic_mode

# Optional ROS2-dependent imports
try:
    from .action_limiter import ActionLimiter
    from .image_converter import ImageConverter
    from .metrics_tracker import MetricsTracker
    from .observation_manager import ObservationManager
    from .shared_image_buffer import SharedImageBuffer, SharedJointStateBuffer
except ImportError:
    # These will fail in non-ROS2 environments due to sensor_msgs, etc.
    # We skip them so that ModelLoader can still be imported for offline eval.
    pass

__all__ = [
    "ModelLoader",
    "set_deterministic_mode",
    "reset_model_state",
    # The following may not be available if ROS2 deps are missing
    "ObservationManager",
    "ImageConverter",
    "ActionLimiter",
    "MetricsTracker",
    "SharedImageBuffer",
    "SharedJointStateBuffer",
]
