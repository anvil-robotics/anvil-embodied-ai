"""LeRobot Control - ROS2 inference for trained LeRobot models

Unified inference node with pluggable strategies:
- mp (default): Multi-process architecture with better isolation
- single: Single-process with threading (simpler, good for debugging)

Usage:
    # Multi-process mode (default)
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p config_file:=/path/to/config.yaml

    # Single-process mode (simpler, good for debugging)
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p mode:=single
"""

__version__ = "0.5.0"

from .model_loader import ModelLoader, set_deterministic_mode, reset_model_state
from .observation_manager import ObservationManager
from .image_converter import ImageConverter
from .action_limiter import ActionLimiter
from .metrics_tracker import MetricsTracker
from .shared_image_buffer import SharedImageBuffer, SharedJointStateBuffer

__all__ = [
    "ModelLoader",
    "set_deterministic_mode",
    "reset_model_state",
    "ObservationManager",
    "ImageConverter",
    "ActionLimiter",
    "MetricsTracker",
    "SharedImageBuffer",
    "SharedJointStateBuffer",
]
