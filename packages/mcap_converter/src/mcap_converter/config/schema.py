"""Configuration schema for MCAP to LeRobot conversion"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class JointNamePattern:
    """
    Configuration for parsing joint names from a single JointState topic.

    Joint names follow the pattern: {source}_{arm}_{joint_id}

    Example joint names:
        "leader_r_joint1"   -> action data, right arm, joint1
        "follower_l_joint3" -> observation data, left arm, joint3

    The 'source' determines whether data goes to action or observation:
        - leader/master = action (target positions the robot should reach)
        - follower/puppet = observation (current robot state)

    The 'arm' identifies which arm for bimanual robots:
        - r/right = right arm
        - l/left = left arm
    """

    # Maps the first part of joint name to observation/action
    # Example: {"leader": "action", "follower": "observation"}
    #   - "leader_*" joints become action data
    #   - "follower_*" joints become observation data
    source: Dict[str, str] = field(
        default_factory=lambda: {
            "leader": "action",
            "follower": "observation",
        }
    )

    # Maps arm identifier to left/right (for bimanual robots)
    # Example: {"r": "right", "l": "left"}
    # Leave empty {} for single-arm robots
    arms: Dict[str, str] = field(
        default_factory=lambda: {
            "r": "right",
            "l": "left",
        }
    )

    # Separator between parts (default: "_")
    separator: str = "_"

    # DEPRECATED: Old field names for backward compatibility
    @property
    def role_prefix(self) -> Dict[str, str]:
        """Deprecated: Use 'source' instead."""
        return self.source

    @property
    def robot_prefix(self) -> Dict[str, str]:
        """Deprecated: Use 'arms' instead."""
        return self.arms


@dataclass
class FeatureMapping:
    """
    Configuration for extracting features from JointState.

    Allows different feature configurations for observation vs action.
    """

    # Primary field for state/action (typically "position")
    state: str = "position"

    # Additional fields to extract (e.g., ["velocity", "effort"])
    others: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """
    Manage parameters that can be dynamically adjusted during data conversion,
    such as topics, motor features, time alignment delays, etc.

    If recorder/robot settings change later, only need to adjust this config,
    without major changes to conversion program.
    """

    # Single topic for all joint states (new architecture)
    # All joints are in one JointState message, differentiated by joint names
    robot_state_topic: str = "/joint_states"

    # Joint name parsing configuration
    joint_name_pattern: JointNamePattern = field(
        default_factory=JointNamePattern
    )

    # Separate feature mappings for observation vs action
    # This allows different features for input (observation) and output (action)
    observation_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(
            state="position",
            others=["velocity", "effort"]
        )
    )

    action_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(
            state="position",
            others=[]  # Actions typically only need position
        )
    )

    # Camera ROS topics
    camera_topics: List[str] = field(
        default_factory=lambda: [
            "/camera1/image_raw",
        ]
    )

    # Mapping camera topics to dataset camera names
    camera_topic_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "/camera1/image_raw": "head",
        }
    )

    # Image resolution configuration
    # Target resolution for resizing images before adding to dataset
    # Format: [width, height]
    image_resolution: List[int] = field(
        default_factory=lambda: [640, 480]  # [width, height]
    )

    # ========== DEPRECATED FIELDS (for backward compatibility) ==========

    # DEPRECATED: Use robot_state_topic (singular) with joint_name_pattern
    robot_state_topics: List[str] = field(default_factory=list)

    # DEPRECATED: Use observation_feature_mapping and action_feature_mapping
    motor_feature_mapping: Dict[str, Any] = field(default_factory=dict)


# Default configuration
DEFAULT_DATA_CONFIG = DataConfig()
