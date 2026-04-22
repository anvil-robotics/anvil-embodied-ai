"""Configuration schema for MCAP to LeRobot conversion"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
class ActionTopicConfig:
    """
    Configuration for a single action command topic (quest teleop mode).

    Specifies which arm the topic controls and the explicit joint ordering
    of the Float64MultiArray.data values.

    Example:
        ActionTopicConfig(
            arm="left",
            joint_order=["joint1", "joint2", ..., "joint7", "finger_joint1"]
        )
    """

    # Arm identifier (e.g., "left", "right")
    arm: str = ""

    # Explicit joint ordering for the Float64MultiArray.data array.
    # Maps each index position to a joint_id name.
    # Example: ["joint1", "joint2", ..., "joint7", "finger_joint1"]
    #   -> data[0] = joint1, data[7] = finger_joint1
    # These names must match the joint_ids parsed from /joint_states.
    joint_order: List[str] = field(default_factory=list)


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
    joint_name_pattern: JointNamePattern = field(default_factory=JointNamePattern)

    # ====== Quest Teleop Mode ======
    # When set, actions are read from separate command topics instead of
    # from leader joints in the robot_state_topic.
    #
    # Maps ROS2 command topic -> ActionTopicConfig with arm identifier
    # and explicit joint ordering for the Float64MultiArray.data array.
    #
    # Example:
    #   {"/follower_l_forward_position_controller/commands":
    #       ActionTopicConfig(arm="left",
    #           joint_order=["joint1", ..., "joint7", "finger_joint1"])}
    #
    # If empty (default), leader-follower mode is used: actions come from
    # leader joints parsed from robot_state_topic via joint_name_pattern.
    action_topics: Dict[str, "ActionTopicConfig"] = field(default_factory=dict)

    # When True, use observation joint positions as action when action_topics
    # are configured but the topics are not present in the MCAP file.
    # Useful for datasets recorded without a separate command topic.
    action_from_observation: bool = False

    # Number of frames to look ahead when action_from_observation=True.
    # action[t] = observation[t + n]. Default: 10.
    action_from_observation_n: int = 10


    # Separate feature mappings for observation vs action
    # This allows different features for input (observation) and output (action)
    observation_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(state="position", others=[])
    )

    action_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(
            state="position",
            others=[],  # Actions typically only need position
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
