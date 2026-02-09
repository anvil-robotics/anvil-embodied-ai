"""Configuration validation for mcap_converter package."""

import warnings
from typing import List

from .schema import DataConfig, FeatureMapping, JointNamePattern


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def validate_joint_name_pattern(
    pattern: JointNamePattern,
    skip_action_role: bool = False,
) -> List[str]:
    """
    Validate joint name pattern configuration.

    Args:
        pattern: JointNamePattern instance to validate
        skip_action_role: If True, don't require an 'action' mapping
            (used in quest_teleop mode where actions come from separate topics)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Must have at least one role mapping
    if not pattern.role_prefix:
        errors.append("joint_name_pattern.role_prefix cannot be empty")
    else:
        # Validate role values
        valid_roles = {"observation", "action"}
        for prefix, role in pattern.role_prefix.items():
            if role not in valid_roles:
                errors.append(
                    f"joint_name_pattern: Invalid role '{role}' for prefix '{prefix}'. "
                    f"Must be 'observation' or 'action'"
                )

        # Should have both observation and action roles
        roles = set(pattern.role_prefix.values())
        if "observation" not in roles:
            errors.append("joint_name_pattern.role_prefix must include an 'observation' mapping")
        if not skip_action_role and "action" not in roles:
            errors.append("joint_name_pattern.role_prefix must include an 'action' mapping")

    # Separator should not be empty
    if not pattern.separator:
        errors.append("joint_name_pattern.separator cannot be empty")

    return errors


def validate_feature_mapping(mapping: FeatureMapping, name: str) -> List[str]:
    """
    Validate feature mapping configuration.

    Args:
        mapping: FeatureMapping instance to validate
        name: Name of the mapping for error messages

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    valid_fields = {"position", "velocity", "effort"}

    # State field must be valid
    if not mapping.state:
        errors.append(f"{name}.state cannot be empty")
    elif mapping.state not in valid_fields:
        errors.append(f"{name}.state '{mapping.state}' is not a valid JointState field")

    # Others must be valid fields
    for field in mapping.others:
        if field not in valid_fields:
            errors.append(f"{name}.others contains invalid field '{field}'")

    return errors


def validate_config(config: DataConfig) -> None:
    """
    Validate configuration completeness and correctness.

    Args:
        config: DataConfig instance to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    errors: List[str] = []

    # Check for deprecated fields and warn
    if config.robot_state_topics:
        warnings.warn(
            "robot_state_topics is deprecated. Use robot_state_topic (singular) "
            "with joint_name_pattern for role detection.",
            DeprecationWarning,
            stacklevel=2,
        )

    if config.motor_feature_mapping:
        warnings.warn(
            "motor_feature_mapping is deprecated. Use observation_feature_mapping "
            "and action_feature_mapping instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Validate control_mode
    valid_control_modes = {"leader_follower", "quest_teleop"}
    if config.control_mode not in valid_control_modes:
        errors.append(
            f"control_mode '{config.control_mode}' is invalid. "
            f"Must be one of: {sorted(valid_control_modes)}"
        )

    # Validate action_topics when quest_teleop
    if config.control_mode == "quest_teleop" and not config.action_topics:
        errors.append("action_topics must be non-empty when control_mode is 'quest_teleop'")

    # Validate robot_state_topic (new single topic)
    if not config.robot_state_topic:
        errors.append("robot_state_topic cannot be empty")

    # Validate joint_name_pattern
    # In quest_teleop mode, action role mapping in joint names is irrelevant
    skip_action_role = config.control_mode == "quest_teleop"
    errors.extend(
        validate_joint_name_pattern(config.joint_name_pattern, skip_action_role=skip_action_role)
    )

    # Validate feature mappings
    errors.extend(
        validate_feature_mapping(config.observation_feature_mapping, "observation_feature_mapping")
    )
    errors.extend(validate_feature_mapping(config.action_feature_mapping, "action_feature_mapping"))

    # Validate camera_topics
    if not config.camera_topics:
        errors.append("camera_topics cannot be empty")

    # Validate camera_topic_mapping
    if not config.camera_topic_mapping:
        errors.append("camera_topic_mapping cannot be empty")
    else:
        # Check all camera topics have mappings
        for topic in config.camera_topics:
            if topic not in config.camera_topic_mapping:
                errors.append(f"camera topic '{topic}' missing from camera_topic_mapping")

    # Validate image_resolution
    if not config.image_resolution or len(config.image_resolution) != 2:
        errors.append("image_resolution must be [width, height]")
    elif any(dim <= 0 for dim in config.image_resolution):
        errors.append("image_resolution dimensions must be positive")

    if errors:
        raise ConfigurationError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


def validate_topics_exist(config: DataConfig, available_topics: List[str]) -> None:
    """
    Validate that configured topics exist in the MCAP file.

    Args:
        config: DataConfig instance
        available_topics: List of topics available in the MCAP file

    Raises:
        ConfigurationError: If required topics are not available
    """
    missing = []

    # Check single robot_state_topic (new architecture)
    if config.robot_state_topic and config.robot_state_topic not in available_topics:
        missing.append(f"robot_state_topic: {config.robot_state_topic}")

    # Legacy support: check robot_state_topics if used
    if config.robot_state_topics:
        for topic in config.robot_state_topics:
            if topic not in available_topics:
                missing.append(f"robot_state_topic: {topic}")

    # Check action topics (quest_teleop mode)
    if config.control_mode == "quest_teleop":
        for topic in config.action_topics:
            if topic not in available_topics:
                missing.append(f"action_topic: {topic}")

    # Check camera topics
    for topic in config.camera_topics:
        if topic not in available_topics:
            missing.append(f"camera_topic: {topic}")

    if missing:
        raise ConfigurationError(
            "Topics not found in MCAP file:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nAvailable topics: {available_topics}"
        )
