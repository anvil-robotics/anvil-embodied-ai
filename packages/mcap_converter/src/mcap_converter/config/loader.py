"""Configuration loader for YAML files"""

import warnings
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .schema import DataConfig, JointNamePattern, FeatureMapping


class ConfigLoader:
    """Load and validate configuration from YAML files"""

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file

        Args:
            config_path: Path to YAML config file

        Returns:
            Dictionary with configuration values

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        return config_dict or {}

    @staticmethod
    def _parse_joint_name_pattern(pattern_dict: Optional[Dict]) -> JointNamePattern:
        """Parse joint_name_pattern / joint_names from dictionary.

        Supports both new field names (source, arms) and legacy names (role_prefix, robot_prefix).

        Args:
            pattern_dict: Dictionary with pattern configuration

        Returns:
            JointNamePattern instance
        """
        if not pattern_dict:
            return JointNamePattern()

        defaults = JointNamePattern()

        # Support both new and legacy field names
        # New: source, arms
        # Legacy: role_prefix, robot_prefix
        source = pattern_dict.get('source') or pattern_dict.get('role_prefix', defaults.source)
        arms = pattern_dict.get('arms') or pattern_dict.get('robot_prefix', defaults.arms)
        separator = pattern_dict.get('separator', defaults.separator)

        return JointNamePattern(
            source=source,
            arms=arms,
            separator=separator,
        )

    @staticmethod
    def _parse_feature_mapping(mapping_dict: Optional[Dict], default: FeatureMapping) -> FeatureMapping:
        """Parse feature_mapping from dictionary.

        Args:
            mapping_dict: Dictionary with feature mapping configuration
            default: Default FeatureMapping to use for missing values

        Returns:
            FeatureMapping instance
        """
        if not mapping_dict:
            return default

        return FeatureMapping(
            state=mapping_dict.get('state', default.state),
            others=mapping_dict.get('others', default.others),
        )

    @staticmethod
    def _migrate_legacy_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy configuration format to new format.

        Args:
            config_dict: Original configuration dictionary

        Returns:
            Migrated configuration dictionary
        """
        # Migrate robot_state_topics to robot_state_topic
        if 'robot_state_topics' in config_dict and 'robot_state_topic' not in config_dict:
            topics = config_dict['robot_state_topics']
            if topics:
                # Use first topic as the single topic
                config_dict['robot_state_topic'] = topics[0]
                warnings.warn(
                    "robot_state_topics is deprecated. Use robot_state_topic (singular) "
                    "with joint_name_pattern for role detection. "
                    f"Using first topic: {topics[0]}",
                    DeprecationWarning,
                    stacklevel=4
                )

        # Migrate motor_feature_mapping to observation/action feature mappings
        if 'motor_feature_mapping' in config_dict:
            old_mapping = config_dict['motor_feature_mapping']
            if old_mapping and 'observation_feature_mapping' not in config_dict:
                config_dict['observation_feature_mapping'] = {
                    'state': old_mapping.get('state', 'position'),
                    'others': old_mapping.get('others', []),
                }
            if old_mapping and 'action_feature_mapping' not in config_dict:
                config_dict['action_feature_mapping'] = {
                    'state': old_mapping.get('state', 'position'),
                    'others': [],  # Actions typically don't need extras
                }
            warnings.warn(
                "motor_feature_mapping is deprecated. Use observation_feature_mapping "
                "and action_feature_mapping instead.",
                DeprecationWarning,
                stacklevel=4
            )

        return config_dict

    @staticmethod
    def from_yaml(config_path: str) -> DataConfig:
        """Create DataConfig from YAML file

        Args:
            config_path: Path to YAML config file

        Returns:
            DataConfig instance with values from YAML

        Example YAML structure (new format):
            robot_state_topic: "/joint_states"
            joint_name_pattern:
              role_prefix:
                leader: "action"
                follower: "observation"
              robot_prefix:
                r: "right"
                l: "left"
              separator: "_"
            observation_feature_mapping:
              state: "position"
              others: ["velocity", "effort"]
            action_feature_mapping:
              state: "position"
              others: []
            camera_topics:
              - "/camera1/image_raw"
            camera_topic_mapping:
              "/camera1/image_raw": "head"
        """
        config_dict = ConfigLoader.load_yaml(config_path)

        # Apply legacy migration
        config_dict = ConfigLoader._migrate_legacy_config(config_dict)

        return ConfigLoader.from_dict(config_dict)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> DataConfig:
        """Create DataConfig from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            DataConfig instance
        """
        # Apply legacy migration
        config_dict = ConfigLoader._migrate_legacy_config(config_dict)

        # Get defaults
        defaults = DataConfig()

        # Parse nested configuration objects
        # Support both 'joint_names' (new) and 'joint_name_pattern' (legacy)
        joint_names_dict = config_dict.get('joint_names') or config_dict.get('joint_name_pattern')
        joint_name_pattern = ConfigLoader._parse_joint_name_pattern(joint_names_dict)

        observation_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get('observation_feature_mapping'),
            defaults.observation_feature_mapping
        )

        action_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get('action_feature_mapping'),
            defaults.action_feature_mapping
        )

        return DataConfig(
            # New fields
            robot_state_topic=config_dict.get('robot_state_topic', defaults.robot_state_topic),
            joint_name_pattern=joint_name_pattern,
            observation_feature_mapping=observation_feature_mapping,
            action_feature_mapping=action_feature_mapping,

            # Camera config
            camera_topics=config_dict.get('camera_topics', defaults.camera_topics),
            camera_topic_mapping=config_dict.get('camera_topic_mapping', defaults.camera_topic_mapping),
            image_resolution=config_dict.get('image_resolution', defaults.image_resolution),

            # Legacy fields (for backward compatibility)
            robot_state_topics=config_dict.get('robot_state_topics', defaults.robot_state_topics),
            motor_feature_mapping=config_dict.get('motor_feature_mapping', defaults.motor_feature_mapping),
        )

    @staticmethod
    def get_default() -> DataConfig:
        """Get default configuration

        Returns:
            Default DataConfig instance
        """
        return DataConfig()
