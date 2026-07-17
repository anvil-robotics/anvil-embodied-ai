"""Configuration loader for the unified YAML format.

Schema (joint and EE share the same top-level keys; ``data_space`` switches
encoding only)::

    schema_version: "1.1"              # optional; absence means v1.0 — see versioning.py
                                        # for exactly what v1.0/v1.1 mean (NOT "whatever
                                        # main's schema is" — verify there before assuming)
    data_space: "joint" | "ee"
    observation_topics:
      <arm_id>: <topic>
      ...
    action_topics:
      <arm_id>:
        topic: <topic>
        joint_order: [...]
      ...                              # empty in EE mode
    action_encoding: "absolute" | "delta" | "relative"   # EE mode only
    observation_encoding: "quaternion" | "rot6d" | "axis_angle"  # EE mode only
    joint_names:                       # joint mode only
      separator: "_"
      source:  { follower: observation }
      arms:    { l: left, r: right }
    observation_feature_mapping:       # new configs use others: []
      state: position
      others: []
    camera_topics: [...]
    camera_topic_mapping: { <topic>: <name>, ... }
    image_resolution: [W, H]

Legacy formats (singular ``robot_state_topic`` field, topic-keyed
``action_topics``, ``robot_state_topics`` plural, ``motor_feature_mapping``,
``action_from_observation*``) are no longer accepted.

``ConfigLoader`` only hydrates YAML into typed :class:`DataConfig` fields — it does not
decide whether a (well-shaped) value is *legal*; that's :meth:`DataConfig.validate`'s job.
It does, however, run schema-version migration (see
``mcap_converter.config.versioning``) unconditionally, and — in ``strict`` mode — reject
top-level keys it doesn't recognize. See ``claude_docs/mcap-converter-encoding-refactor-plan.md``
Part 0 / Part 0b for the full design rationale, including why ``strict=False`` exists (for
reading an already-converted dataset's frozen ``conversion_config.yaml``, where demanding
full current-schema legality would make old datasets unreadable rather than more robust).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import (
    ActionTopicSpec,
    ConfigurationError,
    DataConfig,
    FeatureMapping,
    JointNamePattern,
    RECOGNIZED_YAML_KEYS,
)
from .versioning import migrate_to_current


def _reject_unknown_keys(config_dict: Dict[str, Any]) -> None:
    unknown = set(config_dict.keys()) - RECOGNIZED_YAML_KEYS
    if unknown:
        raise ConfigurationError(
            "Unrecognized configuration key(s): " + ", ".join(sorted(unknown))
            + ". If this is an old-format config, run `dataset-config-migrate` on it, "
              "or pass strict=False to read it as a historical record."
        )


class ConfigLoader:
    """Load configuration from YAML / dict into a :class:`DataConfig`, and back."""

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def from_yaml(config_path: str, strict: bool = True) -> DataConfig:
        return ConfigLoader.from_dict(ConfigLoader.load_yaml(config_path), strict=strict)

    @staticmethod
    def to_yaml(config: DataConfig, path: str) -> None:
        """Write ``config`` to ``path`` as a canonical current-schema-version YAML file."""
        with open(path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def get_default() -> DataConfig:
        return DataConfig()

    # ------------------------------------------------------------------
    # New unified format parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_joint_name_pattern(pattern_dict: Optional[Dict]) -> JointNamePattern:
        if not pattern_dict:
            return JointNamePattern()
        defaults = JointNamePattern()
        return JointNamePattern(
            source=pattern_dict.get("source", defaults.source),
            arms=pattern_dict.get("arms", defaults.arms),
            separator=pattern_dict.get("separator", defaults.separator),
        )

    @staticmethod
    def _parse_observation_topics(value: Any) -> Dict[str, str]:
        if not value:
            return {}
        if not isinstance(value, dict):
            raise ConfigurationError(
                "observation_topics must be a mapping of arm_id -> topic, "
                f"got {type(value).__name__}"
            )
        out: Dict[str, str] = {}
        for arm_id, topic in value.items():
            if not isinstance(topic, str) or not topic:
                raise ConfigurationError(
                    f"observation_topics[{arm_id!r}] must be a non-empty topic string"
                )
            out[str(arm_id)] = topic
        return out

    @staticmethod
    def _parse_action_topics(value: Any) -> Dict[str, ActionTopicSpec]:
        """Parse ``action_topics: { arm_id: { topic, joint_order } }``.

        Empty / missing → ``{}`` (EE mode, or joint "act-from-obs" opt-in).
        """
        if not value:
            return {}
        if not isinstance(value, dict):
            raise ConfigurationError(
                "action_topics must be a mapping of arm_id -> { topic, joint_order }, "
                f"got {type(value).__name__}"
            )
        out: Dict[str, ActionTopicSpec] = {}
        for arm_id, spec in value.items():
            if not isinstance(spec, dict):
                raise ConfigurationError(
                    f"action_topics[{arm_id!r}] must be a mapping with keys "
                    f"'topic' and 'joint_order'; got {type(spec).__name__}"
                )
            topic = spec.get("topic", "")
            if not topic or not isinstance(topic, str):
                raise ConfigurationError(
                    f"action_topics[{arm_id!r}].topic must be a non-empty string"
                )
            joint_order = list(spec.get("joint_order") or [])
            out[str(arm_id)] = ActionTopicSpec(topic=topic, joint_order=joint_order)
        return out

    @staticmethod
    def _parse_feature_mapping(
        mapping_dict: Optional[Dict], default: FeatureMapping
    ) -> FeatureMapping:
        if not mapping_dict:
            return default
        return FeatureMapping(
            state=mapping_dict.get("state", default.state),
            others=list(mapping_dict.get("others", default.others)),
        )

    @staticmethod
    def from_dict(config_dict: Dict[str, Any], strict: bool = True) -> DataConfig:
        # Always runs first, in both strict and lenient modes — migration owns
        # version-tracked field renames; strict/lenient governs everything else
        # (see the Gap 1 boundary in the design doc). By the time this returns,
        # config_dict is at CURRENT_SCHEMA_VERSION.
        config_dict = migrate_to_current(config_dict)

        if strict:
            _reject_unknown_keys(config_dict)

        defaults = DataConfig()

        data_space = str(config_dict.get("data_space", defaults.data_space))
        if data_space not in ("joint", "ee"):
            raise ConfigurationError(
                f"data_space must be 'joint' or 'ee'; got {data_space!r}"
            )

        action_encoding = str(
            config_dict.get("action_encoding", defaults.action_encoding)
        )

        observation_encoding = str(
            config_dict.get("observation_encoding", defaults.observation_encoding)
        )

        observation_topics = ConfigLoader._parse_observation_topics(
            config_dict.get("observation_topics")
        )
        action_topics = ConfigLoader._parse_action_topics(
            config_dict.get("action_topics")
        )

        joint_name_pattern = ConfigLoader._parse_joint_name_pattern(
            config_dict.get("joint_names") or config_dict.get("joint_name_pattern")
        )

        observation_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get("observation_feature_mapping"),
            defaults.observation_feature_mapping,
        )
        action_feature_mapping = ConfigLoader._parse_feature_mapping(
            config_dict.get("action_feature_mapping"),
            defaults.action_feature_mapping,
        )

        return DataConfig(
            schema_version=str(config_dict.get("schema_version", defaults.schema_version)),
            data_space=data_space,
            action_encoding=action_encoding,
            observation_encoding=observation_encoding,
            observation_topics=observation_topics,
            action_topics=action_topics,
            joint_name_pattern=joint_name_pattern,
            observation_feature_mapping=observation_feature_mapping,
            action_feature_mapping=action_feature_mapping,
            camera_topics=list(config_dict.get("camera_topics") or defaults.camera_topics),
            camera_topic_mapping=dict(
                config_dict.get("camera_topic_mapping") or defaults.camera_topic_mapping
            ),
            image_resolution=list(
                config_dict.get("image_resolution") or defaults.image_resolution
            ),
        )
