"""Runtime cross-checks for the unified mcap_converter config.

``DataConfig.validate()`` (schema.py) owns "is this config internally well-formed" —
this module now holds only ``validate_topics_exist``, which is a different kind of check:
whether a config's topic names are actually present in a specific, already-loaded MCAP
file. That's a cross-check against external runtime data, not a property of the config
alone, so it stays a standalone function rather than a ``DataConfig`` method.
"""

from typing import List

from .schema import ConfigurationError, DataConfig

__all__ = ["ConfigurationError", "validate_topics_exist"]


def validate_topics_exist(config: DataConfig, available_topics: List[str]) -> None:
    """Cross-check that observation/action/camera topics are present in the MCAP."""
    missing: List[str] = []
    seen_obs = set()

    for arm_id, topic in config.observation_topics.items():
        if topic in seen_obs:
            continue  # Shared /joint_states across arms — only check once
        seen_obs.add(topic)
        if topic not in available_topics:
            missing.append(f"observation_topics[{arm_id!r}]: {topic}")

    for arm_id, spec in config.action_topics.items():
        if spec.topic and spec.topic not in available_topics:
            missing.append(f"action_topics[{arm_id!r}].topic: {spec.topic}")

    for topic in config.camera_topics:
        if topic not in available_topics:
            missing.append(f"camera_topic: {topic}")

    if missing:
        raise ConfigurationError(
            "Topics not found in MCAP file:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nAvailable topics: {available_topics}"
        )
