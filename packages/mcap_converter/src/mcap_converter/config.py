"""User-facing conversion config (DataConfig) + YAML loader + topic validator."""

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from .core.constants import quest_command_topic
from .exceptions import ConfigurationError


class ActionSource(str, Enum):
    """How action signals are produced for the output dataset."""

    leader = "leader"
    """Leader-follower teleop. Action joints (`leader_*`) live in the same
    JointState topic as observations (`follower_*`)."""

    quest_teleop = "quest_teleop"
    """Quest teleop. Actions arrive on separate `Float64MultiArray` command
    topics — one per arm."""

    future_observations = "future_observations"
    """No recorded actions. Action at time t is synthesized as the observation
    at t + `action_n_step`."""


Arm = Literal["left", "right"]


class DataConfig(BaseModel):
    """User-facing conversion config."""

    robot_state_topic: str = "/joint_states"

    frequency: int = 60
    """Output dataset sample rate (Hz). The CLI uses this as the target rate;
    if the source MCAP rate is lower, the CLI clamps down to the source rate
    (we can't upsample). The `--frequency` CLI flag overrides this value."""

    camera_topic_mapping: dict[str, str]
    """ROS topic -> output camera name (becomes `observation.images.{name}`).
    Topics are assumed to carry `sensor_msgs/CompressedImage`."""

    image_resolution: tuple[int, int] = (640, 480)

    action_source: ActionSource

    arms: list[Arm]
    """Physical arms the robot has. Used to build command topic names for
    `quest_teleop` and to drive per-arm observation grouping everywhere else.
    Use `["left"]` or `["right"]` for single-arm robots."""

    action_n_step: int | None = None
    """Lookahead in frames when `action_source == future_observations`.
    `action[t] = observation[t + action_n_step]`."""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate(self) -> "DataConfig":
        if not self.arms:
            raise ValueError("`arms` must contain at least one arm")
        if len(self.arms) != len(set(self.arms)):
            raise ValueError(f"`arms` must not contain duplicates: {self.arms}")
        if self.action_source is ActionSource.future_observations:
            if self.action_n_step is None or self.action_n_step <= 0:
                raise ValueError(
                    "`action_n_step` (positive int) is required when "
                    "action_source == future_observations"
                )
        elif self.action_n_step is not None:
            raise ValueError(
                "`action_n_step` may only be set when "
                "action_source == future_observations"
            )
        return self


def load_config(path: str | Path) -> DataConfig:
    """Load and validate a DataConfig from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    return DataConfig.model_validate(yaml.safe_load(p.read_text()) or {})


def validate_topics_exist(config: DataConfig, available_topics: Iterable[str]) -> None:
    """Raise ConfigurationError if any configured topic is missing from the MCAP."""
    expected = {config.robot_state_topic, *config.camera_topic_mapping}
    if config.action_source is ActionSource.quest_teleop:
        expected.update(quest_command_topic(a) for a in config.arms)
    available = set(available_topics)
    missing = sorted(expected - available)
    if missing:
        raise ConfigurationError(
            "Topics not found in MCAP file:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nAvailable topics: {sorted(available)}"
        )
