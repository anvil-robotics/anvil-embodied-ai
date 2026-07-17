"""Configuration schema for MCAP to LeRobot conversion.

The unified config (introduced with EE-space support) has a single shape that
works for joint and EE modes:

    data_space:         "joint" | "ee"
    observation_topics: { arm_id -> topic }
    action_topics:      { arm_id -> ActionTopicSpec }   # empty in EE mode
    joint_names:        JointNamePattern (joint mode only — splits /joint_states by arm)
    camera_topics:      [...]
    camera_topic_mapping: { topic -> dataset_camera_name }
    image_resolution:   [W, H]

The legacy joint-extraction code reads ``robot_state_topic`` (single string) and
``action_command_topics`` (``{topic -> ActionTopicConfig}``). Both are exposed
as ``@property`` derivations over the new fields so the joint extractor stays
byte-identical apart from the attribute name.

``DataConfig`` is the primary, self-validating entity for this schema (see
:meth:`DataConfig.validate`) — :class:`~mcap_converter.config.loader.ConfigLoader` only
hydrates YAML into typed fields (plus schema-version migration and, in strict mode,
unrecognized-key rejection); it does not itself decide whether a value is legal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .encodings import (
    IMPLEMENTED_ACTION_ENCODINGS,
    VALID_ACTION_ENCODINGS,
    VALID_OBSERVATION_ENCODINGS,
)

# Current schema version. Defined here (not in versioning.py) because it's DataConfig's
# own field default — schema.py is the primary, authoritative entity for this schema's
# identity, and every other module that needs to know "what version is current"
# (versioning.py's migration registry, loader.py, the migrate-config CLI) imports it
# FROM here, not the other way around. This also means versioning.py can import
# schema.py at module level with no circular-import workaround needed (see versioning.py).
CURRENT_SCHEMA_VERSION = "1.1"

# Every top-level YAML key ConfigLoader recognizes for the CURRENT schema, including
# accepted aliases (``joint_names`` is an alias for the ``joint_name_pattern`` field).
# Lives here, not in loader.py, for the same reason as CURRENT_SCHEMA_VERSION above: this
# is a direct description of DataConfig's own field set, and schema.py is the one place
# that should own it. Must be maintained by hand — it is YAML-key-level, not derivable
# from ``dataclasses.fields(DataConfig)`` alone, since some fields accept more than one
# spelling.
RECOGNIZED_YAML_KEYS = frozenset({
    "schema_version",
    "data_space",
    "observation_topics",
    "action_topics",
    "action_encoding",
    "observation_encoding",
    "joint_names",
    "joint_name_pattern",
    "observation_feature_mapping",
    "action_feature_mapping",
    "camera_topics",
    "camera_topic_mapping",
    "image_resolution",
})


class ConfigurationError(Exception):
    """Raised when configuration is invalid.

    Lives here (not in validators.py) because ``DataConfig.validate()`` is the primary
    raiser of it now, and ``versioning.migrate_to_current`` also raises it for an
    unmigratable schema version — both live in this package's "config authority" layer.
    ``validators.py`` (now holding only ``validate_topics_exist``) imports it FROM here.
    """


def validate_joint_name_pattern(pattern: "JointNamePattern") -> List[str]:
    """Joint mode: source/arms/separator must be coherent."""
    errors: List[str] = []
    if not pattern.source:
        errors.append("joint_names.source cannot be empty")
    else:
        for prefix, role in pattern.source.items():
            if role not in ("observation", "action"):
                errors.append(
                    f"joint_names.source: prefix {prefix!r} maps to {role!r}; "
                    "must be 'observation' or 'action'"
                )
        if "observation" not in set(pattern.source.values()):
            errors.append("joint_names.source must include an 'observation' mapping")
    if not pattern.separator:
        errors.append("joint_names.separator cannot be empty")
    return errors


def validate_feature_mapping(mapping: "FeatureMapping", name: str) -> List[str]:
    errors: List[str] = []
    valid_fields = {"position", "velocity", "effort"}
    if not mapping.state:
        errors.append(f"{name}.state cannot be empty")
    elif mapping.state not in valid_fields:
        errors.append(f"{name}.state {mapping.state!r} is not a valid JointState field")
    for f in mapping.others:
        if f not in valid_fields:
            errors.append(f"{name}.others contains invalid field {f!r}")
    return errors


@dataclass
class JointNamePattern:
    """Parsing rules for joint names inside a shared /joint_states topic.

    Joint names follow ``{source}{separator}{arm}{separator}{joint_id}``::

        "follower_l_joint1" -> observation, left arm, joint1

    Only ``source`` and ``arms`` mappings are required. The ``source`` map
    classifies a name prefix as either ``observation`` or ``action`` (the
    latter only relevant in leader-follower mode, which is not used by the
    new unified configs). ``arms`` maps the per-arm identifier letter to its
    canonical name.
    """

    source: Dict[str, str] = field(
        default_factory=lambda: {
            "leader": "action",
            "follower": "observation",
        }
    )
    arms: Dict[str, str] = field(
        default_factory=lambda: {
            "r": "right",
            "l": "left",
        }
    )
    separator: str = "_"

    @property
    def role_prefix(self) -> Dict[str, str]:
        """Alias kept for joint-extractor code that reads `role_prefix`."""
        return self.source

    @property
    def robot_prefix(self) -> Dict[str, str]:
        """Alias kept for joint-extractor code that reads `robot_prefix`."""
        return self.arms


@dataclass
class ActionTopicConfig:
    """Internal/legacy view of an action command topic, keyed by topic name.

    Returned by :pyattr:`DataConfig.action_command_topics` so the joint
    extractor methods can keep reading ``.arm`` / ``.joint_order``.
    """

    arm: str = ""
    joint_order: List[str] = field(default_factory=list)


@dataclass
class ActionTopicSpec:
    """User-facing per-arm action source.

    The new YAML format is ``action_topics: { arm_id: ActionTopicSpec }``. In
    EE mode this whole map is empty; in joint mode each entry provides the
    Float64MultiArray command topic and the joint ordering that maps the
    flat data array to canonical joint slots.
    """

    topic: str = ""
    joint_order: List[str] = field(default_factory=list)


@dataclass
class FeatureMapping:
    """Selects which JointState fields to extract for a given role.

    New unified configs set ``others: []`` for both observation and action;
    velocity/effort are dropped going forward.
    """

    state: str = "position"
    others: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Unified converter config.

    ``data_space`` is the only switch between joint and EE conversion paths.
    Arm scope is determined entirely by the keys of ``observation_topics`` —
    there is no separate ``arms`` block. Insertion order of
    ``observation_topics`` defines the per-arm concatenation order in the
    output ``observation.state`` / ``action`` features.
    """

    schema_version: str = CURRENT_SCHEMA_VERSION

    data_space: str = "joint"
    observation_topics: Dict[str, str] = field(default_factory=dict)
    action_topics: Dict[str, ActionTopicSpec] = field(default_factory=dict)

    # EE mode only: "absolute" (default, byte-identical to pre-existing behavior) writes
    # the action column as the absolute EE pose (rot6d), same as observation.state's pose
    # just re-encoded. "delta" bakes a per-frame Delta(n-(n-1)) target instead:
    # action[t] = ee_delta_forward(pose[t], pose[t-1]) — world-frame, computed once at
    # convert time, never recomputed during training. "relative" is reserved for future use
    # and is not yet implemented (see IMPLEMENTED_ACTION_ENCODINGS / validate()).
    # observation.state is unaffected by this flag either way. Deliberately a scalar field
    # on the existing "ee" data_space, not a new data_space value, so is_ee and every
    # existing EE branch site stay untouched. Named without an "ee_" prefix (renamed from
    # ee_action_encoding in schema v1.1) specifically so joint-space encoding support, if
    # designed later, needs only to loosen validate()'s EE-only restriction below — no
    # second rename.
    action_encoding: str = "absolute"

    # EE mode only: rotation representation written to observation.state on disk.
    # "quaternion" (default, byte-identical to pre-existing behavior) — [x,y,z,qx,qy,qz,qw,gripper].
    # "rot6d" — [x,y,z,r0..r5,gripper]. "axis_angle" — [x,y,z,ax,ay,az,gripper]. Independent
    # of action_encoding — action is always written as rot6d regardless of this field; only
    # observation.state's rotation component varies. See config/encodings.py for the
    # per-encoding layout table both writer.py and extractor.py read from.
    observation_encoding: str = "quaternion"

    joint_name_pattern: JointNamePattern = field(default_factory=JointNamePattern)

    observation_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(state="position", others=[])
    )
    action_feature_mapping: FeatureMapping = field(
        default_factory=lambda: FeatureMapping(state="position", others=[])
    )

    camera_topics: List[str] = field(default_factory=list)
    camera_topic_mapping: Dict[str, str] = field(default_factory=dict)
    image_resolution: List[int] = field(default_factory=lambda: [640, 480])

    # Kept defaulted-empty so the legacy DataExtractor batch class doesn't
    # blow up at import time. Never populated by the new loader.
    robot_state_topics: List[str] = field(default_factory=list)
    motor_feature_mapping: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties (keep joint-extraction code byte-identical)
    # ------------------------------------------------------------------

    @property
    def is_ee(self) -> bool:
        return self.data_space == "ee"

    @property
    def is_action_delta(self) -> bool:
        """True when EE mode bakes a per-frame Delta(n-(n-1)) action at convert time.

        Renamed from is_ee_delta for consistency with the renamed action_encoding field.
        """
        return self.data_space == "ee" and self.action_encoding == "delta"

    @property
    def output_subdir(self) -> str:
        """Canonical <space>-space/ output directory name for this config.

        Single source of truth for convert.py's output path. A future action_encoding
        value that needs its own subdirectory is a one-line addition HERE, not a new
        branch in convert.py.
        """
        if self.is_action_delta:
            return "ee-delta-space"
        return f"{self.data_space}-space"

    @property
    def arms(self) -> List[str]:
        """Arms in concatenation order, taken from observation_topics insertion order."""
        return list(self.observation_topics.keys())

    @property
    def robot_state_topic(self) -> str:
        """Single distinct value of observation_topics.

        Joint mode always shares ``/joint_states`` across arms, so this
        collapses to one topic. Raises if multiple distinct values were
        listed (which would be nonsensical for the joint path). Returns the
        empty string when observation_topics is empty.
        """
        topics = set(self.observation_topics.values())
        if not topics:
            return ""
        if len(topics) > 1:
            raise ValueError(
                "observation_topics has multiple distinct values "
                f"{sorted(topics)}; robot_state_topic is undefined."
            )
        return next(iter(topics))

    @property
    def action_command_topics(self) -> Dict[str, ActionTopicConfig]:
        """``{topic -> ActionTopicConfig(arm, joint_order)}`` for joint-path code.

        Inverts the new per-arm ``action_topics`` map into the topic-keyed
        shape the joint extractor methods consume. Empty in EE mode and in
        the joint "act-from-obs" opt-in (empty ``action_topics``).
        """
        out: Dict[str, ActionTopicConfig] = {}
        for arm_id, spec in self.action_topics.items():
            if not spec.topic:
                continue
            out[spec.topic] = ActionTopicConfig(arm=arm_id, joint_order=list(spec.joint_order))
        return out

    # ------------------------------------------------------------------
    # Validation (moved in from validators.py — DataConfig is the primary,
    # self-validating entity; ConfigLoader only hydrates)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raises :class:`ConfigurationError` when this config is malformed.

        Deliberately NOT auto-invoked from ``__post_init__`` — several existing tests
        construct a deliberately invalid ``DataConfig`` and then assert this method catches
        it; auto-validating at construction time would prevent building that object in the
        first place. Callers (``convert.py``) call this explicitly right after construction,
        same as they called the old free-function ``validate_config(config)``.
        """
        errors: List[str] = []

        if self.data_space not in ("joint", "ee"):
            errors.append(f"data_space must be 'joint' or 'ee'; got {self.data_space!r}")

        if not self.observation_topics:
            errors.append(
                "observation_topics cannot be empty; list one topic per arm "
                "(e.g. { left: /joint_states, right: /joint_states })."
            )

        if self.data_space == "joint":
            try:
                _ = self.robot_state_topic
            except ValueError as exc:
                errors.append(str(exc))

            errors.extend(validate_joint_name_pattern(self.joint_name_pattern))

            for arm_id, spec in self.action_topics.items():
                if not spec.topic:
                    errors.append(f"action_topics[{arm_id!r}].topic cannot be empty")
                if not spec.joint_order:
                    errors.append(
                        f"action_topics[{arm_id!r}].joint_order cannot be empty in joint "
                        "mode; specify the ordered joint list matching "
                        "Float64MultiArray.data."
                    )
                if arm_id not in self.observation_topics:
                    errors.append(
                        f"action_topics[{arm_id!r}] references an arm not present in "
                        "observation_topics"
                    )

        if self.data_space == "ee":
            if self.action_topics:
                errors.append(
                    "action_topics must be empty in ee mode; the EE action is derived "
                    "from the same /ee_pose_<arm> topics listed in observation_topics."
                )
            if self.action_encoding not in VALID_ACTION_ENCODINGS:
                errors.append(
                    f"action_encoding must be one of {VALID_ACTION_ENCODINGS}; got "
                    f"{self.action_encoding!r}"
                )
            elif self.action_encoding not in IMPLEMENTED_ACTION_ENCODINGS:
                errors.append(
                    f"action_encoding={self.action_encoding!r} is reserved for future use "
                    "and is not yet implemented in mcap_converter."
                )
            if self.observation_encoding not in VALID_OBSERVATION_ENCODINGS:
                errors.append(
                    f"observation_encoding must be one of {VALID_OBSERVATION_ENCODINGS}; "
                    f"got {self.observation_encoding!r}"
                )
        else:
            if self.action_encoding != "absolute":
                errors.append(
                    "action_encoding is only meaningful when data_space == 'ee'; got "
                    f"data_space={self.data_space!r} with action_encoding="
                    f"{self.action_encoding!r}"
                )
            if self.observation_encoding != "quaternion":
                errors.append(
                    "observation_encoding is only meaningful when data_space == 'ee'; got "
                    f"data_space={self.data_space!r} with observation_encoding="
                    f"{self.observation_encoding!r}"
                )

        errors.extend(validate_feature_mapping(
            self.observation_feature_mapping, "observation_feature_mapping"
        ))
        errors.extend(validate_feature_mapping(
            self.action_feature_mapping, "action_feature_mapping"
        ))

        if not self.camera_topics:
            errors.append("camera_topics cannot be empty")
        if not self.camera_topic_mapping:
            errors.append("camera_topic_mapping cannot be empty")
        else:
            for t in self.camera_topics:
                if t not in self.camera_topic_mapping:
                    errors.append(f"camera topic {t!r} missing from camera_topic_mapping")

        if not self.image_resolution or len(self.image_resolution) != 2:
            errors.append("image_resolution must be [width, height]")
        elif any(dim <= 0 for dim in self.image_resolution):
            errors.append("image_resolution dimensions must be positive")

        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Canonical current-schema-version YAML-serializable representation.

        Single source of truth for writing a conversion_config.yaml — replaces convert.py's
        previous ad hoc dict construction, which never included the action-encoding field
        at all (a real, now-fixed rough edge).
        """
        out: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "data_space": self.data_space,
            "observation_topics": dict(self.observation_topics),
            "action_topics": {
                arm_id: {"topic": spec.topic, "joint_order": list(spec.joint_order)}
                for arm_id, spec in self.action_topics.items()
            },
            "action_encoding": self.action_encoding,
            "observation_encoding": self.observation_encoding,
            "camera_topics": list(self.camera_topics),
            "camera_topic_mapping": dict(self.camera_topic_mapping),
            "image_resolution": list(self.image_resolution),
        }
        if not self.is_ee:  # joint_names is joint-mode-only; omit from EE configs
            out["joint_names"] = {
                "separator": self.joint_name_pattern.separator,
                "source": self.joint_name_pattern.source,
                "arms": self.joint_name_pattern.arms,
            }
        return out


# Default configuration (empty maps — convert.py always supplies a real config).
DEFAULT_DATA_CONFIG = DataConfig()
