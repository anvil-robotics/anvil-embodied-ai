"""Hardcoded conventions for the Anvil MCAP -> LeRobot pipeline.

Anything in this file is fixed by how we record data, not user-configurable.
Change here if the recording convention itself changes.
"""

JOINT_NAME_SEPARATOR = "_"

# Joint name prefixes. Joint names follow `{prefix}_{arm}_{joint_id}` for
# bimanual robots and `{prefix}_{joint_id}` for single-arm.
OBSERVATION_PREFIX = "follower"
LEADER_PREFIX = "leader"

# Bidirectional arm aliases used in joint names and topic names.
ARM_PREFIX_TO_NAME: dict[str, str] = {"l": "left", "r": "right"}
ARM_NAME_TO_PREFIX: dict[str, str] = {v: k for k, v in ARM_PREFIX_TO_NAME.items()}

# Quest teleop: per-arm command topics carry sensor_msgs/Float64MultiArray
# whose `data` array follows QUEST_JOINT_ORDER.
QUEST_COMMAND_TOPIC_TEMPLATE = "/follower_{arm_prefix}_forward_position_controller/commands"
QUEST_JOINT_ORDER: list[str] = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
]

# JointState field names we extract.
OBSERVATION_STATE_FIELD = "position"
ACTION_STATE_FIELD = "position"
OBSERVATION_EXTRAS: tuple[str, ...] = ("velocity", "effort")


def quest_command_topic(arm: str) -> str:
    """Return the ros2_control command topic for one arm (`left`/`right`)."""
    return QUEST_COMMAND_TOPIC_TEMPLATE.format(arm_prefix=ARM_NAME_TO_PREFIX[arm])
