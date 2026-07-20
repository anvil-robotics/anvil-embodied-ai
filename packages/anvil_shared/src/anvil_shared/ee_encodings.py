"""Encoding-related constants and dispatch for the unified EE converter config.

Single home for "what values of action_encoding/observation_encoding are legal, what do
they look like on disk, what dimension do they produce" — the concrete fix for the
"hardcoded independently in writer.py and extractor.py" pattern flagged in
``claude_docs/ee-delta-architecture-report.md``. Nothing here is EE math (that stays in
``anvil_shared.rotation``) — this module is the shared on-disk layout contract, consumed
by mcap_converter (writing), anvil_trainer (dataset-shape validation), anvil_eval, and the
ROS2 inference stack alike.

Moved here (2026-07-19) from ``mcap_converter.config.encodings`` — this table describes
the shared on-disk EE contract, not mcap-converter-private schema, so every consumer reads
the same definitions instead of guessing from feature-name suffixes.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

VALID_ACTION_ENCODINGS = ("absolute", "delta", "relative")
IMPLEMENTED_ACTION_ENCODINGS = ("absolute", "delta")

VALID_OBSERVATION_ENCODINGS = ("quaternion", "rot6d", "axis_angle")

# One row per encoding: (per-arm rotation-component feature-name suffixes, dimension).
# Position (x, y, z) and gripper are invariant across all three and are NOT part of this
# table — this table describes only the rotation component.
OBSERVATION_ROTATION_LAYOUTS: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "quaternion": (("qx", "qy", "qz", "qw"), 4),
    "rot6d": (("r0", "r1", "r2", "r3", "r4", "r5"), 6),
    "axis_angle": (("ax", "ay", "az"), 3),
}


def observation_state_dim_per_arm(observation_encoding: str) -> int:
    """3 (xyz) + rotation dim + 1 (gripper), for the given observation_encoding."""
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    return 3 + rot_dim + 1


def observation_state_names_per_arm(observation_encoding: str) -> Tuple[str, ...]:
    rot_names, _ = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    return ("x", "y", "z", *rot_names, "gripper")


def encode_rotation(quat_xyzw: Any, observation_encoding: str) -> Any:
    """Encode a single quaternion sample into the selected observation rotation encoding.

    ``quat_xyzw`` is a length-4 array-like ``[x, y, z, w]``. Returns an array of the
    encoding's rotation dimension (4 for quaternion — passthrough, 6 for rot6d, 3 for
    axis_angle).
    """
    import numpy as np

    if observation_encoding == "quaternion":
        return np.asarray(quat_xyzw, dtype=np.float64)

    from anvil_shared.rotation import matrix_to_axis_angle, matrix_to_rot6d, quat_to_matrix

    R = quat_to_matrix(quat_xyzw)
    if observation_encoding == "rot6d":
        return matrix_to_rot6d(R)
    if observation_encoding == "axis_angle":
        return matrix_to_axis_angle(R)
    raise ValueError(f"unknown observation_encoding: {observation_encoding!r}")
