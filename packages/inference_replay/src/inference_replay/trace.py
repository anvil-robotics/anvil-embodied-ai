"""Load and validate an `inference_data.csv` monitor trace.

The channel layout and the alignment gate here are ported from `scripts/build_trace_replay.py`
rather than re-derived. That script's assertions are the only thing standing between a
mis-permuted channel order and a replay that looks plausible but is wrong, so they are kept
as hard failures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_CHANNELS = 16

# `obs_state` and `control_cmd` are logged in controller order (joint1..7, finger) while
# `raw_output` is in model order (finger, joint1..7). Indexing obs/cmd with this permutation
# puts all three in model order, which is what NAMES describes.
MODEL_TO_CTRL = np.array([7, 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14])

# Channel names in model order. These are the monitor's own names, kept verbatim so output
# can be diffed against build_trace_replay.py.
NAMES: list[str] = (
    ["left_finger_joint1"]
    + [f"left_joint{i}" for i in range(1, 8)]
    + ["right_finger_joint1"]
    + [f"right_joint{i}" for i in range(1, 8)]
)

# The monitor calls the arms left/right; the URDF (generated from the workcell's
# teleop.xacro) calls them follower_l/follower_r. Same joints, different vocabulary.
_URDF_PREFIX = {"left_": "follower_l_", "right_": "follower_r_"}


def _to_urdf_joint(name: str) -> str:
    for src, dst in _URDF_PREFIX.items():
        if name.startswith(src):
            return dst + name[len(src) :]
    raise ValueError(f"channel name {name!r} has no known arm prefix")


# Channel index -> URDF joint name, in model order.
URDF_JOINT_NAMES: list[str] = [_to_urdf_joint(name) for name in NAMES]

# Gripper channels in model order. Excluded from stall detection: a gripper legitimately sits
# still against an object while commanded further closed, which is not a fault.
GRIPPER_CHANNELS = (0, 8)
ARM_CHANNELS: list[int] = [i for i in range(N_CHANNELS) if i not in GRIPPER_CHANNELS]

# Largest believable gripper command deviation, in metres. The URDF's finger joints are
# prismatic over 0.0 -> 0.05, so a full open from fully closed is a legitimate 0.05 deviation:
# anything at or just above the stroke is normal operation, not a channel mix-up. Three times
# the stroke still sits an order of magnitude below the radian-scale deviations an arm channel
# would produce here (its joints span ~2-3 rad), so the guard keeps its margin.
_MAX_GRIPPER_DEVIATION = 0.15

# Action types whose control_cmd is an absolute joint target. Delta types describe motion
# relative to a moving reference and would need different handling to reconstruct poses.
_ABSOLUTE_ACTION_TYPES = frozenset({"absolute", "joint_abs"})


class TraceAlignmentError(Exception):
    """The CSV's channels do not line up the way the monitor is documented to write them."""


@dataclass(frozen=True)
class Trace:
    """One inference run, with all three signals in model order (see NAMES)."""

    path: Path
    action_type: str
    ts: np.ndarray  # (n,) raw monotonic timestamps, seconds
    rel: np.ndarray  # (n,) seconds since the first sample
    raw: np.ndarray  # (n, 16) policy output
    obs: np.ndarray  # (n, 16) measured joint positions
    cmd: np.ndarray  # (n, 16) positions actually commanded
    cap: float  # action limiter's per-step clamp, inferred from the data
    residual: float  # how exactly the clamp identity held; near zero means aligned

    @property
    def n(self) -> int:
        return len(self.ts)

    @property
    def duration_sec(self) -> float:
        return float(self.rel[-1]) if self.n else 0.0

    @property
    def hz(self) -> float:
        return (self.n - 1) / self.duration_sec if self.duration_sec > 0 else 0.0


def _read_metadata(path: Path) -> dict[str, str]:
    """Parse the `# key: value` comment lines the monitor writes above the header."""
    meta: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            meta[key.strip()] = value.strip()
    return meta


def load_trace(path: Path) -> Trace:
    """Read an inference_data.csv, validate channel alignment, and return it in model order.

    Raises TraceAlignmentError if the file's action type is not absolute or if the channel
    permutation does not reproduce the action limiter's clamp identity.
    """
    meta = _read_metadata(path)
    action_type = meta.get("action_type", "absolute")
    # Legacy CSVs recorded the flag instead of the resolved type.
    if action_type == "absolute" and meta.get("use_delta_actions", "").lower() == "true":
        action_type = "delta_obs_t"
    if action_type not in _ABSOLUTE_ACTION_TYPES:
        raise TraceAlignmentError(
            f"{path} has action_type '{action_type}', which is a delta encoding.\n"
            f"This viewer reconstructs poses from absolute joint targets only "
            f"(one of: {', '.join(sorted(_ABSOLUTE_ACTION_TYPES))}).\n"
            f"Replaying it as absolute would render a trajectory the robot never followed."
        )

    with open(path) as f:
        rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
    if not rows:
        raise TraceAlignmentError(f"{path} has no data rows")

    def column_block(prefix: str) -> np.ndarray:
        try:
            return np.asarray(
                [[float(r[f"{prefix}_{i}"]) for i in range(N_CHANNELS)] for r in rows], dtype=float
            )
        except KeyError as e:
            raise TraceAlignmentError(
                f"{path} is missing column {e.args[0]!r}; expected {N_CHANNELS} channels of "
                f"obs_state/raw_output/control_cmd (bimanual OpenArm: 7 joints + gripper per arm)"
            ) from e

    ts = np.asarray([float(r["timestamp"]) for r in rows], dtype=float)
    raw = column_block("raw_output")
    obs = column_block("obs_state")[:, MODEL_TO_CTRL]
    cmd = column_block("control_cmd")[:, MODEL_TO_CTRL]

    # The action limiter clamps each command to within `cap` of the observed position, so
    # cmd == obs + clip(raw - obs, -cap, cap) must hold exactly. It only holds if all three
    # blocks are in the same channel order, which makes it a precise alignment check.
    cap = float(np.abs(cmd - obs).max())
    residual = float(np.abs(obs + np.clip(raw - obs, -cap, cap) - cmd).max())
    if residual > 1e-5:
        raise TraceAlignmentError(
            f"{path}: clamp identity failed (residual {residual:.3e}).\n"
            f"The channel permutation does not match this file -- refusing to render it."
        )
    # A joint-scale delta on a gripper channel means the arm/gripper split is off, which the
    # clamp identity alone would not catch. Both grippers are checked: the permutation puts
    # them at opposite ends of the layout, so a one-sided check misses a right-arm mix-up.
    for channel, side in zip(GRIPPER_CHANNELS, ("left", "right")):
        deviation = float(np.abs(raw[:, channel] - obs[:, channel]).max())
        if deviation > _MAX_GRIPPER_DEVIATION:
            raise TraceAlignmentError(
                f"{path}: {side} gripper delta is joint-scale ({deviation:.3f} > "
                f"{_MAX_GRIPPER_DEVIATION}) -- channel alignment is wrong."
            )

    return Trace(
        path=path,
        action_type=action_type,
        ts=ts,
        rel=ts - ts[0],
        raw=raw,
        obs=obs,
        cmd=cmd,
        cap=cap,
        residual=residual,
    )


def undersampling_warning(trace: Trace, control_frequency: float | None) -> str | None:
    """Flag a trace whose logging rate is below the rate commands were issued at.

    The monitor flushes at a fixed ~30 Hz regardless of the inference node's
    `control_frequency`. When the node ran faster, the CSV is a subsample and some commands
    the arm received are simply absent -- the replay is still useful but is not complete.
    """
    if not control_frequency or trace.hz <= 0:
        return None
    if control_frequency <= trace.hz * 1.1:
        return None
    missing = 1.0 - trace.hz / control_frequency
    return (
        f"CSV logged at {trace.hz:.1f} Hz but commands were issued at {control_frequency:.0f} Hz: "
        f"roughly {missing:.0%} of commands are not in this file. Motion between samples is "
        f"held, not interpolated, so brief excursions may be invisible."
    )
