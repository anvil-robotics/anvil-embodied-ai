"""EE Cartesian runtime utilities for inference.

``resolve_action_type(cfg)``
    Normalises the action_type field from a checkpoint anvil_config dict.
    Accepts the three canonical types: joint_abs, ee_abs, ee_relative — plus
    the permanent legacy alias "ee_rel" (existing checkpoints), which is
    normalized to "ee_relative" via ``anvil_shared.action_types``.

``read_checkpoint_anvil_config(model_path)``
    Resolves a checkpoint path (bare / pretrained_model/ / HF-cache snapshot)
    and reads its anvil_config.json, if present.

``ee_relative_restore_chunk(chunk_np, obs_t)``
    Restores EE relative actions (ee_relative; chunk-anchor, n-0) to absolute
    EE poses.
    Thin wrapper around ``anvil_shared.ee_transform.ee_relative_inverse``.

``ee_poses_from_chunk(chunk_np, n_arms)``
    Converts a chunk of absolute rot6d EE actions to per-step per-arm
    pose dicts suitable for building ``CommandedEEPose`` messages.
    Thin wrapper around ``anvil_shared.ee_transform.ee_action_to_poses``.

``ee_delta_restore_step(delta, obs_t)``
    Restores ONE Delta(n->n+1) model output to an absolute EE pose, composed
    fresh against the freshest observed pose (``obs_t``) — this is a
    per-publish-tick composition, NOT a chunk-anchor restore (contrast with
    ``ee_relative_restore_chunk``, which restores a whole chunk against ONE
    fixed chunk-generation-time anchor). World-frame; thin wrapper around
    ``anvil_shared.ee_transform.ee_delta_inverse``.

``pose_arrival_error(current, target)``
    Position/orientation distance between two 8-dim EE poses (quat layout).
    Shared by ``gt_replay_verifier_node``'s trajectory comparison and
    ``dataset_gt_replayer_node``'s pre-replay homing arrival check — same
    "how far apart are these two poses" math, different callers/tolerances.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ensure_anvil_shared() -> None:
    """Add packages/anvil_shared/src to sys.path so ee_transform helpers are importable.

    Called lazily inside EE functions so the import overhead is paid only when
    those functions are actually used.
    """
    import os
    import sys

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    _shared_src = os.path.join(_repo_root, "packages", "anvil_shared", "src")
    if _shared_src not in sys.path:
        sys.path.insert(0, _shared_src)


def resolve_action_type(cfg: dict) -> str:
    """Return the normalised action_type string from an anvil_config dict.

    Accepts "joint_abs", "ee_abs", "ee_relative" — plus the permanent legacy
    alias "ee_rel", which is mapped to "ee_relative" here (see
    ``anvil_shared.action_types.normalize_action_type``). Old checkpoints
    that pre-date the three-type scheme will have ``action_type="joint_abs"``
    (or absent, defaulting to "joint_abs").
    """
    _ensure_anvil_shared()
    from anvil_shared.action_types import normalize_action_type

    return normalize_action_type(cfg.get("action_type", "joint_abs"))


def read_checkpoint_anvil_config(model_path: str) -> dict:
    """Resolve *model_path* to a checkpoint dir and read its anvil_config.json.

    Mirrors the path resolution in ``inference_node._load_run_metadata``
    (bare checkpoint dir / ``pretrained_model/`` subdir / HF-cache
    ``snapshots/<hash>/`` layout) so callers get the same answer regardless
    of which convention *model_path* uses.

    Returns ``{}`` if *model_path* is falsy or no anvil_config.json is found —
    callers should fall back to their own default (e.g. a ROS param) in that case.
    """
    if not model_path:
        return {}

    checkpoint = Path(model_path)

    pretrained = checkpoint / "pretrained_model"
    if pretrained.exists() and (pretrained / "config.json").exists():
        checkpoint = pretrained

    if not (checkpoint / "config.json").exists():
        snapshots = checkpoint / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    checkpoint = snap
                    break

    anvil_path = checkpoint / "anvil_config.json"
    if not anvil_path.exists():
        return {}
    return json.loads(anvil_path.read_text())


def ee_relative_restore_chunk(
    chunk_np: np.ndarray,
    obs_t: np.ndarray,
) -> np.ndarray:
    """Restore EE relative actions (ee_relative; chunk-anchor, n-0) to absolute EE poses.

    Inverse of the SE(3) forward transform applied at training time.

    Per arm (10 action dims, 8 state dims):
        abs_xyz   = obs_xyz + delta_xyz
        R_abs     = R_state @ rot6ds_to_matrices(delta_rot6d)
        abs_rot6d = matrices_to_rot6d(R_abs)
        gripper   = delta_gripper  (kept absolute during training)

    Args:
        chunk_np: (chunk_size, 10*n_arms) relative-space model output.
        obs_t:    (8*n_arms,) or (n_obs_steps, 8*n_arms) observation state
                  at chunk generation time. If 2-D, the last row is used.

    Returns:
        (chunk_size, 10*n_arms) absolute EE actions (rot6d encoded).
    """
    try:
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_relative_inverse
    except ImportError as e:
        raise ImportError(
            "ee_relative_restore_chunk requires anvil_shared.ee_transform. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)

    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    # Accept stacked multi-step obs (e.g. shape (n_obs_steps, 8*n_arms)); use last row.
    if obs_t.ndim > 1:
        obs_t = obs_t[-1]

    return ee_relative_inverse(chunk_np, obs_t)


def ee_delta_restore_step(
    delta: np.ndarray,
    obs_t: np.ndarray,
) -> np.ndarray:
    """Restore ONE Delta(n->n+1) model output to an absolute EE pose.

    Unlike :func:`ee_relative_restore_chunk` (which restores a whole chunk
    against a single anchor captured at chunk-generation time), this composes
    a single delta against the FRESHEST observed pose, at publish time —
    intended to be called once per publish tick, every tick, with whatever
    ``obs_t`` is current at that instant. World-frame (verified against
    robosuite 1.4.0's own OSC composition — see
    ``anvil_shared.ee_transform.ee_delta_inverse``'s docstring).

    Per arm (10 action/delta dims, 8 state dims):
        abs_xyz   = obs_xyz + delta_xyz                (plain world-frame addition)
        R_abs     = R_delta @ R_state                  (world-frame/extrinsic)
        abs_rot6d = matrices_to_rot6d(R_abs)
        gripper   = delta_gripper  (kept absolute during training)

    Args:
        delta: (10*n_arms,) or (1, 10*n_arms) model-output delta (already
            denormalized — physical units, not normalized-space values).
        obs_t: (8*n_arms,) the freshest observed EE pose at this publish tick.

    Returns:
        (10*n_arms,) absolute EE action (rot6d encoded).
    """
    try:
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_delta_inverse
    except ImportError as e:
        raise ImportError(
            "ee_delta_restore_step requires anvil_shared.ee_transform. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    delta = np.asarray(delta, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)

    single = delta.ndim == 1
    if single:
        delta = delta[np.newaxis, :]
    if obs_t.ndim > 1:
        obs_t = obs_t[-1]

    result = ee_delta_inverse(delta, obs_t)
    return result[0] if single else result


def pose_arrival_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Position/orientation distance between two 8-dim EE poses (quat layout).

    Both ``current`` and ``target`` are ``[x, y, z, qx, qy, qz, qw, gripper]`` (a
    trailing gripper value, if present, is ignored — only indices 0:7 are read, so
    a bare 7-element ``[x, y, z, qx, qy, qz, qw]`` works too). No dependency on
    anvil_shared — plain numpy, safe to import anywhere this module already is.

    Returns:
        ``(pos_err_m, rot_err_deg)`` — plain L2 position distance, and the
        sign-invariant geodesic angle between the two quaternions
        (``2*arccos(|dot(q1,q2)|)``, since ``q`` and ``-q`` represent the same
        rotation).
    """
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pos_err = float(np.linalg.norm(current[0:3] - target[0:3]))
    dot = float(np.clip(np.abs(np.dot(current[3:7], target[3:7])), 0.0, 1.0))
    rot_err = float(np.degrees(2.0 * np.arccos(dot)))
    return pos_err, rot_err


def ramp_toward_pose(
    current: np.ndarray,
    target: np.ndarray,
    max_pos_delta_m: float,
    max_rot_delta_deg: float,
) -> np.ndarray:
    """One ramped step from ``current`` toward ``target`` (8-dim quat-layout
    EE poses: ``[x, y, z, qx, qy, qz, qw, gripper]``), moving at most
    ``max_pos_delta_m`` metres and ``max_rot_delta_deg`` degrees this step.
    Gripper passes through from ``target`` unramped — bounded by the
    hardware's own clamp already, not a motion-safety concern the way
    position/orientation are.

    Exists because ``inference_node.py``'s ``action_limiter`` (the joint-space
    per-tick delta-limiting safety net) is explicitly not applied in EE mode —
    a one-shot absolute EE command (e.g. pre-replay homing) would otherwise
    jump straight to its target regardless of how far the robot's current
    pose is from it. Calling this every tick with the SAME fixed target and a
    freshly-read ``current`` converges toward it gradually instead.

    Position: clamps the step's magnitude, preserving direction (not
    per-axis — a real safety speed limit is naturally expressed as total
    Euclidean distance per tick, not per-component).

    Orientation: clamps the geodesic rotation angle via SLERP (spherical
    linear interpolation) — the correct way to take a bounded step along the
    shortest rotation path between two quaternions; clamping components
    independently would not stay on that path.
    """
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    pos_delta = target[0:3] - current[0:3]
    pos_dist = float(np.linalg.norm(pos_delta))
    if pos_dist > max_pos_delta_m and pos_dist > 1e-12:
        ramped_pos = current[0:3] + pos_delta / pos_dist * max_pos_delta_m
    else:
        ramped_pos = target[0:3]

    q0, q1 = current[3:7].copy(), target[3:7].copy()
    dot = float(np.dot(q0, q1))
    if dot < 0:  # shortest path — q and -q are the same rotation
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle_deg = float(np.degrees(2.0 * np.arccos(dot)))
    if angle_deg > max_rot_delta_deg and angle_deg > 1e-9:
        theta_0 = np.arccos(dot)
        theta = theta_0 * (max_rot_delta_deg / angle_deg)
        q_perp = q1 - q0 * dot
        q_perp_norm = float(np.linalg.norm(q_perp))
        if q_perp_norm > 1e-12:
            q_perp = q_perp / q_perp_norm
            ramped_quat = q0 * np.cos(theta) + q_perp * np.sin(theta)
        else:
            ramped_quat = q0
    else:
        ramped_quat = q1

    return np.concatenate([ramped_pos, ramped_quat, [target[7]]])


def ee_poses_from_chunk(
    chunk_np: np.ndarray,
    n_arms: int | None = None,
) -> list[dict]:
    """Convert a chunk of absolute rot6d EE actions to per-step per-arm pose dicts.

    Args:
        chunk_np: (chunk_size, 10*n_arms) absolute rot6d actions,
                  or (10*n_arms,) for a single step.
        n_arms:   Number of arms. Derived from chunk_np.shape[1] // 10 when None.

    Returns:
        List of chunk_size dicts. Each dict maps arm_index (int) to:
          {"pos": np.ndarray (3,), "quat_xyzw": np.ndarray (4,), "gripper": float}
        where quat_xyzw = [qx, qy, qz, qw] (ROS convention).
    """
    try:
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_action_to_poses
    except ImportError as e:
        raise ImportError(
            "ee_poses_from_chunk requires anvil_shared.ee_transform. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    return ee_action_to_poses(chunk_np, n_arms=n_arms)
