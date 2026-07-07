"""SE(3) EE action transforms shared by trainer, anvil_eval, and ROS inference.

Layout conventions
------------------
State per arm  (8 dims): [x, y, z, qx, qy, qz, qw, gripper]
Action per arm (10 dims): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]

  - Rotation in action is 6D (rot6d, Zhou et al. 2019): first two columns of the
    3×3 rotation matrix, stacked column-major as [R[:,0], R[:,1]].
  - Rotation in state is quaternion [qx, qy, qz, qw] (ROS / TF2 convention).
  - Gripper is in metres; training keeps it in absolute space (no delta).

Bimanual: state (16,), action (20,) — left arm first, right arm second.

Public API
----------
n_arms_from_dims(state_dim, action_dim)      → int
ee_rel_forward(action_abs, state)            → np.ndarray   abs → rel action (training), UMI body-frame
ee_rel_inverse(action_rel, state)            → np.ndarray   rel → abs action (inference/eval), UMI body-frame
ee_rel_world_forward(action_abs, state)      → np.ndarray   abs → rel action, WORLD frame (experimental —
                                                              not part of the ee_abs/ee_rel contract, see
                                                              anvil_trainer.transforms.EERelWorldTransform)
ee_rel_world_inverse(action_rel, state)      → np.ndarray   rel → abs action, WORLD frame (experimental)
ee_obs_rel_forward(obs_abs, anchor)          → np.ndarray   abs obs (8n) → rel obs (10n)
ee_obs_rel_world_forward(obs_abs, anchor)    → np.ndarray   abs obs (8n) → rel obs (10n), WORLD frame
                                                              (experimental, see ee_rel_world_forward)
ee_obs_abs_forward(obs_abs)                  → np.ndarray   abs obs (8n quat) → abs obs (10n rot6d)
ee_action_to_poses(action_abs, n_arms)       → list[dict]   for CommandedEEPose
ee_rot6d_to_quat_layout(actions_10)         → np.ndarray   (T,10n) rot6d → (T,8n) quat
ee_quat_layout_names(rot6d_names)            → list[str]    feature name conversion
"""
from __future__ import annotations

import numpy as np

from anvil_shared.rotation import (
    matrices_to_quats,
    matrices_to_rot6d,
    matrix_to_quat,
    quat_to_matrix,
    quats_to_matrices,
    rot6d_to_matrix,
    rot6ds_to_matrices,
)

EE_STATE_DIM_PER_ARM = 8   # [x, y, z, qx, qy, qz, qw, gripper]
EE_ACTION_DIM_PER_ARM = 10  # [x, y, z, r0..r5, gripper]


def n_arms_from_dims(state_dim: int, action_dim: int) -> int:
    """Validate EE layout dimensions and return the number of arms.

    Raises
    ------
    ValueError
        If ``state_dim`` is not a positive multiple of 8, or if
        ``action_dim != 10 * (state_dim // 8)``.
    """
    if state_dim <= 0 or state_dim % EE_STATE_DIM_PER_ARM != 0:
        raise ValueError(
            f"EE observation.state dim {state_dim} is not a positive multiple of "
            f"{EE_STATE_DIM_PER_ARM}; expected 8 * n_arms (bimanual=16, single=8)."
        )
    n = state_dim // EE_STATE_DIM_PER_ARM
    expected_action = EE_ACTION_DIM_PER_ARM * n
    if action_dim != expected_action:
        raise ValueError(
            f"EE action dim {action_dim} != {expected_action} ({EE_ACTION_DIM_PER_ARM} * {n} arms). "
            f"State suggests {n} arm(s)."
        )
    return n


def ee_rel_forward(
    action_abs: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Convert absolute EE actions to full SE(3)-relative representation.

    This is the forward transform applied at training time (and used for
    computing stats and GT in evaluation).  Matches UMI 'relative' mode:
    ``T_rel = inv(T_state) @ T_action``, so translation is in **body frame**.

    Per arm:
        body_delta  = R_state.T @ (act_xyz - state_xyz)   (body-frame translation)
        delta_rot6d = matrices_to_rot6d(R_state.T @ R_action)
        gripper     = act_gripper  (passthrough — kept in absolute space, not relativised)

    Parameters
    ----------
    action_abs:
        Absolute EE actions in rot6d encoding.
        Shape ``(..., 10 * n_arms)``.  A 1-D input ``(10 * n_arms,)`` is
        also accepted.
    state:
        EE observation state.
        Either ``(8 * n_arms,)`` — a **single** reference state broadcast
        over all time steps; or ``(..., 8 * n_arms)`` — **per-sample**
        states with the same leading dims as ``action_abs`` (used for
        dataset-wide stats computation where every frame has its own state).

    Returns
    -------
    np.ndarray
        Relative actions with the same shape as ``action_abs``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_abs.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_abs.copy()
    per_sample_state = state.ndim > 1  # True when state has same batch leading dims

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]    # (3,) or (..., 3)
        state_quat = state[..., s0 + 3:s0 + 7]  # (4,) or (..., 4)
        world_delta = action_abs[..., a0:a0 + 3] - state_xyz  # (..., 3)

        # rot6d: R_rel = R_state.T @ R_action — vectorised over time/batch dims
        act_r6d = action_abs[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_action = rot6ds_to_matrices(act_r6d)      # (..., 3, 3)

        if per_sample_state:
            Rs_state = quats_to_matrices(state_quat)          # (..., 3, 3)
            Rs_state_T = Rs_state.swapaxes(-2, -1)            # (..., 3, 3)
            Rs_rel = Rs_state_T @ Rs_action                   # (..., 3, 3)
            # Body-frame translation: R_state.T @ world_delta per sample
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_state_T, world_delta)
        else:
            R_state = quat_to_matrix(state_quat)              # (3, 3)
            Rs_rel = R_state.T @ Rs_action                    # (3,3) @ (...,3,3) → (...,3,3)
            # Body-frame translation: world_delta @ R_state = R_state.T applied (row-vector)
            result[..., a0:a0 + 3] = world_delta @ R_state

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)  # (..., 6)
        # gripper unchanged (already copied via .copy())

    return result


def ee_rel_inverse(
    action_rel: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Restore SE(3)-relative EE actions to absolute representation.

    Inverse of :func:`ee_rel_forward`.  Used at inference time to convert
    model outputs back to absolute EE poses before publishing.

    Per arm:
        abs_xyz     = state_xyz + R_state @ body_delta   (body-frame → world-frame translation)
        R_abs       = R_state @ rot6ds_to_matrices(delta_rot6d)
        abs_rot6d   = matrices_to_rot6d(R_abs)
        gripper     = rel_gripper  (passthrough — was kept absolute during forward transform)

    Parameters
    ----------
    action_rel:
        Relative EE actions.  Shape ``(..., 10 * n_arms)``.
    state:
        EE observation state used as the restore reference.
        Either ``(8 * n_arms,)`` (single reference, broadcasts) or
        ``(..., 8 * n_arms)`` (per-sample, same leading dims).

    Returns
    -------
    np.ndarray
        Absolute EE actions (rot6d encoded) with the same shape as
        ``action_rel``.
    """
    action_rel = np.asarray(action_rel, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_rel.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_rel.copy()
    per_sample_state = state.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_quat = state[..., s0 + 3:s0 + 7]
        body_delta = action_rel[..., a0:a0 + 3]      # body-frame translation

        # rot6d: R_abs = R_state @ R_rel
        rel_r6d = action_rel[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_rel = rot6ds_to_matrices(rel_r6d)          # (..., 3, 3)

        if per_sample_state:
            Rs_state = quats_to_matrices(state_quat)  # (..., 3, 3)
            Rs_abs = Rs_state @ Rs_rel                # (..., 3, 3)
            # abs_xyz = R_state @ body_delta + state_xyz (body→world rotation)
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_state, body_delta) + state_xyz
        else:
            R_state = quat_to_matrix(state_quat)      # (3, 3)
            Rs_abs = R_state @ Rs_rel                 # (3,3) @ (...,3,3) → (...,3,3)
            # body_delta @ R_state.T → world_delta (row-vector convention)
            result[..., a0:a0 + 3] = body_delta @ R_state.T + state_xyz

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_abs)
        # gripper unchanged

    return result


def ee_rel_world_forward(
    action_abs: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Convert absolute EE actions to SE(3)-relative representation, WORLD frame.

    Experimental variant of :func:`ee_rel_forward` used to isolate whether
    UMI's body-frame translation choice (vs a world-frame delta) explains
    part of ee_rel's closed-loop performance gap in the anvil-sim LIBERO
    benchmark — NOT part of the ee_abs/ee_rel production contract. Unlike
    ``ee_rel_forward``, translation is NOT projected into the reference
    state's frame, and rotation composition order matches the WORLD-frame
    delta convention used elsewhere for calibration
    (``R_delta_world = R_target @ R_current.T``), not UMI's body-frame
    ``R_state.T @ R_action``.

    Per arm:
        world_delta = act_xyz - state_xyz        (world-frame translation, no projection)
        delta_rot6d = matrices_to_rot6d(R_action @ R_state.T)
        gripper     = act_gripper  (passthrough)

    Parameters
    ----------
    action_abs, state:
        Same shapes/conventions as :func:`ee_rel_forward`.

    Returns
    -------
    np.ndarray
        Relative actions with the same shape as ``action_abs``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_abs.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_abs.copy()
    per_sample_state = state.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_quat = state[..., s0 + 3:s0 + 7]
        world_delta = action_abs[..., a0:a0 + 3] - state_xyz  # (..., 3) -- no frame projection

        act_r6d = action_abs[..., a0 + 3:a0 + 9]
        Rs_action = rot6ds_to_matrices(act_r6d)

        if per_sample_state:
            Rs_state = quats_to_matrices(state_quat)
            Rs_state_T = Rs_state.swapaxes(-2, -1)
            Rs_rel = Rs_action @ Rs_state_T  # world-frame: R_delta_world = R_target @ R_current.T
        else:
            R_state = quat_to_matrix(state_quat)
            Rs_rel = Rs_action @ R_state.T

        result[..., a0:a0 + 3] = world_delta
        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)
        # gripper unchanged (already copied via .copy())

    return result


def ee_rel_world_inverse(
    action_rel: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Restore world-frame SE(3)-relative EE actions to absolute representation.

    Exact inverse of :func:`ee_rel_world_forward`.

    Per arm:
        abs_xyz     = state_xyz + world_delta
        R_abs       = R_rel @ R_state   (since R_rel = R_target @ R_state.T)
        abs_rot6d   = matrices_to_rot6d(R_abs)
        gripper     = rel_gripper  (passthrough)

    Parameters
    ----------
    action_rel, state:
        Same shapes/conventions as :func:`ee_rel_inverse`.

    Returns
    -------
    np.ndarray
        Absolute EE actions (rot6d encoded) with the same shape as ``action_rel``.
    """
    action_rel = np.asarray(action_rel, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_rel.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_rel.copy()
    per_sample_state = state.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_quat = state[..., s0 + 3:s0 + 7]
        world_delta = action_rel[..., a0:a0 + 3]  # already world-frame, no un-projection needed

        rel_r6d = action_rel[..., a0 + 3:a0 + 9]
        Rs_rel = rot6ds_to_matrices(rel_r6d)

        if per_sample_state:
            Rs_state = quats_to_matrices(state_quat)
            Rs_abs = Rs_rel @ Rs_state  # R_abs = R_rel @ R_state
        else:
            R_state = quat_to_matrix(state_quat)
            Rs_abs = Rs_rel @ R_state

        result[..., a0:a0 + 3] = world_delta + state_xyz
        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_abs)
        # gripper unchanged

    return result


def ee_obs_rel_world_forward(obs_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """World-frame counterpart of :func:`ee_obs_rel_forward`, for the
    experimental ``ee_rel_world`` action type.

    When ``obs_abs == anchor`` (the self-anchored single-step-observation
    case used at eval time, see ``anvil_sim.libero_processor.AnvilEEObsProcessorStep``)
    this is mathematically IDENTICAL to ``ee_obs_rel_forward`` (both reduce to
    zero translation + identity rotation regardless of frame convention).
    It only differs from ``ee_obs_rel_forward`` for genuinely distinct
    obs/anchor pairs — i.e. the multi-step-observation stats computation in
    ``anvil_trainer.patches`` when ``n_obs_steps > 1`` (Diffusion Policy).

    Per arm:
        world_delta = obs_xyz - anchor_xyz        (no frame projection)
        rel_rot6d   = matrices_to_rot6d(R_obs @ R_anchor.T)
        gripper     = obs_gripper  (kept absolute)
    """
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)

    n_arms = anchor.shape[-1] // EE_STATE_DIM_PER_ARM
    out_shape = obs_abs.shape[:-1] + (n_arms * EE_ACTION_DIM_PER_ARM,)
    result = np.empty(out_shape, dtype=np.float64)
    per_sample_anchor = anchor.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        obs_xyz = obs_abs[..., s0:s0 + 3]
        obs_quat = obs_abs[..., s0 + 3:s0 + 7]
        obs_grip = obs_abs[..., s0 + 7:s0 + 8]
        anchor_xyz = anchor[..., s0:s0 + 3]
        anchor_quat = anchor[..., s0 + 3:s0 + 7]

        Rs_obs = quats_to_matrices(obs_quat)
        world_delta = obs_xyz - anchor_xyz  # no frame projection

        if per_sample_anchor:
            Rs_anchor = quats_to_matrices(anchor_quat)
            Rs_rel = Rs_obs @ Rs_anchor.swapaxes(-2, -1)  # world-frame: R_obs @ R_anchor.T
        else:
            R_anchor = quat_to_matrix(anchor_quat)
            Rs_rel = Rs_obs @ R_anchor.T

        result[..., a0:a0 + 3] = world_delta
        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)
        result[..., a0 + 9:a0 + 10] = obs_grip

    return result


def ee_obs_rel_forward(obs_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Convert absolute EE observations to SE(3)-relative, rot6d layout.

    Matches UMI's obs relativisation: ``T_rel = inv(T_anchor) @ T_obs``.
    Input obs use quaternion layout (8 dims/arm), output uses rot6d (10 dims/arm).

    Per arm:
        body_delta  = R_anchor.T @ (obs_xyz - anchor_xyz)
        rel_rot6d   = matrices_to_rot6d(R_anchor.T @ R_obs)
        gripper     = obs_gripper  (kept absolute)

    Parameters
    ----------
    obs_abs:
        Absolute EE observations in quaternion layout.
        Shape ``(..., 8 * n_arms)``.
    anchor:
        Reference EE state (quaternion layout) to relativise against.
        Either ``(8 * n_arms,)`` — single anchor (broadcast) or
        ``(..., 8 * n_arms)`` — per-sample anchor (same leading dims as obs_abs).

    Returns
    -------
    np.ndarray
        Relative observations in rot6d layout, shape ``(..., 10 * n_arms)``.
    """
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)

    n_arms = anchor.shape[-1] // EE_STATE_DIM_PER_ARM
    out_shape = obs_abs.shape[:-1] + (n_arms * EE_ACTION_DIM_PER_ARM,)
    result = np.empty(out_shape, dtype=np.float64)
    per_sample_anchor = anchor.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        obs_xyz = obs_abs[..., s0:s0 + 3]
        obs_quat = obs_abs[..., s0 + 3:s0 + 7]
        obs_grip = obs_abs[..., s0 + 7:s0 + 8]
        anchor_xyz = anchor[..., s0:s0 + 3]
        anchor_quat = anchor[..., s0 + 3:s0 + 7]

        Rs_obs = quats_to_matrices(obs_quat)          # (..., 3, 3)
        world_delta = obs_xyz - anchor_xyz

        if per_sample_anchor:
            Rs_anchor = quats_to_matrices(anchor_quat)        # (..., 3, 3)
            Rs_anchor_T = Rs_anchor.swapaxes(-2, -1)
            Rs_rel = Rs_anchor_T @ Rs_obs
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_anchor_T, world_delta)
        else:
            R_anchor = quat_to_matrix(anchor_quat)            # (3, 3)
            Rs_rel = R_anchor.T @ Rs_obs                      # (..., 3, 3)
            result[..., a0:a0 + 3] = world_delta @ R_anchor   # row-vector: R_anchor.T applied

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)
        result[..., a0 + 9:a0 + 10] = obs_grip

    return result


def ee_obs_abs_forward(obs_abs: np.ndarray) -> np.ndarray:
    """Convert absolute EE observations from quaternion layout to rot6d layout.

    Input obs use quaternion layout (8 dims/arm), output uses rot6d (10 dims/arm).
    No SE(3) relative computation — xyz and gripper are passthrough; only rotation
    is re-encoded from quaternion to rot6d for network compatibility.

    Per arm:
        xyz     = obs_xyz  (passthrough)
        rot6d   = matrices_to_rot6d(quats_to_matrices(obs_quat))
        gripper = obs_gripper  (kept absolute)

    Parameters
    ----------
    obs_abs:
        Absolute EE observations in quaternion layout.
        Shape ``(..., 8 * n_arms)``.

    Returns
    -------
    np.ndarray
        Absolute observations in rot6d layout, shape ``(..., 10 * n_arms)``.
    """
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    n_arms = obs_abs.shape[-1] // EE_STATE_DIM_PER_ARM
    out_shape = obs_abs.shape[:-1] + (n_arms * EE_ACTION_DIM_PER_ARM,)
    result = np.empty(out_shape, dtype=np.float64)

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        obs_xyz = obs_abs[..., s0:s0 + 3]
        obs_quat = obs_abs[..., s0 + 3:s0 + 7]
        obs_grip = obs_abs[..., s0 + 7:s0 + 8]

        Rs_obs = quats_to_matrices(obs_quat)          # (..., 3, 3)
        result[..., a0:a0 + 3] = obs_xyz
        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_obs)
        result[..., a0 + 9:a0 + 10] = obs_grip

    return result


def ee_rot6d_to_quat_layout(actions_10: np.ndarray) -> np.ndarray:
    """Convert EE actions from rot6d layout to quaternion layout.

    Parameters
    ----------
    actions_10:
        ``(T, 10 * n_arms)`` absolute EE actions in rot6d encoding.
        Per arm: [x, y, z, r0..r5, gripper].

    Returns
    -------
    np.ndarray
        ``(T, 8 * n_arms)`` with per-arm layout [x, y, z, qx, qy, qz, qw, gripper].
        Uses vectorised ``rot6ds_to_matrices`` → ``matrices_to_quats``.
    """
    arr = np.asarray(actions_10, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"ee_rot6d_to_quat_layout: expected 2D (T, 10*n), got {arr.shape}"
        )
    _T, D = arr.shape
    if D % EE_ACTION_DIM_PER_ARM != 0:
        raise ValueError(
            f"ee_rot6d_to_quat_layout: dim {D} not divisible by {EE_ACTION_DIM_PER_ARM}"
        )
    n_arms = D // EE_ACTION_DIM_PER_ARM

    out_cols: list[np.ndarray] = []
    for arm_idx in range(n_arms):
        a0 = arm_idx * EE_ACTION_DIM_PER_ARM
        xyz  = arr[:, a0:a0 + 3]           # (T, 3)
        r6d  = arr[:, a0 + 3:a0 + 9]       # (T, 6)
        grip = arr[:, a0 + 9:a0 + 10]      # (T, 1)
        R    = rot6ds_to_matrices(r6d)      # (T, 3, 3)
        quat = matrices_to_quats(R)         # (T, 4) [qx, qy, qz, qw]
        out_cols.extend([xyz, quat, grip])

    return np.concatenate(out_cols, axis=1)  # (T, 8*n_arms)


def ee_quat_layout_names(rot6d_names: list[str]) -> list[str]:
    """Convert EE feature names from rot6d layout (10/arm) to quaternion layout (8/arm).

    Example::

        ["right_x","right_y","right_z","right_r0",...,"right_r5","right_gripper"]
        → ["right_x","right_y","right_z","right_qx","right_qy","right_qz","right_qw",
           "right_gripper"]
    """
    n = len(rot6d_names)
    if n % EE_ACTION_DIM_PER_ARM != 0:
        raise ValueError(
            f"ee_quat_layout_names: expected multiple of {EE_ACTION_DIM_PER_ARM} names, got {n}"
        )
    n_arms = n // EE_ACTION_DIM_PER_ARM
    out: list[str] = []
    for arm_idx in range(n_arms):
        prefix = rot6d_names[arm_idx * EE_ACTION_DIM_PER_ARM].rsplit("_", 1)[0]
        out += [
            f"{prefix}_x", f"{prefix}_y", f"{prefix}_z",
            f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw",
            f"{prefix}_gripper",
        ]
    return out


def ee_action_to_poses(
    action_abs: np.ndarray,
    n_arms: int | None = None,
) -> list[dict]:
    """Convert a chunk of absolute rot6d EE actions to per-step per-arm pose dicts.

    Replaces the old ``rot6d_chunk_to_quat`` in ``delta_restore.py``.

    Parameters
    ----------
    action_abs:
        Absolute EE actions, shape ``(chunk_size, 10 * n_arms)`` or
        ``(10 * n_arms,)`` for a single step.
    n_arms:
        Number of arms.  Derived from ``action_abs.shape[1] // 10`` when
        ``None`` (default).

    Returns
    -------
    list of dict
        One dict per time step.  Each dict maps ``arm_index (int)`` →
        ``{"pos": np.ndarray (3,), "quat_xyzw": np.ndarray (4,), "gripper": float}``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    if action_abs.ndim == 1:
        action_abs = action_abs[np.newaxis, :]  # (1, D)

    chunk_size, D = action_abs.shape
    if n_arms is None:
        n_arms = D // EE_ACTION_DIM_PER_ARM

    result: list[dict] = []
    for k in range(chunk_size):
        step: dict = {}
        for arm in range(n_arms):
            a0 = arm * EE_ACTION_DIM_PER_ARM
            pos = action_abs[k, a0:a0 + 3].copy()
            r6d = action_abs[k, a0 + 3:a0 + 9]
            grip = float(action_abs[k, a0 + 9])
            R = rot6d_to_matrix(r6d)
            quat = matrix_to_quat(R)  # [qx, qy, qz, qw]
            step[arm] = {"pos": pos, "quat_xyzw": quat, "gripper": grip}
        result.append(step)
    return result
