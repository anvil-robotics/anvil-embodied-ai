"""SE(3) EE action transforms shared by trainer, anvil_eval, and ROS inference.

Layout conventions
------------------
State per arm, ``observation_encoding="quaternion"`` (default, 8 dims): [x, y, z, qx, qy, qz, qw, gripper]
State per arm, ``observation_encoding="rot6d"``      (10 dims): [x, y, z, r0..r5, gripper]
State per arm, ``observation_encoding="axis_angle"``  (7 dims): [x, y, z, ax, ay, az, gripper]
Action per arm (always 10 dims, regardless of observation_encoding): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]

  - Rotation in action is ALWAYS 6D (rot6d, Zhou et al. 2019): first two columns of the
    3×3 rotation matrix, stacked column-major as [R[:,0], R[:,1]] — mcap_converter writes
    action in rot6d regardless of the dataset's observation_encoding (see
    ``mcap_converter/config/schema.py``'s ``action_encoding``/``observation_encoding`` fields).
  - Rotation in state depends on the dataset's ``observation_encoding``: quaternion
    [qx, qy, qz, qw] (ROS / TF2 convention, the schema default), rot6d, or axis-angle.
    Every function below that reads ``state``/``anchor``/``obs_abs`` takes an
    ``observation_encoding: str = "quaternion"`` keyword so callers whose dataset uses a
    non-default encoding must pass it explicitly — the default exists only to preserve
    behavior for the (many) existing call sites that predate this parameter and were
    written when quaternion was the only supported encoding; it is NOT a claim that
    quaternion is always correct. See ``anvil_shared.ee_encodings`` for the canonical
    per-encoding layout table (this module used to hardcode a copy of it).
  - Gripper is in metres; training keeps it in absolute space (no delta).

Bimanual: state (16/20/14 dims depending on observation_encoding), action (20,) —
left arm first, right arm second.

Public API
----------
(all functions below that take a state/anchor/obs array also accept
``observation_encoding: str = "quaternion"`` — pass the dataset's actual
encoding for any non-default dataset)

n_arms_from_dims(state_dim, action_dim)        → int
ee_relative_forward(action_abs, state)         → np.ndarray   abs → rel action (training) [BODY-frame, n-0 mechanism]
ee_relative_inverse(action_rel, state)         → np.ndarray   rel → abs action (inference/eval) [BODY-frame, n-0 mechanism]
ee_delta_forward(action_abs, state)            → np.ndarray   abs → delta action (training) [WORLD-frame, n->n+1 mechanism]
ee_delta_inverse(delta, state)                 → np.ndarray   delta → abs action (inference/eval) [WORLD-frame, n->n+1 mechanism]
ee_obs_relative_forward(obs_abs, anchor)       → np.ndarray   abs obs (state_dim_per_arm·n) → rel obs (10n)
ee_obs_abs_forward(obs_abs)                    → np.ndarray   abs obs (state_dim_per_arm·n) → abs obs (10n rot6d)
ee_action_to_poses(action_abs, n_arms)         → list[dict]   for CommandedEEPose (action-side only, encoding-independent)
ee_rot6d_to_quat_layout(actions_10)            → np.ndarray   (T,10n) rot6d → (T,8n) quat (action-side only, encoding-independent)
ee_quat_layout_names(rot6d_names)              → list[str]    feature name conversion (action-side only, encoding-independent)

Body-frame vs. world-frame — do not confuse these two pairs
-------------------------------------------------------------
``ee_relative_forward``/``ee_relative_inverse`` implement the **Relative (n-0)**
mechanism (chunk-anchor relativization, UMI-style): translation and rotation are
BOTH expressed in the anchor state's own local/body frame
(``R_state.T @ (...)``). This is the diagnosed root cause of the real-hardware
jitter failure; kept only for the existing ``ee_relative`` action_type.

``ee_delta_forward``/``ee_delta_inverse`` implement the **Delta (n->n+1)**
mechanism (per-frame anchor, forward-looking): translation and rotation are
BOTH expressed in the WORLD frame (no rotation by the anchor state at all for
translation; extrinsic/left-multiply composition for rotation), verified to
match robosuite 1.4.0's own ``OperationalSpaceController`` composition exactly
(``goal_orientation = delta_rotation @ current_orientation``,
``goal_position = current_position + delta``) — the same delivery convention
LIBERO's validated ``native`` condition relies on. These are NOT thin wrappers
around the relative-pair's per-sample branch: the two pairs use genuinely
different composition formulas, not just a different anchor.
"""
from __future__ import annotations

import numpy as np

from anvil_shared.ee_encodings import OBSERVATION_ROTATION_LAYOUTS, observation_state_dim_per_arm
from anvil_shared.rotation import (
    axis_angle_to_matrix,
    axis_angles_to_matrices,
    matrices_to_quats,
    matrices_to_rot6d,
    matrix_to_quat,
    quat_to_matrix,
    quats_to_matrices,
    rot6d_to_matrix,
    rot6ds_to_matrices,
)

EE_STATE_DIM_PER_ARM = 8   # [x, y, z, qx, qy, qz, qw, gripper] — quaternion (schema default) only
EE_ACTION_DIM_PER_ARM = 10  # [x, y, z, r0..r5, gripper] — always rot6d, any observation_encoding

# Per-encoding dispatch: convert a state's rotation component (whichever encoding it's
# in) to a 3x3 matrix. Single-sample vs. batched mirrors anvil_shared.rotation's own
# X_to_matrix / Xs_to_matrices split.
_STATE_ROTATION_TO_MATRIX = {
    "quaternion": quat_to_matrix,
    "rot6d": rot6d_to_matrix,
    "axis_angle": axis_angle_to_matrix,
}
_STATE_ROTATIONS_TO_MATRICES = {
    "quaternion": quats_to_matrices,
    "rot6d": rot6ds_to_matrices,
    "axis_angle": axis_angles_to_matrices,
}


def n_arms_from_dims(
    state_dim: int, action_dim: int, observation_encoding: str = "quaternion"
) -> int:
    """Validate EE layout dimensions and return the number of arms.

    Parameters
    ----------
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding — determines the
        expected per-arm state dim (quaternion=8, rot6d=10, axis_angle=7; see
        ``anvil_shared.ee_encodings``). Defaults to "quaternion" for callers
        that predate this parameter; pass the dataset's actual encoding
        explicitly for any non-default dataset.

    Raises
    ------
    ValueError
        If ``state_dim`` is not a positive multiple of the encoding's per-arm
        dim, or if ``action_dim != 10 * n_arms``.
    """
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    if state_dim <= 0 or state_dim % state_dim_per_arm != 0:
        raise ValueError(
            f"EE observation.state dim {state_dim} is not a positive multiple of "
            f"{state_dim_per_arm} ({observation_encoding!r} per-arm dim); "
            f"expected {state_dim_per_arm} * n_arms."
        )
    n = state_dim // state_dim_per_arm
    expected_action = EE_ACTION_DIM_PER_ARM * n
    if action_dim != expected_action:
        raise ValueError(
            f"EE action dim {action_dim} != {expected_action} ({EE_ACTION_DIM_PER_ARM} * {n} arms). "
            f"State suggests {n} arm(s) ({observation_encoding!r} encoding)."
        )
    return n


def ee_relative_forward(
    action_abs: np.ndarray,
    state: np.ndarray,
    observation_encoding: str = "quaternion",
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
        EE observation state, in ``observation_encoding`` layout.
        Either ``(state_dim_per_arm * n_arms,)`` — a **single** reference
        state broadcast over all time steps; or
        ``(..., state_dim_per_arm * n_arms)`` — **per-sample** states with
        the same leading dims as ``action_abs`` (used for dataset-wide stats
        computation where every frame has its own state).
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

    Returns
    -------
    np.ndarray
        Relative actions with the same shape as ``action_abs``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_abs.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim, observation_encoding)
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrix = _STATE_ROTATION_TO_MATRIX[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    result = action_abs.copy()
    per_sample_state = state.ndim > 1  # True when state has same batch leading dims

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]    # (3,) or (..., 3)
        state_rot = state[..., s0 + 3:s0 + 3 + rot_dim]  # (rot_dim,) or (..., rot_dim)
        world_delta = action_abs[..., a0:a0 + 3] - state_xyz  # (..., 3)

        # rot6d: R_rel = R_state.T @ R_action — vectorised over time/batch dims
        act_r6d = action_abs[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_action = rot6ds_to_matrices(act_r6d)      # (..., 3, 3)

        if per_sample_state:
            Rs_state = to_matrices(state_rot)                 # (..., 3, 3)
            Rs_state_T = Rs_state.swapaxes(-2, -1)            # (..., 3, 3)
            Rs_rel = Rs_state_T @ Rs_action                   # (..., 3, 3)
            # Body-frame translation: R_state.T @ world_delta per sample
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_state_T, world_delta)
        else:
            R_state = to_matrix(state_rot)                    # (3, 3)
            Rs_rel = R_state.T @ Rs_action                    # (3,3) @ (...,3,3) → (...,3,3)
            # Body-frame translation: world_delta @ R_state = R_state.T applied (row-vector)
            result[..., a0:a0 + 3] = world_delta @ R_state

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)  # (..., 6)
        # gripper unchanged (already copied via .copy())

    return result


def ee_relative_inverse(
    action_rel: np.ndarray,
    state: np.ndarray,
    observation_encoding: str = "quaternion",
) -> np.ndarray:
    """Restore SE(3)-relative EE actions to absolute representation.

    Inverse of :func:`ee_relative_forward`.  Used at inference time to convert
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
        EE observation state used as the restore reference, in
        ``observation_encoding`` layout.
        Either ``(state_dim_per_arm * n_arms,)`` (single reference,
        broadcasts) or ``(..., state_dim_per_arm * n_arms)`` (per-sample,
        same leading dims).
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

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
    n_arms = n_arms_from_dims(state_dim, action_dim, observation_encoding)
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrix = _STATE_ROTATION_TO_MATRIX[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    result = action_rel.copy()
    per_sample_state = state.ndim > 1

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_rot = state[..., s0 + 3:s0 + 3 + rot_dim]
        body_delta = action_rel[..., a0:a0 + 3]      # body-frame translation

        # rot6d: R_abs = R_state @ R_rel
        rel_r6d = action_rel[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_rel = rot6ds_to_matrices(rel_r6d)          # (..., 3, 3)

        if per_sample_state:
            Rs_state = to_matrices(state_rot)         # (..., 3, 3)
            Rs_abs = Rs_state @ Rs_rel                # (..., 3, 3)
            # abs_xyz = R_state @ body_delta + state_xyz (body→world rotation)
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_state, body_delta) + state_xyz
        else:
            R_state = to_matrix(state_rot)            # (3, 3)
            Rs_abs = R_state @ Rs_rel                 # (3,3) @ (...,3,3) → (...,3,3)
            # body_delta @ R_state.T → world_delta (row-vector convention)
            result[..., a0:a0 + 3] = body_delta @ R_state.T + state_xyz

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_abs)
        # gripper unchanged

    return result


def ee_delta_forward(
    action_abs: np.ndarray,
    state: np.ndarray,
    observation_encoding: str = "quaternion",
) -> np.ndarray:
    """Convert absolute EE actions to per-frame WORLD-frame delta representation.

    This is the forward transform for the Delta (n->n+1) mechanism: each
    target is relative to THIS frame's own state (per-frame anchor), never a
    fixed chunk-start anchor. mcap_converter calls this as
    ``ee_delta_forward(pose[t+1], anchor=pose[t])`` — the delta needed to go
    from here to the next frame, a genuine forward-looking control target
    (see ``core/extractor.py``'s ``_finalize_pending_action``). Unlike
    :func:`ee_relative_forward` (body-frame, UMI-style), BOTH translation and
    rotation here are expressed in the WORLD frame — verified to match
    robosuite 1.4.0's own ``OperationalSpaceController`` composition
    (``control_utils.py``): ``goal_orientation = delta_rotation @
    current_orientation`` (left-multiply, extrinsic) and ``goal_position =
    current_position + delta`` (raw addition).

    Per arm:
        delta_xyz = act_xyz - state_xyz                   (WORLD-frame translation — no
                                                             rotation by state at all)
        R_delta   = R_action @ R_state.T                   (WORLD-frame/extrinsic rotation)
        gripper   = act_gripper  (passthrough — kept in absolute space, not relativised)

    Identity-delta invariant: calling with ``action_abs == state`` (in
    absolute-pose terms) yields a zero translation delta and identity rot6d
    ``[1,0,0,0,1,0]`` — this degenerates identically whether body-frame or
    world-frame (see ``test_identity_state_identity_action_zero_delta``).

    Parameters
    ----------
    action_abs:
        Absolute EE actions in rot6d encoding. Shape ``(..., 10 * n_arms)``.
    state:
        EE observation state to anchor against — for Delta(n->n+1) this is
        THIS frame's own state, and ``action_abs`` is the NEXT frame's pose
        (that pairing lives entirely in what the caller passes here, not in
        this function). Either ``(8 * n_arms,)`` (single reference,
        broadcasts) or ``(..., 8 * n_arms)`` (per-sample, same leading dims
        as ``action_abs``) — both are handled uniformly via numpy
        broadcasting, no branching needed (translation is a plain
        subtraction; rotation composition via ``@`` broadcasts a bare
        ``(3, 3)`` against a stack of ``(..., 3, 3)`` matrices natively).
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

    Returns
    -------
    np.ndarray
        Delta actions with the same shape as ``action_abs``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_abs.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim, observation_encoding)
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    result = action_abs.copy()

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_rot = state[..., s0 + 3:s0 + 3 + rot_dim]
        act_xyz = action_abs[..., a0:a0 + 3]
        act_r6d = action_abs[..., a0 + 3:a0 + 9]

        # WORLD-frame translation: plain difference, no rotation by state.
        result[..., a0:a0 + 3] = act_xyz - state_xyz

        # WORLD-frame/extrinsic rotation: R_delta = R_action @ R_state.T.
        Rs_action = rot6ds_to_matrices(act_r6d)         # (..., 3, 3)
        Rs_state = to_matrices(state_rot)               # (..., 3, 3) — broadcasts if state is (3,3)
        Rs_state_T = Rs_state.swapaxes(-2, -1)          # (..., 3, 3)
        Rs_delta = Rs_action @ Rs_state_T                # (..., 3, 3) — numpy broadcasts (3,3) @ (...,3,3)

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_delta)
        # gripper unchanged (already copied via .copy())

    return result


def ee_delta_inverse(
    delta: np.ndarray,
    state: np.ndarray,
    observation_encoding: str = "quaternion",
) -> np.ndarray:
    """Restore a per-frame WORLD-frame delta to absolute EE actions.

    Exact inverse of :func:`ee_delta_forward`. This is the composition Item 2's
    decoupled delta-mode publish loop uses at inference time: given the
    freshest observed ``state`` (``R_obs``/``obs_xyz``) and the model's
    predicted delta, compute the absolute target to publish.

    Per arm:
        abs_xyz = state_xyz + delta_xyz                    (WORLD-frame; matches robosuite's
                                                              own ``goal_position = current + delta``)
        R_abs   = R_delta @ R_state                         (WORLD-frame/extrinsic; matches
                                                              robosuite's own
                                                              ``goal_orientation = delta_rotation
                                                              @ current_orientation``)
        gripper = delta_gripper  (passthrough — was kept absolute during forward transform)

    Algebraic check: substituting the forward formula,
    ``R_delta @ R_state = (R_action @ R_state.T) @ R_state = R_action``
    (since ``R_state`` is orthonormal, ``R_state.T @ R_state = I``) — confirmed
    exact inverse, no asymmetry between forward and inverse composition order.

    Parameters
    ----------
    delta:
        Delta EE actions (rot6d encoding). Shape ``(..., 10 * n_arms)``.
    state:
        The state to restore against — at inference time, the FRESHEST
        observed state at publish time (not a stale chunk-generation-time
        anchor). Either ``(state_dim_per_arm * n_arms,)`` or
        ``(..., state_dim_per_arm * n_arms)``.
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

    Returns
    -------
    np.ndarray
        Absolute EE actions (rot6d encoded), same shape as ``delta``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = delta.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim, observation_encoding)
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    result = delta.copy()

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_rot = state[..., s0 + 3:s0 + 3 + rot_dim]
        delta_xyz = delta[..., a0:a0 + 3]
        delta_r6d = delta[..., a0 + 3:a0 + 9]

        # WORLD-frame translation: plain addition.
        result[..., a0:a0 + 3] = state_xyz + delta_xyz

        # WORLD-frame/extrinsic rotation: R_abs = R_delta @ R_state.
        Rs_delta = rot6ds_to_matrices(delta_r6d)        # (..., 3, 3)
        Rs_state = to_matrices(state_rot)               # (..., 3, 3)
        Rs_abs = Rs_delta @ Rs_state                     # (..., 3, 3) — broadcasts (3,3) case natively

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_abs)
        # gripper unchanged

    return result


def ee_obs_relative_forward(
    obs_abs: np.ndarray,
    anchor: np.ndarray,
    observation_encoding: str = "quaternion",
) -> np.ndarray:
    """Convert absolute EE observations to SE(3)-relative, rot6d layout.

    Matches UMI's obs relativisation: ``T_rel = inv(T_anchor) @ T_obs``.
    Input obs use ``observation_encoding`` layout, output always uses rot6d
    (10 dims/arm — the network-facing layout regardless of the dataset's
    on-disk observation_encoding).

    Per arm:
        body_delta  = R_anchor.T @ (obs_xyz - anchor_xyz)
        rel_rot6d   = matrices_to_rot6d(R_anchor.T @ R_obs)
        gripper     = obs_gripper  (kept absolute)

    Parameters
    ----------
    obs_abs:
        Absolute EE observations in ``observation_encoding`` layout.
        Shape ``(..., state_dim_per_arm * n_arms)``.
    anchor:
        Reference EE state (same ``observation_encoding`` layout) to
        relativise against. Either ``(state_dim_per_arm * n_arms,)`` —
        single anchor (broadcast) or ``(..., state_dim_per_arm * n_arms)``
        — per-sample anchor (same leading dims as obs_abs).
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

    Returns
    -------
    np.ndarray
        Relative observations in rot6d layout, shape ``(..., 10 * n_arms)``.
    """
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)

    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    if anchor.shape[-1] <= 0 or anchor.shape[-1] % state_dim_per_arm != 0:
        raise ValueError(
            f"ee_obs_relative_forward: anchor dim {anchor.shape[-1]} is not a positive "
            f"multiple of {state_dim_per_arm} ({observation_encoding!r} per-arm dim)."
        )
    n_arms = anchor.shape[-1] // state_dim_per_arm
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrix = _STATE_ROTATION_TO_MATRIX[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    out_shape = obs_abs.shape[:-1] + (n_arms * EE_ACTION_DIM_PER_ARM,)
    result = np.empty(out_shape, dtype=np.float64)
    per_sample_anchor = anchor.ndim > 1

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        obs_xyz = obs_abs[..., s0:s0 + 3]
        obs_rot = obs_abs[..., s0 + 3:s0 + 3 + rot_dim]
        obs_grip = obs_abs[..., s0 + 3 + rot_dim:s0 + 4 + rot_dim]
        anchor_xyz = anchor[..., s0:s0 + 3]
        anchor_rot = anchor[..., s0 + 3:s0 + 3 + rot_dim]

        Rs_obs = to_matrices(obs_rot)          # (..., 3, 3)
        world_delta = obs_xyz - anchor_xyz

        if per_sample_anchor:
            Rs_anchor = to_matrices(anchor_rot)               # (..., 3, 3)
            Rs_anchor_T = Rs_anchor.swapaxes(-2, -1)
            Rs_rel = Rs_anchor_T @ Rs_obs
            result[..., a0:a0 + 3] = np.einsum('...ij,...j->...i', Rs_anchor_T, world_delta)
        else:
            R_anchor = to_matrix(anchor_rot)                  # (3, 3)
            Rs_rel = R_anchor.T @ Rs_obs                      # (..., 3, 3)
            result[..., a0:a0 + 3] = world_delta @ R_anchor   # row-vector: R_anchor.T applied

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)
        result[..., a0 + 9:a0 + 10] = obs_grip

    return result


def ee_obs_abs_forward(obs_abs: np.ndarray, observation_encoding: str = "quaternion") -> np.ndarray:
    """Convert absolute EE observations from ``observation_encoding`` layout to rot6d layout.

    Input obs use ``observation_encoding`` layout, output always uses rot6d
    (10 dims/arm — the network-facing layout regardless of the dataset's
    on-disk observation_encoding). No SE(3) relative computation — xyz and
    gripper are passthrough; only rotation is re-encoded.

    Per arm:
        xyz     = obs_xyz  (passthrough)
        rot6d   = matrices_to_rot6d(<obs rotation, decoded per observation_encoding>)
        gripper = obs_gripper  (kept absolute)

    Parameters
    ----------
    obs_abs:
        Absolute EE observations in ``observation_encoding`` layout.
        Shape ``(..., state_dim_per_arm * n_arms)``.
    observation_encoding:
        The dataset's ``observation.state`` rotation encoding
        (quaternion/rot6d/axis_angle). Defaults to "quaternion" for callers
        that predate this parameter.

    Returns
    -------
    np.ndarray
        Absolute observations in rot6d layout, shape ``(..., 10 * n_arms)``.
    """
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    state_dim_per_arm = observation_state_dim_per_arm(observation_encoding)
    if obs_abs.shape[-1] <= 0 or obs_abs.shape[-1] % state_dim_per_arm != 0:
        raise ValueError(
            f"ee_obs_abs_forward: obs dim {obs_abs.shape[-1]} is not a positive multiple "
            f"of {state_dim_per_arm} ({observation_encoding!r} per-arm dim)."
        )
    n_arms = obs_abs.shape[-1] // state_dim_per_arm
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    to_matrices = _STATE_ROTATIONS_TO_MATRICES[observation_encoding]

    out_shape = obs_abs.shape[:-1] + (n_arms * EE_ACTION_DIM_PER_ARM,)
    result = np.empty(out_shape, dtype=np.float64)

    for arm in range(n_arms):
        s0 = arm * state_dim_per_arm
        a0 = arm * EE_ACTION_DIM_PER_ARM

        obs_xyz = obs_abs[..., s0:s0 + 3]
        obs_rot = obs_abs[..., s0 + 3:s0 + 3 + rot_dim]
        obs_grip = obs_abs[..., s0 + 3 + rot_dim:s0 + 4 + rot_dim]

        Rs_obs = to_matrices(obs_rot)          # (..., 3, 3)
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
