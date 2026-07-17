"""Rotation utilities for EE Cartesian-space action representations.

Quaternion convention is ``[x, y, z, w]`` (matches ROS / TF2 and the
existing :class:`lerobot.utils.rotation.Rotation` class). All functions
accept numpy arrays or any array-like; inputs are converted to
``float64`` internally.

The 6D rotation representation (Zhou et al. 2019) takes the first two
columns of the 3×3 rotation matrix and flattens them column-major into a
6-vector::

    rot6d = [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]

This is preferred over Euler / quaternion for regression targets in
diffusion policies because it has no discontinuities and is easy to
reconstruct (Gram-Schmidt of the first two columns + cross product for
the third).

The axis-angle representation is a 3-vector ``angle * unit_axis`` (a.k.a. rotation vector /
Rodrigues vector). Unlike rot6d, extracting it FROM a matrix has two genuine numerical
singularities — ``angle ≈ 0`` (axis undefined; handled by returning the zero vector
directly) and ``angle ≈ π`` (the direct skew-symmetric extraction is a 0/0 division there;
handled by recovering the axis from the diagonal of ``(R + I) / 2`` instead, which equals
``axis ⊗ axis`` at that angle). Going the other direction (axis-angle → matrix, via
Rodrigues' rotation formula) has no singularity at any angle.
"""
from __future__ import annotations

import numpy as np


def quat_to_matrix(quat_xyzw) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to a 3×3 rotation matrix.

    Input is normalised internally; unit-norm is not required.
    """
    q = np.asarray(quat_xyzw, dtype=float).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"quat_to_matrix expects shape (4,), got {q.shape}")
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def matrix_to_quat(R) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion ``[x, y, z, w]``.

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_quat expects shape (3,3), got {R.shape}")
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=float)


def matrix_to_rot6d(R) -> np.ndarray:
    """Flatten the first two columns of a 3×3 rotation matrix into a 6-vector."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_rot6d expects shape (3,3), got {R.shape}")
    return R[:, :2].T.reshape(6).astype(float)


def rot6d_to_matrix(r6d) -> np.ndarray:
    """Reconstruct a 3×3 rotation matrix from a 6-vector via Gram-Schmidt.

    Inverse of :func:`matrix_to_rot6d` up to numerical precision.
    """
    v = np.asarray(r6d, dtype=float).reshape(-1)
    if v.shape != (6,):
        raise ValueError(f"rot6d_to_matrix expects shape (6,), got {v.shape}")
    a1, a2 = v[:3], v[3:6]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-10:
        raise ValueError(
            f"rot6d_to_matrix: first column (a1={a1}) has near-zero norm {n1:.2e}; "
            "input is degenerate and cannot be orthonormalized."
        )
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-10:
        raise ValueError(
            f"rot6d_to_matrix: second column after projection has near-zero norm {n2:.2e}; "
            "a1 and a2 are nearly parallel — input is degenerate."
        )
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


# =============================================================================
# Batch rotation utilities (arbitrary leading dimensions)
# =============================================================================


def quats_to_matrices(quats) -> np.ndarray:
    """Convert (..., 4) quaternions ``[x, y, z, w]`` to (..., 3, 3) rotation matrices.

    Inputs are normalised internally.  Supports arbitrary leading batch dimensions.
    """
    q = np.asarray(quats, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"quats_to_matrices: last dim must be 4, got {q.shape}")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(n > 0, n, 1.0)
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[..., 0, 1] = 2 * (qx * qy - qz * qw)
    R[..., 0, 2] = 2 * (qx * qz + qy * qw)
    R[..., 1, 0] = 2 * (qx * qy + qz * qw)
    R[..., 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[..., 1, 2] = 2 * (qy * qz - qx * qw)
    R[..., 2, 0] = 2 * (qx * qz - qy * qw)
    R[..., 2, 1] = 2 * (qy * qz + qx * qw)
    R[..., 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def matrices_to_quats(Rs) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 4) quaternions ``[x, y, z, w]``.

    Uses a vectorised Shepperd's method (branch-free via ``np.where``).
    Supports arbitrary leading batch dimensions.
    """
    Rs = np.asarray(Rs, dtype=float)
    if Rs.shape[-2:] != (3, 3):
        raise ValueError(f"matrices_to_quats: last two dims must be (3,3), got {Rs.shape}")

    trace = Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]

    # Branch 1: trace > 0  (large qw)
    s1 = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2  # 4 * qw
    d1 = np.where(np.abs(s1) > 1e-10, s1, 1.0)
    qw1 = 0.25 * s1
    qx1 = (Rs[..., 2, 1] - Rs[..., 1, 2]) / d1
    qy1 = (Rs[..., 0, 2] - Rs[..., 2, 0]) / d1
    qz1 = (Rs[..., 1, 0] - Rs[..., 0, 1]) / d1

    # Branch 2: R[0,0] dominant  (large qx)
    s2 = np.sqrt(np.maximum(1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2], 0.0)) * 2
    d2 = np.where(np.abs(s2) > 1e-10, s2, 1.0)
    qw2 = (Rs[..., 2, 1] - Rs[..., 1, 2]) / d2
    qx2 = 0.25 * s2
    qy2 = (Rs[..., 0, 1] + Rs[..., 1, 0]) / d2
    qz2 = (Rs[..., 0, 2] + Rs[..., 2, 0]) / d2

    # Branch 3: R[1,1] dominant  (large qy)
    s3 = np.sqrt(np.maximum(1.0 + Rs[..., 1, 1] - Rs[..., 0, 0] - Rs[..., 2, 2], 0.0)) * 2
    d3 = np.where(np.abs(s3) > 1e-10, s3, 1.0)
    qw3 = (Rs[..., 0, 2] - Rs[..., 2, 0]) / d3
    qx3 = (Rs[..., 0, 1] + Rs[..., 1, 0]) / d3
    qy3 = 0.25 * s3
    qz3 = (Rs[..., 1, 2] + Rs[..., 2, 1]) / d3

    # Branch 4: R[2,2] dominant  (large qz)
    s4 = np.sqrt(np.maximum(1.0 + Rs[..., 2, 2] - Rs[..., 0, 0] - Rs[..., 1, 1], 0.0)) * 2
    d4 = np.where(np.abs(s4) > 1e-10, s4, 1.0)
    qw4 = (Rs[..., 1, 0] - Rs[..., 0, 1]) / d4
    qx4 = (Rs[..., 0, 2] + Rs[..., 2, 0]) / d4
    qy4 = (Rs[..., 1, 2] + Rs[..., 2, 1]) / d4
    qz4 = 0.25 * s4

    # Select branch per sample
    b1 = trace > 0
    b2 = ~b1 & (Rs[..., 0, 0] > Rs[..., 1, 1]) & (Rs[..., 0, 0] > Rs[..., 2, 2])
    b3 = ~b1 & ~b2 & (Rs[..., 1, 1] > Rs[..., 2, 2])
    # b4: everything else

    qw = np.where(b1, qw1, np.where(b2, qw2, np.where(b3, qw3, qw4)))
    qx = np.where(b1, qx1, np.where(b2, qx2, np.where(b3, qx3, qx4)))
    qy = np.where(b1, qy1, np.where(b2, qy2, np.where(b3, qy3, qy4)))
    qz = np.where(b1, qz1, np.where(b2, qz2, np.where(b3, qz3, qz4)))

    return np.stack([qx, qy, qz, qw], axis=-1)


def matrices_to_rot6d(Rs) -> np.ndarray:
    """Flatten the first two columns of (..., 3, 3) rotation matrices into (..., 6) rot6d.

    Batch counterpart of :func:`matrix_to_rot6d`.  Supports arbitrary leading dims.
    """
    Rs = np.asarray(Rs, dtype=float)
    if Rs.shape[-2:] != (3, 3):
        raise ValueError(f"matrices_to_rot6d: last two dims must be (3,3), got {Rs.shape}")
    # First two columns: Rs[..., :, :2] → (..., 3, 2)
    # Transpose last two dims → (..., 2, 3), then reshape → (..., 6)
    col01 = Rs[..., :, :2]  # (..., 3, 2)
    return col01.swapaxes(-2, -1).reshape(Rs.shape[:-2] + (6,)).astype(float)


def rot6ds_to_matrices(r6ds) -> np.ndarray:
    """Reconstruct (..., 3, 3) rotation matrices from (..., 6) rot6d vectors via Gram-Schmidt.

    Batch counterpart of :func:`rot6d_to_matrix`.  Degenerate inputs (near-zero
    columns) are handled gracefully — columns are clamped to a small norm rather
    than raising, to avoid masking bugs in downstream code.
    """
    v = np.asarray(r6ds, dtype=float)
    if v.shape[-1] != 6:
        raise ValueError(f"rot6ds_to_matrices: last dim must be 6, got {v.shape}")
    a1 = v[..., :3]   # (..., 3)
    a2 = v[..., 3:6]  # (..., 3)

    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)  # (..., 1)
    b1 = a1 / np.where(n1 > 1e-10, n1, 1e-10)

    dot = (b1 * a2).sum(axis=-1, keepdims=True)  # (..., 1)
    b2_raw = a2 - dot * b1
    n2 = np.linalg.norm(b2_raw, axis=-1, keepdims=True)
    b2 = b2_raw / np.where(n2 > 1e-10, n2, 1e-10)

    b3 = np.cross(b1, b2)  # (..., 3)
    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)


# =============================================================================
# Axis-angle utilities
# =============================================================================

_AXIS_ANGLE_NEAR_ZERO = 1e-8
_AXIS_ANGLE_NEAR_PI = np.pi - 1e-6


def matrix_to_axis_angle(R) -> np.ndarray:
    """Convert a 3×3 rotation matrix to an axis-angle vector (``angle * unit_axis``).

    Two degenerate regimes are handled with explicit branches, not a single epsilon guard:

    - ``angle ≈ 0`` (identity): the axis is undefined but the output is unambiguous — the
      zero vector, no division needed.
    - ``angle ≈ π``: the skew-symmetric extraction (``1/(2 sin(angle))`` times the
      antisymmetric part of ``R``) is a 0/0 division there, since ``sin(π) = 0``. The axis
      is instead recovered from the diagonal of ``(R + I) / 2``, which equals
      ``axis ⊗ axis`` at this angle (standard identity: at ``angle=π``,
      ``R = 2 · axis⊗axis - I``). Signs are disambiguated from the off-diagonal terms,
      anchored on the largest-magnitude axis component for numerical stability. The sign of
      the whole axis is otherwise arbitrary at ``angle=π`` (``(axis, π)`` and
      ``(-axis, π)`` represent the same rotation) — this is a deterministic tie-break, not
      a correctness concern.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_axis_angle expects shape (3,3), got {R.shape}")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if angle < _AXIS_ANGLE_NEAR_ZERO:
        return np.zeros(3, dtype=float)

    if angle > _AXIS_ANGLE_NEAR_PI:
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        for i in range(3):
            if i != k and B[k, i] < 0:
                axis[i] = -axis[i]
        return axis * angle

    axis = (1.0 / (2.0 * np.sin(angle))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    return axis * angle


def axis_angle_to_matrix(v) -> np.ndarray:
    """Convert an axis-angle vector (``angle * unit_axis``) to a 3×3 rotation matrix via
    Rodrigues' rotation formula. No singularity at any angle, including π — this direction
    is well-defined everywhere, unlike :func:`matrix_to_axis_angle`.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"axis_angle_to_matrix expects shape (3,), got {v.shape}")
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z = v / angle
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrices_to_axis_angles(Rs) -> np.ndarray:
    """Batch counterpart of :func:`matrix_to_axis_angle`. Supports arbitrary leading dims.

    Per-sample branch selection (near-zero / near-π / generic) via ``np.where``, matching
    this file's existing vectorization convention (see :func:`matrices_to_quats`) — not a
    Python-level loop.
    """
    Rs = np.asarray(Rs, dtype=float)
    if Rs.shape[-2:] != (3, 3):
        raise ValueError(f"matrices_to_axis_angles: last two dims must be (3,3), got {Rs.shape}")

    trace = Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)  # (...,)

    near_zero = angle < _AXIS_ANGLE_NEAR_ZERO
    near_pi = angle > _AXIS_ANGLE_NEAR_PI
    generic = ~near_zero & ~near_pi

    # Generic branch: skew-symmetric extraction. Guard the denominator everywhere (not just
    # where `generic` holds) so non-generic entries never divide by ~0 before being
    # overwritten by np.where below.
    sin_angle = np.sin(angle)
    safe_sin = np.where(generic, sin_angle, 1.0)
    axis_generic = np.stack([
        Rs[..., 2, 1] - Rs[..., 1, 2],
        Rs[..., 0, 2] - Rs[..., 2, 0],
        Rs[..., 1, 0] - Rs[..., 0, 1],
    ], axis=-1) / (2.0 * safe_sin)[..., None]

    # near-pi branch: axis from the diagonal of (R + I) / 2 = axis ⊗ axis at angle = π.
    B = (Rs + np.eye(3)) / 2.0
    axis_pi_mag = np.sqrt(np.clip(np.diagonal(B, axis1=-2, axis2=-1), 0.0, None))  # (..., 3)
    k = np.argmax(axis_pi_mag, axis=-1)  # (...,)
    B_k_row = np.take_along_axis(B, k[..., None, None], axis=-2)[..., 0, :]  # (..., 3)
    signs = np.where(B_k_row < 0, -1.0, 1.0)
    axis_pi = axis_pi_mag * signs

    axis = np.where(
        near_zero[..., None], np.zeros_like(axis_generic),
        np.where(near_pi[..., None], axis_pi, axis_generic),
    )
    return axis * angle[..., None]


def axis_angles_to_matrices(vs) -> np.ndarray:
    """Batch counterpart of :func:`axis_angle_to_matrix`. Supports arbitrary leading dims."""
    vs = np.asarray(vs, dtype=float)
    if vs.shape[-1] != 3:
        raise ValueError(f"axis_angles_to_matrices: last dim must be 3, got {vs.shape}")

    angle = np.linalg.norm(vs, axis=-1)  # (...,)
    is_identity = angle < 1e-12
    safe_angle = np.where(is_identity, 1.0, angle)
    axis = vs / safe_angle[..., None]  # (..., 3) — garbage where is_identity, overwritten below

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack([
        np.stack([zeros, -z, y], axis=-1),
        np.stack([z, zeros, -x], axis=-1),
        np.stack([-y, x, zeros], axis=-1),
    ], axis=-2)  # (..., 3, 3)

    sin_a = np.sin(angle)[..., None, None]
    cos_term = (1.0 - np.cos(angle))[..., None, None]
    R = np.eye(3) + sin_a * K + cos_term * (K @ K)

    return np.where(is_identity[..., None, None], np.broadcast_to(np.eye(3), R.shape), R)
