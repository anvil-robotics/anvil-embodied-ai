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

Axis-angle (a.k.a. rotation vector) encodes a rotation as a single 3-vector
whose direction is the rotation axis and whose norm is the rotation angle
(radians). This is the convention robosuite/LIBERO use for their
end-effector orientation state and action. The axis-angle <-> matrix
functions here are implemented via the quaternion round-trip
(:func:`matrix_to_quat`/:func:`quat_to_matrix`) rather than the textbook
Rodrigues formula directly, because the direct matrix<->axis-angle formulas
have a numerically ill-conditioned branch near angle=pi (exactly the regime
LIBERO's data lives in — a downward-facing gripper is ~pi rotation from
identity) which Shepperd's method (already used for quaternions here)
already handles robustly.
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
# Axis-angle utilities (robosuite / LIBERO convention)
# =============================================================================


def _quat_to_axis_angle(quat_xyzw) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` -> axis-angle 3-vector (angle = norm, radians)."""
    q = np.asarray(quat_xyzw, dtype=float).reshape(-1)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    qxyz, qw = q[:3], q[3]
    # Canonicalize to the qw >= 0 hemisphere so the recovered angle stays in
    # [0, pi] (a quaternion and its negation represent the same rotation).
    if qw < 0:
        qxyz, qw = -qxyz, -qw
    sin_half = np.linalg.norm(qxyz)
    angle = 2.0 * np.arctan2(sin_half, qw)
    if sin_half < 1e-10:
        return np.zeros(3, dtype=float)
    return (qxyz / sin_half) * angle


def _axis_angle_to_quat(aa) -> np.ndarray:
    """Axis-angle 3-vector (angle = norm, radians) -> quaternion ``[x, y, z, w]``."""
    v = np.asarray(aa, dtype=float).reshape(-1)
    angle = np.linalg.norm(v)
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = v / angle
    half = angle / 2.0
    return np.array([*(axis * np.sin(half)), np.cos(half)], dtype=float)


def matrix_to_axis_angle(R) -> np.ndarray:
    """Convert a 3×3 rotation matrix to an axis-angle 3-vector.

    Routed through :func:`matrix_to_quat` (Shepperd's method) for numerical
    stability near angle=pi rather than the direct Rodrigues inverse.
    """
    return _quat_to_axis_angle(matrix_to_quat(R))


def axis_angle_to_matrix(aa) -> np.ndarray:
    """Convert an axis-angle 3-vector to a 3×3 rotation matrix."""
    return quat_to_matrix(_axis_angle_to_quat(aa))


def matrices_to_axis_angles(Rs) -> np.ndarray:
    """Batch counterpart of :func:`matrix_to_axis_angle`. Supports arbitrary leading dims."""
    Rs = np.asarray(Rs, dtype=float)
    if Rs.shape[-2:] != (3, 3):
        raise ValueError(f"matrices_to_axis_angles: last two dims must be (3,3), got {Rs.shape}")
    quats = matrices_to_quats(Rs)  # (..., 4)
    return _quats_to_axis_angles(quats)


def axis_angles_to_matrices(aas) -> np.ndarray:
    """Batch counterpart of :func:`axis_angle_to_matrix`. Supports arbitrary leading dims."""
    return quats_to_matrices(_axis_angles_to_quats(aas))


def _quats_to_axis_angles(quats) -> np.ndarray:
    q = np.asarray(quats, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(n > 0, n, 1.0)
    qxyz, qw = q[..., :3], q[..., 3]
    flip = qw < 0
    qxyz = np.where(flip[..., None], -qxyz, qxyz)
    qw = np.where(flip, -qw, qw)
    sin_half = np.linalg.norm(qxyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, qw)
    safe_sin_half = np.where(sin_half > 1e-10, sin_half, 1.0)
    axis = qxyz / safe_sin_half[..., None]
    aa = axis * angle[..., None]
    return np.where((sin_half > 1e-10)[..., None], aa, 0.0)


def _axis_angles_to_quats(aas) -> np.ndarray:
    v = np.asarray(aas, dtype=float)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_angle = np.where(angle > 1e-10, angle, 1.0)
    axis = v / safe_angle
    half = angle / 2.0
    qxyz = axis * np.sin(half)
    qw = np.cos(half)
    quat = np.concatenate([qxyz, qw], axis=-1)
    identity = np.zeros(quat.shape, dtype=float)
    identity[..., 3] = 1.0
    return np.where((angle > 1e-10), quat, identity)
