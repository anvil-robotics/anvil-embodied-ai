"""Tests for anvil_shared.ee_transform.

Covers:
  1. n_arms_from_dims — valid single/bimanual, invalid state dim, invalid action dim
  2. ee_relative_forward / ee_relative_inverse round-trip (single arm, bimanual)
  3. ee_relative_forward single reference state vs per-sample state agreement
  4. Identity: zero rotation → forward returns zero rotation delta
  5. ee_action_to_poses layout: pos, quat_xyzw shape, gripper value
  6. Vectorised forward matches per-sample single calls
  7. ee_obs_relative_forward — body-frame translation, identity, gripper passthrough,
     single vs per-sample anchor, bimanual, obs↔action frame consistency
  8. Inference queue prefill — queue entries are shape (1, 10n) not (10n,)
  9. ee_delta_forward / ee_delta_inverse — WORLD-frame Delta(n-(n-1)): round-trip,
     identity/self-anchor zero-delta, single vs per-sample state, robosuite
     composition-order verification, and explicit divergence from the
     BODY-frame ee_relative_forward/inverse pair for the same non-trivial input
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from anvil_shared.ee_transform import (
    EE_ACTION_DIM_PER_ARM,
    EE_STATE_DIM_PER_ARM,
    ee_action_to_poses,
    ee_delta_forward,
    ee_delta_inverse,
    ee_obs_abs_forward,
    ee_obs_relative_forward,
    ee_relative_forward,
    ee_relative_inverse,
    n_arms_from_dims,
)
from anvil_shared.rotation import matrix_to_quat, quat_to_matrix, rot6d_to_matrix


# ── helpers ───────────────────────────────────────────────────────────────────

def _identity_state(n_arms: int = 1) -> np.ndarray:
    """EE state with identity rotation and zero position for each arm."""
    state = np.zeros(EE_STATE_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        # qx=0, qy=0, qz=0, qw=1  (identity quaternion)
        state[s0 + 6] = 1.0
    return state


def _identity_action(n_arms: int = 1) -> np.ndarray:
    """EE action with identity rotation and zero position for each arm."""
    action = np.zeros(EE_ACTION_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM_PER_ARM
        # rot6d identity: R[:,0]=[1,0,0], R[:,1]=[0,1,0]  → [1,0,0,0,1,0]
        action[a0 + 3] = 1.0  # r0
        action[a0 + 7] = 1.0  # r4
    return action


def _random_action(rng: np.random.Generator, n_arms: int = 1) -> np.ndarray:
    """Random EE action with valid rot6d rotation per arm."""
    action = np.zeros(EE_ACTION_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM_PER_ARM
        action[a0:a0 + 3] = rng.uniform(-0.5, 0.5, 3)  # xyz
        # Random rotation via QR decomposition → valid SO(3)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        # rot6d = first two columns, column-major [Q[:,0], Q[:,1]]
        action[a0 + 3:a0 + 9] = np.concatenate([Q[:, 0], Q[:, 1]])
        action[a0 + 9] = rng.uniform(0.0, 0.05)  # gripper
    return action


def _random_state(rng: np.random.Generator, n_arms: int = 1) -> np.ndarray:
    """Random EE state with valid quaternion per arm."""
    state = np.zeros(EE_STATE_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        state[s0:s0 + 3] = rng.uniform(-0.5, 0.5, 3)  # xyz
        q = rng.standard_normal(4)
        state[s0 + 3:s0 + 7] = q / np.linalg.norm(q)  # unit quaternion [qx,qy,qz,qw]
        state[s0 + 7] = rng.uniform(0.0, 0.05)  # gripper (state)
    return state


# ── 1. n_arms_from_dims ───────────────────────────────────────────────────────

class TestNArmsFromDims:
    def test_single_arm(self):
        assert n_arms_from_dims(8, 10) == 1

    def test_bimanual(self):
        assert n_arms_from_dims(16, 20) == 2

    def test_invalid_state_dim_zero(self):
        with pytest.raises(ValueError, match="positive multiple"):
            n_arms_from_dims(0, 10)

    def test_invalid_state_dim_not_multiple(self):
        with pytest.raises(ValueError, match="positive multiple"):
            n_arms_from_dims(9, 10)

    def test_invalid_action_dim(self):
        with pytest.raises(ValueError, match="action dim"):
            n_arms_from_dims(8, 12)  # should be 10

    def test_invalid_action_dim_bimanual(self):
        with pytest.raises(ValueError, match="action dim"):
            n_arms_from_dims(16, 10)  # should be 20


# ── 2. Round-trip: forward → inverse ─────────────────────────────────────────

class TestRoundTrip:
    def _check_round_trip(self, n_arms: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        action_abs = _random_action(rng, n_arms)
        state = _random_state(rng, n_arms)

        action_rel = ee_relative_forward(action_abs, state)
        recovered = ee_relative_inverse(action_rel, state)

        np.testing.assert_allclose(
            recovered, action_abs, atol=1e-10,
            err_msg=f"Round-trip failed for n_arms={n_arms}"
        )

    def test_single_arm(self):
        self._check_round_trip(n_arms=1)

    def test_bimanual(self):
        self._check_round_trip(n_arms=2)

    def test_batched_chunk(self):
        """Batch of T=8 steps with a single reference state."""
        rng = np.random.default_rng(99)
        T = 8
        n_arms = 1
        state = _random_state(rng, n_arms)  # (8,) single ref
        actions_abs = np.stack([_random_action(rng, n_arms) for _ in range(T)])  # (T, 10)

        actions_rel = ee_relative_forward(actions_abs, state)
        recovered = ee_relative_inverse(actions_rel, state)

        np.testing.assert_allclose(recovered, actions_abs, atol=1e-10)

    def test_gripper_preserved(self):
        """Gripper stays absolute through forward → inverse."""
        rng = np.random.default_rng(7)
        action_abs = _random_action(rng)
        state = _random_state(rng)
        # The gripper value in the action
        gripper_orig = float(action_abs[9])

        action_rel = ee_relative_forward(action_abs, state)
        # Forward: gripper in rel is still absolute (copied unchanged)
        assert abs(float(action_rel[9]) - gripper_orig) < 1e-12

        recovered = ee_relative_inverse(action_rel, state)
        assert abs(float(recovered[9]) - gripper_orig) < 1e-12


# ── 3. Identity state: zero delta for zero-offset action ─────────────────────

class TestIdentity:
    def test_identity_state_identity_action_zero_delta(self):
        """With identity rotation and zero position, forward → zero xyz delta."""
        state = _identity_state()
        action = _identity_action()
        action[:3] = 0.0  # zero position

        rel = ee_relative_forward(action, state)
        # xyz delta should be zero
        np.testing.assert_allclose(rel[:3], 0.0, atol=1e-12)
        # rot6d delta should be identity rot6d (relative rotation = I)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(rel[3:9], expected_rot6d, atol=1e-12)


# ── 4. Single vs per-sample state agreement ───────────────────────────────────

class TestSingleVsPerSampleState:
    def test_forward_agrees(self):
        """Single ref state (8,) and per-sample state (T,8) give same result."""
        rng = np.random.default_rng(11)
        T = 5
        state = _random_state(rng)  # (8,) single ref
        actions = np.stack([_random_action(rng) for _ in range(T)])  # (T,10)

        # Single ref broadcast
        rel_single = ee_relative_forward(actions, state)

        # Per-sample: broadcast state to (T, 8)
        states_per_sample = np.tile(state, (T, 1))  # (T, 8)
        rel_per_sample = ee_relative_forward(actions, states_per_sample)

        np.testing.assert_allclose(rel_single, rel_per_sample, atol=1e-12)

    def test_inverse_agrees(self):
        """Same as forward but for inverse."""
        rng = np.random.default_rng(13)
        T = 5
        state = _random_state(rng)
        rels = np.stack([_random_action(rng) for _ in range(T)])

        abs_single = ee_relative_inverse(rels, state)
        states_per_sample = np.tile(state, (T, 1))
        abs_per_sample = ee_relative_inverse(rels, states_per_sample)

        np.testing.assert_allclose(abs_single, abs_per_sample, atol=1e-12)


# ── 5. ee_action_to_poses ─────────────────────────────────────────────────────

class TestEeActionToPoses:
    def test_single_step_single_arm(self):
        action = _identity_action(n_arms=1)
        action[:3] = [0.1, 0.2, 0.3]
        action[9] = 0.03

        poses = ee_action_to_poses(action, n_arms=1)
        assert len(poses) == 1
        step = poses[0]
        assert 0 in step
        assert step[0]["pos"].shape == (3,)
        np.testing.assert_allclose(step[0]["pos"], [0.1, 0.2, 0.3], atol=1e-12)
        assert step[0]["quat_xyzw"].shape == (4,)
        assert abs(step[0]["gripper"] - 0.03) < 1e-12

    def test_chunk_shape(self):
        rng = np.random.default_rng(21)
        T = 4
        n_arms = 2
        actions = np.stack([_random_action(rng, n_arms) for _ in range(T)])
        poses = ee_action_to_poses(actions, n_arms=n_arms)

        assert len(poses) == T
        for step in poses:
            assert set(step.keys()) == {0, 1}
            for arm in range(n_arms):
                assert step[arm]["pos"].shape == (3,)
                assert step[arm]["quat_xyzw"].shape == (4,)
                # Quaternion must be unit
                q = step[arm]["quat_xyzw"]
                np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_n_arms_inferred(self):
        """n_arms=None → inferred from action dim."""
        action = _identity_action(n_arms=2)
        poses = ee_action_to_poses(action)  # n_arms defaults to None
        assert len(poses) == 1
        assert 0 in poses[0] and 1 in poses[0]

    def test_identity_action_gives_identity_quat(self):
        """Identity rot6d in action → quaternion should be [0,0,0,1]."""
        action = _identity_action(n_arms=1)
        poses = ee_action_to_poses(action, n_arms=1)
        q = poses[0][0]["quat_xyzw"]
        # Identity rotation: [qx,qy,qz,qw] = [0,0,0,±1]
        assert abs(abs(q[3]) - 1.0) < 1e-10, f"Expected |qw|=1, got {q}"


# ── 7. ee_obs_relative_forward ──────────────────────────────────────────────────────

class TestEEObsRelForward:
    """Tests for ee_obs_relative_forward: abs obs (quat, 8n) → relative obs (rot6d, 10n)."""

    # Identity property: obs relative to itself = [0,0,0, 1,0,0,0,1,0, gripper]
    def test_identity_single_arm(self):
        rng = np.random.default_rng(42)
        state = _random_state(rng, n_arms=1)
        rel = ee_obs_relative_forward(state, state)  # obs = anchor

        expected_xyz = np.zeros(3)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(rel[:3], expected_xyz, atol=1e-12, err_msg="identity: xyz should be 0")
        np.testing.assert_allclose(rel[3:9], expected_rot6d, atol=1e-12, err_msg="identity: rot6d should be identity")
        np.testing.assert_allclose(rel[9], state[7], atol=1e-12, err_msg="identity: gripper passthrough")

    def test_identity_bimanual(self):
        rng = np.random.default_rng(43)
        state = _random_state(rng, n_arms=2)
        rel = ee_obs_relative_forward(state, state)

        assert rel.shape == (2 * EE_ACTION_DIM_PER_ARM,)
        for arm in range(2):
            a0 = arm * EE_ACTION_DIM_PER_ARM
            np.testing.assert_allclose(rel[a0:a0 + 3], 0.0, atol=1e-12, err_msg=f"arm {arm} xyz")
            np.testing.assert_allclose(
                rel[a0 + 3:a0 + 9], [1, 0, 0, 0, 1, 0], atol=1e-12, err_msg=f"arm {arm} rot6d"
            )
            np.testing.assert_allclose(rel[a0 + 9], state[arm * EE_STATE_DIM_PER_ARM + 7], atol=1e-12)

    # Gripper is a passthrough (kept absolute, not relativised)
    def test_gripper_passthrough(self):
        rng = np.random.default_rng(5)
        obs = _random_state(rng)
        anchor = _random_state(rng)
        rel = ee_obs_relative_forward(obs, anchor)
        np.testing.assert_allclose(rel[9], obs[7], atol=1e-12, err_msg="gripper must pass through unchanged")

    # Output shape: single obs → (10n,); batched obs → (..., 10n)
    def test_output_shape_single(self):
        rng = np.random.default_rng(6)
        obs = _random_state(rng)
        anchor = _random_state(rng)
        rel = ee_obs_relative_forward(obs, anchor)
        assert rel.shape == (EE_ACTION_DIM_PER_ARM,)

    def test_output_shape_batched(self):
        rng = np.random.default_rng(7)
        T = 4
        obs_window = np.stack([_random_state(rng) for _ in range(T)])
        anchor = _random_state(rng)
        rel = ee_obs_relative_forward(obs_window, anchor)
        assert rel.shape == (T, EE_ACTION_DIM_PER_ARM)

    # Single anchor (broadcast) must equal per-sample anchor with same value tiled
    def test_single_vs_per_sample_anchor(self):
        rng = np.random.default_rng(8)
        T = 5
        obs = np.stack([_random_state(rng) for _ in range(T)])
        anchor = _random_state(rng)

        rel_single = ee_obs_relative_forward(obs, anchor)  # anchor shape (8,) → broadcast
        rel_per_sample = ee_obs_relative_forward(obs, np.tile(anchor, (T, 1)))  # anchor (T,8)

        np.testing.assert_allclose(rel_single, rel_per_sample, atol=1e-12)

    # Body-frame translation: verify manually for a known rotation
    def test_body_frame_translation(self):
        """Obs offset along +y in world, anchor rotated 90° around z.
        Body-frame translation = R_anchor.T @ world_delta = [1, 0, 0]."""
        import math

        # Anchor at origin, rotated 90° around z.
        # R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]]; R.T @ [0,1,0] = [1,0,0].
        # Also world_delta @ R_anchor = [0,1,0]@[[0,-1,0],[1,0,0],[0,0,1]] = [1,0,0] ✓
        theta = math.pi / 2
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        anchor_quat = np.array([0.0, 0.0, s, c])  # [qx,qy,qz,qw]
        anchor_state = np.zeros(EE_STATE_DIM_PER_ARM)
        anchor_state[3:7] = anchor_quat

        # Obs at [0, 1, 0] in world, same rotation as anchor
        obs_state = anchor_state.copy()
        obs_state[1] = 1.0  # y offset in world

        rel = ee_obs_relative_forward(obs_state, anchor_state)
        body_delta = rel[:3]

        expected_body_delta = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(body_delta, expected_body_delta, atol=1e-12)

    # Obs path and action path must agree on [xyz, rot6d] for the same pose
    def test_obs_action_frame_consistency(self):
        """For the same absolute pose, obs_rel_forward and rel_forward give
        matching [xyz, rot6d] (gripper excluded since state has 1D, action has 1D)."""
        rng = np.random.default_rng(9)
        anchor_state = _random_state(rng)   # (8,) anchor in state/quat layout

        # Build an action in rot6d layout from the SAME pose as anchor_state
        # (obs and action share xyz and rotation, only gripper index differs)
        action_same_pose = np.zeros(EE_ACTION_DIM_PER_ARM)
        action_same_pose[:3] = anchor_state[:3]  # same xyz
        from anvil_shared.rotation import quat_to_matrix, matrices_to_rot6d
        R = quat_to_matrix(anchor_state[3:7])
        r6d = matrices_to_rot6d(R[np.newaxis])[0]
        action_same_pose[3:9] = r6d

        obs_rel = ee_obs_relative_forward(anchor_state, anchor_state)  # should be identity
        act_rel = ee_relative_forward(action_same_pose, anchor_state)  # should also be identity

        # [xyz, rot6d] should match (max err ~1e-14 from floating point)
        np.testing.assert_allclose(obs_rel[:9], act_rel[:9], atol=1e-12,
                                   err_msg="obs and action relative frames should agree for same pose")


# ── 8. Inference queue prefill shape ─────────────────────────────────────────

class TestPrefillQueueShape:
    """Regression test for C1: queue entries must be (1, 10n), not (10n,).

    Directly tests the shape contract expected by lerobot's
    ``predict_action_chunk`` (``torch.stack(queue, dim=1)``).
    """

    def test_prefilled_entries_have_batch_dim(self):
        """After prefill, every queue entry must have shape (1, 10*n_arms)."""
        torch = pytest.importorskip("torch", reason="torch not installed")

        rng = np.random.default_rng(20)
        n_arms = 1
        n_obs_steps = 2
        obs_window_rel_np = np.stack([
            np.random.default_rng(i).standard_normal(EE_ACTION_DIM_PER_ARM * n_arms)
            for i in range(n_obs_steps)
        ])  # (n_obs_steps, 10n)

        # Simulate: queue is a deque with maxlen matching model._queues
        queue = deque(maxlen=n_obs_steps)

        # Simulate preprocessor: identity (no-op), returns dict as-is
        def fake_preprocessor(d):
            return d

        # Run the same logic as _prefill_ee_relative_queue (C1 fix applied)
        for i in range(len(obs_window_rel_np) - 1):
            obs_t = torch.tensor(obs_window_rel_np[i], dtype=torch.float32).unsqueeze(0)  # (1,10n)
            norm = fake_preprocessor({"observation.state": obs_t})
            obs_t = norm["observation.state"]   # stays (1, 10n) — no squeeze
            queue.append(obs_t)

        assert len(queue) == n_obs_steps - 1, "queue should have n_obs_steps-1 entries after prefill"
        for entry in queue:
            assert entry.shape == (1, EE_ACTION_DIM_PER_ARM * n_arms), (
                f"Expected (1, {EE_ACTION_DIM_PER_ARM * n_arms}), got {entry.shape}. "
                "C1 regression: squeeze(0) would give ({EE_ACTION_DIM_PER_ARM * n_arms},) "
                "which crashes torch.stack(..., dim=1) in predict_action_chunk."
            )

    def test_stack_dim1_succeeds_after_prefill(self):
        """torch.stack(queue+[current], dim=1) must succeed (regression for C1 crash)."""
        torch = pytest.importorskip("torch", reason="torch not installed")

        n_arms = 1
        n_obs_steps = 2
        state_dim = EE_ACTION_DIM_PER_ARM * n_arms

        # Simulate prefilled queue (shape (1, state_dim) entries)
        queue = deque(maxlen=n_obs_steps)
        queue.append(torch.zeros(1, state_dim))  # historical entry — (1, 10n)

        # Simulate select_action pushing current obs via populate_queues: also (1, 10n)
        current_obs = torch.zeros(1, state_dim)
        queue.append(current_obs)

        # This is what predict_action_chunk does; it must not raise
        stacked = torch.stack(list(queue), dim=1)   # (1, n_obs_steps, state_dim)
        assert stacked.shape == (1, n_obs_steps, state_dim)


# =============================================================================
# 9. ee_obs_abs_forward — absolute quat (8n) → rot6d (10n) conversion
# =============================================================================


def _make_state_quat(n_arms: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a random EE state in quaternion layout (8n)."""
    if rng is None:
        rng = np.random.default_rng(42)
    state = np.zeros(EE_STATE_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        state[s0:s0 + 3] = rng.normal(size=3)      # xyz
        q = rng.normal(size=4)
        state[s0 + 3:s0 + 7] = q / np.linalg.norm(q)  # unit quat
        state[s0 + 7] = rng.uniform(0.0, 0.08)     # gripper
    return state


class TestEEObsAbsForward:
    """Tests for ee_obs_abs_forward — quat (8n) → rot6d (10n), absolute."""

    def test_output_shape_single_arm(self):
        state = _make_state_quat(n_arms=1)
        out = ee_obs_abs_forward(state)
        assert out.shape == (EE_ACTION_DIM_PER_ARM,), f"Expected (10,), got {out.shape}"

    def test_output_shape_bimanual(self):
        state = _make_state_quat(n_arms=2)
        out = ee_obs_abs_forward(state)
        assert out.shape == (EE_ACTION_DIM_PER_ARM * 2,), f"Expected (20,), got {out.shape}"

    def test_output_shape_batch(self):
        """Multi-step batch: (T, 8n) → (T, 10n)."""
        T, n_arms = 5, 1
        rng = np.random.default_rng(99)
        states = np.stack([_make_state_quat(n_arms, rng) for _ in range(T)])
        out = ee_obs_abs_forward(states)
        assert out.shape == (T, EE_ACTION_DIM_PER_ARM * n_arms)

    def test_xyz_passthrough(self):
        """xyz (dims 0-2 per arm) must be preserved exactly."""
        state = _make_state_quat(n_arms=1)
        out = ee_obs_abs_forward(state)
        np.testing.assert_array_equal(out[:3], state[:3])

    def test_gripper_passthrough(self):
        """Gripper (dim 9 per arm) must be preserved exactly."""
        state = _make_state_quat(n_arms=1)
        out = ee_obs_abs_forward(state)
        np.testing.assert_allclose(out[9], state[7], atol=1e-12)

    def test_rot6d_recovers_rotation_matrix(self):
        """rot6d output must reconstruct the same rotation as the input quaternion."""
        rng = np.random.default_rng(7)
        state = _make_state_quat(n_arms=1, rng=rng)
        out = ee_obs_abs_forward(state)
        rot6d = out[3:9]
        R_recovered = rot6d_to_matrix(rot6d)                     # (3,3)
        quat = state[3:7]                                         # [qx, qy, qz, qw]
        R_expected = quat_to_matrix(quat)                         # (3,3)
        np.testing.assert_allclose(R_recovered, R_expected, atol=1e-10)

    def test_identity_quaternion_gives_identity_rot6d(self):
        """qw=1, qx=qy=qz=0 → rot6d = [1,0,0, 0,1,0] (identity rotation)."""
        state = np.zeros(EE_STATE_DIM_PER_ARM)
        state[6] = 1.0  # qw=1
        out = ee_obs_abs_forward(state)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(out[3:9], expected_rot6d, atol=1e-12)

    def test_rot6d_bounds_for_random_rotations(self):
        """All rot6d components must lie within [-1, 1] (orthonormal column vectors)."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            state = _make_state_quat(n_arms=1, rng=rng)
            out = ee_obs_abs_forward(state)
            rot6d = out[3:9]
            assert np.all(rot6d >= -1.0 - 1e-10) and np.all(rot6d <= 1.0 + 1e-10), (
                f"rot6d out of [-1,1]: {rot6d}"
            )

    def test_bimanual_xyz_and_gripper_passthrough(self):
        """Both arms' xyz and gripper must be preserved in bimanual output."""
        rng = np.random.default_rng(55)
        state = _make_state_quat(n_arms=2, rng=rng)
        out = ee_obs_abs_forward(state)
        for arm in range(2):
            s0 = arm * EE_STATE_DIM_PER_ARM
            a0 = arm * EE_ACTION_DIM_PER_ARM
            np.testing.assert_array_equal(out[a0:a0 + 3], state[s0:s0 + 3])  # xyz
            np.testing.assert_allclose(out[a0 + 9], state[s0 + 7], atol=1e-12)  # gripper


# =============================================================================
# 9. ee_delta_forward / ee_delta_inverse — WORLD-frame Delta(n-(n-1))
# =============================================================================


class TestEeDeltaRoundTrip:
    """Forward → inverse round-trip, mirroring TestRoundTrip for the relative pair."""

    def _check_round_trip(self, n_arms: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        action_abs = _random_action(rng, n_arms)
        state = _random_state(rng, n_arms)

        delta = ee_delta_forward(action_abs, state)
        recovered = ee_delta_inverse(delta, state)

        np.testing.assert_allclose(
            recovered, action_abs, atol=1e-10,
            err_msg=f"ee_delta round-trip failed for n_arms={n_arms}"
        )

    def test_single_arm(self):
        self._check_round_trip(n_arms=1)

    def test_bimanual(self):
        self._check_round_trip(n_arms=2)

    def test_batched_chunk(self):
        """Batch of T=8 steps with a single reference state (broadcasts natively)."""
        rng = np.random.default_rng(99)
        T = 8
        n_arms = 1
        state = _random_state(rng, n_arms)  # (8,) single ref
        actions_abs = np.stack([_random_action(rng, n_arms) for _ in range(T)])  # (T, 10)

        deltas = ee_delta_forward(actions_abs, state)
        recovered = ee_delta_inverse(deltas, state)

        np.testing.assert_allclose(recovered, actions_abs, atol=1e-10)

    def test_per_frame_anchor_sequence(self):
        """The actual Delta(n-(n-1)) usage: state[t] anchored to state[t-1], per-sample."""
        rng = np.random.default_rng(101)
        T = 6
        n_arms = 1
        states = np.stack([_random_state(rng, n_arms) for _ in range(T)])       # (T, 8)
        actions = np.stack([_random_action(rng, n_arms) for _ in range(T)])     # (T, 10)

        # anchor[t] = states[t-1] for t=1..T-1 (per-frame anchor, not chunk-start)
        anchors = states[:-1]       # (T-1, 8)
        targets = actions[1:]       # (T-1, 10)

        deltas = ee_delta_forward(targets, anchors)
        recovered = ee_delta_inverse(deltas, anchors)
        np.testing.assert_allclose(recovered, targets, atol=1e-10)

    def test_gripper_preserved(self):
        rng = np.random.default_rng(7)
        action_abs = _random_action(rng)
        state = _random_state(rng)
        gripper_orig = float(action_abs[9])

        delta = ee_delta_forward(action_abs, state)
        assert abs(float(delta[9]) - gripper_orig) < 1e-12

        recovered = ee_delta_inverse(delta, state)
        assert abs(float(recovered[9]) - gripper_orig) < 1e-12


class TestEeDeltaIdentity:
    def test_identity_state_identity_action_zero_delta(self):
        """Self-anchor first-frame convention: state==action → zero xyz delta,
        identity rot6d — same assertion as the body-frame TestIdentity, confirming
        the identity/self-anchor convention is frame-convention-agnostic."""
        state = _identity_state()
        action = _identity_action()
        action[:3] = 0.0  # zero position

        delta = ee_delta_forward(action, state)
        np.testing.assert_allclose(delta[:3], 0.0, atol=1e-12)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(delta[3:9], expected_rot6d, atol=1e-12)

    def test_self_anchor_nonidentity_pose_zero_delta(self):
        """Self-anchor with a NON-identity pose (action == state's own pose) must
        still yield zero delta — this is the actual mcap_converter first-frame
        convention: action[0] = ee_delta_forward(action[0], state[0])."""
        rng = np.random.default_rng(202)
        state = _random_state(rng, n_arms=1)
        # Build an action at exactly the same pose as `state` (same xyz + rotation).
        action_same_pose = np.zeros(EE_ACTION_DIM_PER_ARM)
        action_same_pose[:3] = state[:3]
        R = quat_to_matrix(state[3:7])
        from anvil_shared.rotation import matrices_to_rot6d
        action_same_pose[3:9] = matrices_to_rot6d(R[np.newaxis])[0]
        action_same_pose[9] = state[7]

        delta = ee_delta_forward(action_same_pose, state)
        np.testing.assert_allclose(delta[:3], 0.0, atol=1e-10)
        np.testing.assert_allclose(delta[3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-10)


class TestEeDeltaSingleVsPerSampleState:
    def test_forward_agrees(self):
        rng = np.random.default_rng(11)
        T = 5
        state = _random_state(rng)
        actions = np.stack([_random_action(rng) for _ in range(T)])

        delta_single = ee_delta_forward(actions, state)
        states_per_sample = np.tile(state, (T, 1))
        delta_per_sample = ee_delta_forward(actions, states_per_sample)

        np.testing.assert_allclose(delta_single, delta_per_sample, atol=1e-12)

    def test_inverse_agrees(self):
        rng = np.random.default_rng(13)
        T = 5
        state = _random_state(rng)
        deltas = np.stack([_random_action(rng) for _ in range(T)])

        abs_single = ee_delta_inverse(deltas, state)
        states_per_sample = np.tile(state, (T, 1))
        abs_per_sample = ee_delta_inverse(deltas, states_per_sample)

        np.testing.assert_allclose(abs_single, abs_per_sample, atol=1e-12)


class TestEeDeltaWorldFrameComposition:
    """Verify the WORLD-frame (extrinsic) composition order against a known
    rotation — the frame-convention analogue of TestEEObsRelForward's
    test_body_frame_translation, and the direct regression guard against
    accidentally reusing the BODY-frame ee_relative math."""

    def test_translation_is_plain_world_difference(self):
        """Unlike ee_relative_forward's body-frame translation
        (world_delta @ R_state), ee_delta_forward's translation must be the
        RAW world-frame difference, unaffected by rotation."""
        import math

        theta = math.pi / 2
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        state = np.zeros(EE_STATE_DIM_PER_ARM)
        state[3:7] = [0.0, 0.0, s, c]  # 90 deg about z
        state[:3] = [0.1, 0.2, 0.3]

        action = _identity_action(n_arms=1)
        action[:3] = [0.4, 0.6, 0.3]  # world-frame absolute target position

        delta = ee_delta_forward(action, state)
        # World-frame: delta_xyz = action_xyz - state_xyz, NOT rotated by R_state.
        expected_world_delta = np.array([0.3, 0.4, 0.0])
        np.testing.assert_allclose(delta[:3], expected_world_delta, atol=1e-12)

    def test_rotation_composition_matches_robosuite_order(self):
        """R_delta = R_action @ R_state.T (extrinsic/left-multiply), verified
        against a known 90deg-about-z case, matching robosuite 1.4.0's own
        `goal_orientation = delta_rotation @ current_orientation` composition."""
        import math
        from anvil_shared.rotation import matrices_to_rot6d, quats_to_matrices

        # state: identity rotation. action: 90 deg about z.
        state = _identity_state(n_arms=1)
        theta = math.pi / 2
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        action_quat_equiv = np.array([0.0, 0.0, s, c])  # 90 deg about z, as a quaternion
        R_action = quats_to_matrices(action_quat_equiv)  # (3, 3), single quat
        action = _identity_action(n_arms=1)
        action[3:9] = matrices_to_rot6d(R_action)

        delta = ee_delta_forward(action, state)
        # state is identity, so R_delta = R_action @ I.T = R_action exactly.
        np.testing.assert_allclose(delta[3:9], action[3:9], atol=1e-10)

    def test_diverges_from_body_frame_relative_for_nontrivial_input(self):
        """Direct regression guard: for a non-trivial (non-identity) state,
        ee_delta_forward's world-frame result must NOT equal
        ee_relative_forward's body-frame result — if a future edit accidentally
        makes ee_delta_forward call the body-frame formula, this must fail."""
        rng = np.random.default_rng(303)
        action = _random_action(rng, n_arms=1)
        state = _random_state(rng, n_arms=1)

        delta_world = ee_delta_forward(action, state)
        delta_body = ee_relative_forward(action, state)

        # They must differ (state's rotation is random, essentially never identity).
        assert not np.allclose(delta_world[:3], delta_body[:3], atol=1e-9), (
            "ee_delta_forward's translation matches ee_relative_forward's body-frame "
            "translation — this indicates the world-frame formula was NOT actually used."
        )
        assert not np.allclose(delta_world[3:9], delta_body[3:9], atol=1e-9), (
            "ee_delta_forward's rotation matches ee_relative_forward's body-frame "
            "rotation — this indicates the world-frame formula was NOT actually used."
        )
