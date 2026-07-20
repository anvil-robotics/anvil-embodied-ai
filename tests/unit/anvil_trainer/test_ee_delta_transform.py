"""Tests for EEDeltaTransform and _compute_ee_delta_stats.

Covers:
  1. EEDeltaTransform.is_enabled — enabled for ee_delta only
  2. EEDeltaTransform.apply — obs 8n→10n, action COMPLETELY unchanged
     (no double-transform — action is already a baked delta), no-op without obs
  3. EEDeltaTransform.patch_metadata — observation.state shape 8n→10n via runner
  4. _compute_ee_delta_stats — reads stats straight off the static action column
     (no live replay), rot6d dims clamped ±1, xyz/gripper retain real values,
     epsilon-floor on std for near-constant dims, correct return structure
  5. Regression: existing ee_abs/ee_relative tests still pass (shared registry)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import EEDeltaTransform


# =============================================================================
# Helpers
# =============================================================================

EE_STATE_DIM = 8   # per arm
EE_ACTION_DIM = 10  # per arm


def _make_obs_tensor(n_arms: int = 1, n_steps: int = 1, seed: int = 42):
    """Return a torch tensor of quat-layout obs (n_steps, 8*n_arms) or (8*n_arms,)."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    rng = np.random.default_rng(seed)
    data = np.zeros((n_steps, EE_STATE_DIM * n_arms))
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM
        data[:, s0:s0 + 3] = rng.normal(size=(n_steps, 3))       # xyz
        q = rng.normal(size=(n_steps, 4))
        data[:, s0 + 3:s0 + 7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        data[:, s0 + 7] = rng.uniform(0.0, 0.08, size=n_steps)   # gripper
    if n_steps == 1:
        return torch.tensor(data[0], dtype=torch.float32)
    return torch.tensor(data, dtype=torch.float32)


def _make_action_tensor(n_arms: int = 1, horizon: int = 16, seed: int = 99):
    """Return a torch tensor of rot6d-layout (baked-delta) action (horizon, 10*n_arms)."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(size=(horizon, EE_ACTION_DIM * n_arms)), dtype=torch.float32)


# =============================================================================
# 1. is_enabled
# =============================================================================


class TestEEDeltaIsEnabled:
    def test_enabled_for_ee_delta(self):
        cfg = TrainingConfig(action_type="ee_delta")
        assert EEDeltaTransform().is_enabled(cfg) is True

    def test_disabled_for_ee_relative(self):
        cfg = TrainingConfig(action_type="ee_relative")
        assert EEDeltaTransform().is_enabled(cfg) is False

    def test_disabled_for_ee_abs(self):
        cfg = TrainingConfig(action_type="ee_abs")
        assert EEDeltaTransform().is_enabled(cfg) is False

    def test_disabled_for_joint_abs(self):
        cfg = TrainingConfig(action_type="joint_abs")
        assert EEDeltaTransform().is_enabled(cfg) is False


# =============================================================================
# 2. apply
# =============================================================================


class TestEEDeltaApply:
    def test_obs_shape_8n_to_10n_single_arm(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=1)}
        result = EEDeltaTransform().apply(item, cfg)
        assert result["observation.state"].shape == (EE_ACTION_DIM,)

    def test_obs_shape_8n_to_10n_multi_step(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=4)}
        result = EEDeltaTransform().apply(item, cfg)
        assert result["observation.state"].shape == (4, EE_ACTION_DIM)

    def test_obs_shape_bimanual(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"observation.state": _make_obs_tensor(n_arms=2, n_steps=2)}
        result = EEDeltaTransform().apply(item, cfg)
        assert result["observation.state"].shape == (2, EE_ACTION_DIM * 2)

    def test_action_completely_unchanged_no_double_transform(self):
        """Action must be BYTE-IDENTICAL after apply — it's already the baked
        delta written by mcap_converter; re-transforming it would silently
        double-apply the delta computation."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        action = _make_action_tensor(n_arms=1, horizon=8)
        item = {
            "observation.state": _make_obs_tensor(n_arms=1, n_steps=1),
            "action": action.clone(),
        }
        result = EEDeltaTransform().apply(item, cfg)
        assert torch.equal(result["action"], action), (
            "EEDeltaTransform must not modify action at all — it is already "
            "the baked Delta(n->n+1) target from mcap_converter."
        )

    def test_noop_without_obs_state(self):
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"action": np.zeros(10), "other_key": "value"}
        result = EEDeltaTransform().apply(item, cfg)
        assert "observation.state" not in result
        assert result["other_key"] == "value"

    def test_xyz_passthrough_in_apply(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        obs = _make_obs_tensor(n_arms=1, n_steps=1)
        original_xyz = obs[:3].clone()
        item = {"observation.state": obs}
        result = EEDeltaTransform().apply(item, cfg)
        np.testing.assert_allclose(
            result["observation.state"][:3].numpy(),
            original_xyz.numpy(),
            atol=1e-6,
        )

    def test_gripper_passthrough_in_apply(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        obs = _make_obs_tensor(n_arms=1, n_steps=1)
        original_gripper = obs[7].item()
        item = {"observation.state": obs}
        result = EEDeltaTransform().apply(item, cfg)
        assert abs(result["observation.state"][9].item() - original_gripper) < 1e-6

    def test_output_is_float32(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=1)}
        result = EEDeltaTransform().apply(item, cfg)
        assert result["observation.state"].dtype == torch.float32


# =============================================================================
# 3. patch_metadata — observation.state shape 8n → 10n
# =============================================================================


class TestEEDeltaPatchMetadata:
    def _build_fake_lerobot(self, monkeypatch):
        fake_lerobot = types.ModuleType("lerobot")
        fake_datasets = types.ModuleType("lerobot.datasets")
        fake_policies = types.ModuleType("lerobot.policies")
        fake_lerobot.datasets = fake_datasets
        fake_lerobot.policies = fake_policies

        orig_fn = lambda f: f  # noqa: E731

        feature_utils = types.ModuleType("lerobot.datasets.feature_utils")
        feature_utils.dataset_to_policy_features = orig_fn
        fake_datasets.feature_utils = feature_utils

        policies_factory = types.ModuleType("lerobot.policies.factory")
        policies_factory.dataset_to_policy_features = orig_fn
        fake_policies.factory = policies_factory

        monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
        monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
        monkeypatch.setitem(sys.modules, "lerobot.datasets.feature_utils", feature_utils)
        monkeypatch.setitem(sys.modules, "lerobot.policies", fake_policies)
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", policies_factory)
        return feature_utils, policies_factory

    def test_obs_state_shape_patched_via_runner(self, monkeypatch):
        from anvil_trainer.patches import TransformRunner

        feature_utils, policies_factory = self._build_fake_lerobot(monkeypatch)
        orig_fu = feature_utils.dataset_to_policy_features
        orig_pf = policies_factory.dataset_to_policy_features

        cfg = TrainingConfig(action_type="ee_delta")
        runner = TransformRunner(cfg)
        transform = EEDeltaTransform()
        transform.patch_metadata(cfg, runner=runner)

        assert feature_utils.dataset_to_policy_features is not orig_fu
        assert policies_factory.dataset_to_policy_features is not orig_pf

        runner.restore_all_patches()
        assert feature_utils.dataset_to_policy_features is orig_fu
        assert policies_factory.dataset_to_policy_features is orig_pf

    def test_noop_for_non_ee_delta(self, monkeypatch):
        from anvil_trainer.patches import TransformRunner

        feature_utils, policies_factory = self._build_fake_lerobot(monkeypatch)
        orig_fu = feature_utils.dataset_to_policy_features

        cfg = TrainingConfig(action_type="ee_relative")
        runner = TransformRunner(cfg)
        transform = EEDeltaTransform()
        transform.patch_metadata(cfg, runner=runner)

        assert feature_utils.dataset_to_policy_features is orig_fu


# =============================================================================
# 4. _compute_ee_delta_stats
# =============================================================================


def _build_fake_dataset(
    n_arms: int = 1,
    n_frames: int = 50,
    seed: int = 17,
    near_constant_dim: int | None = None,
):
    """Build a minimal fake LeRobotDataset for _compute_ee_delta_stats.

    ``action`` here represents an ALREADY-BAKED per-frame delta (small
    magnitudes near zero for xyz, near-identity rot6d), unlike
    _build_fake_dataset in test_ee_abs_transform.py where action is absolute.
    ``near_constant_dim`` optionally forces one action dim to near-zero
    variance, simulating the realistic n=10-episode degenerate case flagged
    in claude_docs/ee-delta-flow-plan.md.
    """
    rng = np.random.default_rng(seed)

    states = np.zeros((n_frames, EE_STATE_DIM * n_arms))
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM
        states[:, s0:s0 + 3] = rng.normal(size=(n_frames, 3))
        q = rng.normal(size=(n_frames, 4))
        states[:, s0 + 3:s0 + 7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        states[:, s0 + 7] = rng.uniform(0.0, 0.08, n_frames)

    # Baked per-frame delta: small xyz deltas, near-identity rot6d, small gripper deltas.
    actions = np.zeros((n_frames, EE_ACTION_DIM * n_arms))
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM
        actions[:, a0:a0 + 3] = rng.normal(scale=0.005, size=(n_frames, 3))  # small xyz delta
        actions[:, a0 + 3] = 1.0  # near-identity rot6d, perturbed slightly below
        actions[:, a0 + 7] = 1.0
        actions[:, a0 + 3:a0 + 9] += rng.normal(scale=0.01, size=(n_frames, 6))
        actions[:, a0 + 9] = rng.normal(scale=0.001, size=n_frames)  # small gripper delta

    if near_constant_dim is not None:
        actions[:, near_constant_dim] = 0.001 + rng.normal(scale=1e-9, size=n_frames)

    hf = {
        "action": actions,
        "observation.state": states,
        "episode_index": np.zeros(n_frames, dtype=np.int64),
    }

    meta = MagicMock()
    meta.stats = {"action": {}, "observation.state": {}}

    ds = MagicMock()
    ds.hf_dataset = hf
    ds.meta = meta
    return ds


class TestComputeEEDeltaStats:
    def _make_runner(self):
        from anvil_trainer.patches import TransformRunner
        cfg = TrainingConfig(action_type="ee_delta")
        return TransformRunner(cfg)

    def _make_cfg_mock(self):
        cfg = MagicMock()
        cfg.policy.n_obs_steps = 2
        cfg.policy.action_delta_indices = list(range(16))
        return cfg

    def test_returns_dict_with_correct_keys(self):
        runner = self._make_runner()
        ds = _build_fake_dataset()
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        assert result is not None
        assert "action" in result
        assert "observation.state" in result

    def test_action_rot6d_dims_are_pm1(self):
        """rot6d dims (3-8 per arm) in action stats must be exactly ±1
        (identity trick) even though the baked delta's raw rot6d values are
        near-identity, not exactly ±1."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        act_min = np.array(result["action"]["min"])
        act_max = np.array(result["action"]["max"])
        for r in range(3, 9):
            assert act_min[r] == -1.0, f"action min[{r}] should be -1, got {act_min[r]}"
            assert act_max[r] == 1.0, f"action max[{r}] should be +1, got {act_max[r]}"

    def test_obs_rot6d_dims_are_pm1(self):
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        obs_min = np.array(result["observation.state"]["min"])
        obs_max = np.array(result["observation.state"]["max"])
        for r in range(3, 9):
            assert obs_min[r] == -1.0
            assert obs_max[r] == 1.0

    def test_action_xyz_reflects_real_small_delta_distribution(self):
        """Unlike _compute_ee_abs_stats (absolute pose range), the action xyz
        stats here must reflect the SMALL per-frame delta scale — not clamped,
        and not accidentally huge (would indicate the wrong column was read)."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1, n_frames=200, seed=5)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        act_std = np.array(result["action"]["std"])
        for d in range(3):
            assert 0 < act_std[d] < 0.1, (
                f"action xyz delta dim {d} std={act_std[d]} is not in the expected "
                "small per-frame-delta range"
            )

    def test_epsilon_floor_on_near_constant_dim(self):
        """A near-constant action dim (realistic at n=10 episodes) must be
        floored at 1e-6 std, not zero — replicating the existing
        _compute_ee_abs_stats / _compute_ee_relative_stats epsilon-floor
        pattern verbatim, per the plan's explicit requirement."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1, n_frames=50, near_constant_dim=0)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        act_std = np.array(result["action"]["std"])
        assert act_std[0] == pytest.approx(1e-6, abs=1e-12), (
            f"near-constant dim 0 std should be floored at 1e-6, got {act_std[0]}"
        )
        assert act_std[0] > 0.0, "std must never be exactly zero (divide-by-zero risk)"

    def test_stats_injected_into_dataset(self):
        runner = self._make_runner()
        ds = _build_fake_dataset()
        runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        act_min = np.array(ds.meta.stats["action"]["min"])
        assert act_min[3] == -1.0

    def test_returns_none_for_non_ee_delta(self):
        from anvil_trainer.patches import TransformRunner
        cfg = TrainingConfig(action_type="ee_abs")
        runner = TransformRunner(cfg)
        result = runner._compute_ee_delta_stats(MagicMock(), MagicMock())
        assert result is None

    def test_bimanual_all_rot6d_dims_pm1(self):
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=2)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())
        obs_min = np.array(result["observation.state"]["min"])
        obs_max = np.array(result["observation.state"]["max"])
        for arm in range(2):
            for r in range(3, 9):
                idx = arm * EE_ACTION_DIM + r
                assert obs_min[idx] == -1.0
                assert obs_max[idx] == 1.0

    def test_does_not_replay_live_transform(self):
        """Regression guard: unlike _compute_ee_relative_stats, this method must
        read `action` straight off the dataset without calling any
        ee_relative_forward/ee_delta_forward replay — the baked column IS the
        target already. Verify by confirming the returned action mean/std
        matches a plain numpy computation over the raw column exactly."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1, n_frames=64, seed=3)
        result = runner._compute_ee_delta_stats(ds, self._make_cfg_mock())

        raw_actions = ds.hf_dataset["action"]
        expected_mean = raw_actions.mean(axis=0)
        np.testing.assert_allclose(result["action"]["mean"], expected_mean, atol=1e-10)
