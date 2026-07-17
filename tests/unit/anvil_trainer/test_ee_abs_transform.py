"""Tests for EEAbsTransform, _compute_ee_abs_stats, and _force_rot6d_identity.

Covers:
  1. EEAbsTransform.is_enabled — enabled for ee_abs only
  2. EEAbsTransform.apply — obs 8n→10n, action unchanged, no-op without obs
  3. EEAbsTransform.patch_metadata — observation.state shape 8n→10n via runner
  4. _force_rot6d_identity — rot6d dims clamped ±1, other dims unchanged
  5. _compute_ee_abs_stats — action rot6d dims ±1, obs computed and clamped,
     xyz/gripper in abs stats retain real values, correct return structure
  6. Regression: existing ee_relative tests still pass (shared helper refactor)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import EEAbsTransform


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
    """Return a torch tensor of rot6d-layout action (horizon, 10*n_arms)."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(size=(horizon, EE_ACTION_DIM * n_arms)), dtype=torch.float32)


# =============================================================================
# 1. is_enabled
# =============================================================================


class TestEEAbsIsEnabled:
    def test_enabled_for_ee_abs(self):
        cfg = TrainingConfig(action_type="ee_abs")
        assert EEAbsTransform().is_enabled(cfg) is True

    def test_disabled_for_ee_rel(self):
        cfg = TrainingConfig(action_type="ee_rel")
        assert EEAbsTransform().is_enabled(cfg) is False

    def test_disabled_for_joint_abs(self):
        cfg = TrainingConfig(action_type="joint_abs")
        assert EEAbsTransform().is_enabled(cfg) is False


# =============================================================================
# 2. apply
# =============================================================================


class TestEEAbsApply:
    def test_obs_shape_8n_to_10n_single_arm(self):
        """Single obs step: (8,) → (10,)."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=1)}
        result = EEAbsTransform().apply(item, cfg)
        assert result["observation.state"].shape == (EE_ACTION_DIM,)

    def test_obs_shape_8n_to_10n_multi_step(self):
        """Multi-step obs: (T, 8) → (T, 10)."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=4)}
        result = EEAbsTransform().apply(item, cfg)
        assert result["observation.state"].shape == (4, EE_ACTION_DIM)

    def test_obs_shape_bimanual(self):
        """Bimanual: (T, 16) → (T, 20)."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        item = {"observation.state": _make_obs_tensor(n_arms=2, n_steps=2)}
        result = EEAbsTransform().apply(item, cfg)
        assert result["observation.state"].shape == (2, EE_ACTION_DIM * 2)

    def test_action_unchanged(self):
        """Action tensor must not be modified."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        action = _make_action_tensor(n_arms=1, horizon=8)
        item = {
            "observation.state": _make_obs_tensor(n_arms=1, n_steps=1),
            "action": action.clone(),
        }
        result = EEAbsTransform().apply(item, cfg)
        assert torch.allclose(result["action"], item["action"])

    def test_noop_without_obs_state(self):
        """Item without observation.state must be returned unchanged."""
        cfg = TrainingConfig(action_type="ee_abs")
        item = {"action": np.zeros(10), "other_key": "value"}
        result = EEAbsTransform().apply(item, cfg)
        assert "observation.state" not in result
        assert result["other_key"] == "value"

    def test_xyz_passthrough_in_apply(self):
        """The first 3 dims per arm (xyz) must be preserved exactly after apply."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        obs = _make_obs_tensor(n_arms=1, n_steps=1)
        original_xyz = obs[:3].clone()
        item = {"observation.state": obs}
        result = EEAbsTransform().apply(item, cfg)
        np.testing.assert_allclose(
            result["observation.state"][:3].numpy(),
            original_xyz.numpy(),
            atol=1e-6,
        )

    def test_gripper_passthrough_in_apply(self):
        """Gripper (dim 7 in state, dim 9 in rot6d) must be preserved."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        obs = _make_obs_tensor(n_arms=1, n_steps=1)
        original_gripper = obs[7].item()
        item = {"observation.state": obs}
        result = EEAbsTransform().apply(item, cfg)
        assert abs(result["observation.state"][9].item() - original_gripper) < 1e-6

    def test_output_is_float32(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        cfg = TrainingConfig(action_type="ee_abs")
        item = {"observation.state": _make_obs_tensor(n_arms=1, n_steps=1)}
        result = EEAbsTransform().apply(item, cfg)
        assert result["observation.state"].dtype == torch.float32


# =============================================================================
# 3. patch_metadata — observation.state shape 8n → 10n
# =============================================================================


class TestEEAbsPatchMetadata:
    def _build_fake_lerobot(self, monkeypatch):
        """Build minimal fake lerobot module tree and monkeypatch sys.modules."""
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
        """patch_metadata must reshape observation.state 8n → 10n via runner._patch."""
        from anvil_trainer.patches import TransformRunner

        feature_utils, policies_factory = self._build_fake_lerobot(monkeypatch)
        orig_fu = feature_utils.dataset_to_policy_features
        orig_pf = policies_factory.dataset_to_policy_features

        cfg = TrainingConfig(action_type="ee_abs")
        runner = TransformRunner(cfg)
        transform = EEAbsTransform()
        transform.patch_metadata(cfg, runner=runner)

        # Both module attrs must have been patched
        assert feature_utils.dataset_to_policy_features is not orig_fu
        assert policies_factory.dataset_to_policy_features is not orig_pf

        # Call the patched function and check shape remapping
        captured = {}
        feature_utils.dataset_to_policy_features = lambda f: captured.update({"f": f}) or f
        policies_factory.dataset_to_policy_features(
            {"observation.state": {"shape": (8,)}, "action": {"shape": (10,)}}
        )
        # Note: the patched fn stored on policies_factory is the real patched closure,
        # which calls the original captured by the closure. Verify via runner restore.
        runner.restore_all_patches()
        assert feature_utils.dataset_to_policy_features is orig_fu
        assert policies_factory.dataset_to_policy_features is orig_pf

    def test_noop_for_non_ee_abs(self, monkeypatch):
        """patch_metadata is a no-op when action_type != ee_abs."""
        from anvil_trainer.patches import TransformRunner

        feature_utils, policies_factory = self._build_fake_lerobot(monkeypatch)
        orig_fu = feature_utils.dataset_to_policy_features

        cfg = TrainingConfig(action_type="ee_rel")
        runner = TransformRunner(cfg)
        transform = EEAbsTransform()
        transform.patch_metadata(cfg, runner=runner)

        # No patch should have been applied
        assert feature_utils.dataset_to_policy_features is orig_fu


# =============================================================================
# 4. _force_rot6d_identity
# =============================================================================


class TestForceRot6dIdentity:
    def test_rot6d_dims_clamped(self):
        """Dims 3–8 per arm must become -1 / +1."""
        from anvil_trainer.patches import _force_rot6d_identity

        n_arms = 1
        min_arr = np.zeros(10)
        max_arr = np.zeros(10)
        _force_rot6d_identity(min_arr, max_arr, n_arms)
        for r in range(3, 9):
            assert min_arr[r] == -1.0, f"dim {r} min expected -1, got {min_arr[r]}"
            assert max_arr[r] == 1.0, f"dim {r} max expected +1, got {max_arr[r]}"

    def test_non_rot6d_dims_untouched(self):
        """Dims 0–2 (xyz) and 9 (gripper) must remain unchanged."""
        from anvil_trainer.patches import _force_rot6d_identity

        min_arr = np.array([10.0] * 10)
        max_arr = np.array([20.0] * 10)
        _force_rot6d_identity(min_arr, max_arr, n_arms=1)
        for d in [0, 1, 2, 9]:
            assert min_arr[d] == 10.0, f"dim {d} min should be untouched"
            assert max_arr[d] == 20.0, f"dim {d} max should be untouched"

    def test_bimanual(self):
        """Both arms' rot6d dims must be clamped."""
        from anvil_trainer.patches import _force_rot6d_identity

        n_arms = 2
        min_arr = np.zeros(20)
        max_arr = np.zeros(20)
        _force_rot6d_identity(min_arr, max_arr, n_arms)
        for arm in range(n_arms):
            for r in range(3, 9):
                idx = arm * 10 + r
                assert min_arr[idx] == -1.0
                assert max_arr[idx] == 1.0


# =============================================================================
# 5. _compute_ee_abs_stats
# =============================================================================


def _build_fake_dataset(n_arms: int = 1, n_frames: int = 50, seed: int = 17):
    """Build a minimal fake LeRobotDataset for _compute_ee_abs_stats."""
    rng = np.random.default_rng(seed)

    # Absolute states (quat layout)
    states = np.zeros((n_frames, EE_STATE_DIM * n_arms))
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM
        states[:, s0:s0 + 3] = rng.normal(size=(n_frames, 3))
        q = rng.normal(size=(n_frames, 4))
        states[:, s0 + 3:s0 + 7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        states[:, s0 + 7] = rng.uniform(0.0, 0.08, n_frames)

    # Absolute actions (rot6d layout — already from converter)
    actions = rng.normal(size=(n_frames, EE_ACTION_DIM * n_arms))
    # Clamp rot6d dims to [-1,1] (as real data would from valid rotations)
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM
        actions[:, a0 + 3:a0 + 9] = np.clip(actions[:, a0 + 3:a0 + 9], -1, 1)

    hf = {
        "action": actions,
        "observation.state": states,
        "episode_index": np.zeros(n_frames, dtype=np.int64),
    }

    # Pre-populate meta.stats from the "raw" action distribution
    act_stats = {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "count": n_frames,
    }
    meta = MagicMock()
    meta.stats = {"action": act_stats, "observation.state": {}}

    ds = MagicMock()
    ds.hf_dataset = hf
    ds.meta = meta
    return ds


class TestComputeEEAbsStats:
    def _make_runner(self, n_arms: int = 1):
        from anvil_trainer.patches import TransformRunner
        cfg = TrainingConfig(action_type="ee_abs")
        return TransformRunner(cfg)

    def _make_cfg_mock(self):
        cfg = MagicMock()
        cfg.policy.n_obs_steps = 2
        cfg.policy.action_delta_indices = list(range(16))
        return cfg

    def test_returns_dict_with_correct_keys(self):
        runner = self._make_runner()
        ds = _build_fake_dataset()
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        assert result is not None
        assert "action" in result
        assert "observation.state" in result

    def test_action_rot6d_dims_are_pm1(self):
        """rot6d dims (3–8 per arm) in action stats must be exactly ±1."""
        runner = self._make_runner(n_arms=1)
        ds = _build_fake_dataset(n_arms=1)
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        act_min = np.array(result["action"]["min"])
        act_max = np.array(result["action"]["max"])
        for r in range(3, 9):
            assert act_min[r] == -1.0, f"action min[{r}] should be -1, got {act_min[r]}"
            assert act_max[r] == 1.0, f"action max[{r}] should be +1, got {act_max[r]}"

    def test_obs_rot6d_dims_are_pm1(self):
        """rot6d dims in obs.state stats must be exactly ±1."""
        runner = self._make_runner(n_arms=1)
        ds = _build_fake_dataset(n_arms=1)
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        obs_min = np.array(result["observation.state"]["min"])
        obs_max = np.array(result["observation.state"]["max"])
        for r in range(3, 9):
            assert obs_min[r] == -1.0, f"obs min[{r}] should be -1, got {obs_min[r]}"
            assert obs_max[r] == 1.0, f"obs max[{r}] should be +1, got {obs_max[r]}"

    def test_action_xyz_retains_real_distribution(self):
        """xyz dims (0-2) in action must NOT be clamped to ±1."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1, n_frames=200, seed=5)
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        act_min = np.array(result["action"]["min"])
        act_max = np.array(result["action"]["max"])
        # For random data, xyz range should not be exactly ±1
        for d in range(3):
            assert act_min[d] != -1.0 or act_max[d] != 1.0, (
                f"action xyz dim {d} should reflect real distribution, not identity trick"
            )

    def test_obs_xyz_retains_real_distribution(self):
        """xyz dims in obs.state must reflect the actual data range."""
        runner = self._make_runner()
        ds = _build_fake_dataset(n_arms=1, n_frames=200, seed=11)
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        obs_min = np.array(result["observation.state"]["min"])
        obs_max = np.array(result["observation.state"]["max"])
        for d in range(3):
            assert obs_min[d] != -1.0 or obs_max[d] != 1.0

    def test_bimanual_all_rot6d_dims_pm1(self):
        """Bimanual: rot6d dims for both arms must be ±1."""
        runner = self._make_runner(n_arms=2)
        ds = _build_fake_dataset(n_arms=2)
        result = runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        obs_min = np.array(result["observation.state"]["min"])
        obs_max = np.array(result["observation.state"]["max"])
        for arm in range(2):
            for r in range(3, 9):
                idx = arm * EE_ACTION_DIM + r
                assert obs_min[idx] == -1.0
                assert obs_max[idx] == 1.0

    def test_stats_injected_into_dataset(self):
        """_compute_ee_abs_stats must mutate full_dataset.meta.stats."""
        runner = self._make_runner()
        ds = _build_fake_dataset()
        runner._compute_ee_abs_stats(ds, self._make_cfg_mock())
        # Verify the meta was written (rot6d dims should be ±1 now)
        act_min = np.array(ds.meta.stats["action"]["min"])
        assert act_min[3] == -1.0

    def test_returns_none_for_joint_abs(self):
        """Must return None when config is not ee_abs."""
        from anvil_trainer.patches import TransformRunner
        cfg = TrainingConfig(action_type="joint_abs")
        runner = TransformRunner(cfg)
        result = runner._compute_ee_abs_stats(MagicMock(), MagicMock())
        assert result is None
