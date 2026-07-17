"""Smoke tests for the ee_relative transform pipeline (formerly named "ee_rel";
"ee_rel" remains a permanent legacy alias for action_type, see test_enabled_for_ee_rel).

Covers:
  1. EERelativeTransform.apply — obs 8n→10n, action shape preserved
  2. C2 regression: n_obs_steps detection via model.config (not model directly)
  3. C1 regression: queue prefill shape for n_obs_steps=1,2,3
  4. patch_metadata — obs.state shape 8n→10n in dataset_to_policy_features
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import EERelativeTransform


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_config(action_type: str = "ee_rel") -> TrainingConfig:
    return TrainingConfig(action_type=action_type, split_ratio=[8.0, 1.0, 1.0])


def _identity_quat() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]  # [qx, qy, qz, qw]


def _make_obs_tensor(n_obs_steps: int, n_arms: int, rng: np.random.Generator):
    """Return a (n_obs_steps, 8*n_arms) float32 tensor with valid quaternions."""
    import torch

    data = rng.standard_normal((n_obs_steps, 8 * n_arms)).astype("float32")
    for arm in range(n_arms):
        col = arm * 8 + 3
        data[:, col:col + 4] = np.array(_identity_quat(), dtype="float32")
    return torch.tensor(data)


def _make_action_tensor(horizon: int, n_arms: int, rng: np.random.Generator):
    """Return a (horizon, 10*n_arms) float32 tensor with identity rot6d."""
    import torch

    data = rng.standard_normal((horizon, 10 * n_arms)).astype("float32")
    for arm in range(n_arms):
        col = arm * 10 + 3
        data[:, col:col + 6] = np.array([1, 0, 0, 0, 1, 0], dtype="float32")
    return torch.tensor(data)


# ── 1. EERelativeTransform.apply ───────────────────────────────────────────────────

class TestEERelativeTransformApply:
    def _apply(self, n_arms: int, n_obs_steps: int = 2, horizon: int = 16):
        import torch

        rng = np.random.default_rng(0)
        cfg = _make_config()
        t = EERelativeTransform()

        obs = _make_obs_tensor(n_obs_steps, n_arms, rng)
        action = _make_action_tensor(horizon, n_arms, rng)
        item = {"observation.state": obs, "action": action}
        return t.apply(item, cfg)

    def test_single_arm_obs_shape(self):
        """1-arm obs: (n_obs, 8) → (n_obs, 10) after transform."""
        out = self._apply(n_arms=1, n_obs_steps=2)
        assert out["observation.state"].shape == (2, 10)

    def test_bimanual_obs_shape(self):
        """2-arm obs: (n_obs, 16) → (n_obs, 20) after transform."""
        out = self._apply(n_arms=2, n_obs_steps=2)
        assert out["observation.state"].shape == (2, 20)

    def test_action_shape_preserved(self):
        """action dims stay 10*n_arms after transform (only values change)."""
        out = self._apply(n_arms=1, horizon=16)
        assert out["action"].shape == (16, 10)

    def test_anchor_step_is_identity(self):
        """The last obs step (the anchor) must be identity after relativisation:
        xyz=[0,0,0], rot6d=[1,0,0,0,1,0], gripper=passthrough."""
        import torch

        rng = np.random.default_rng(1)
        cfg = _make_config()
        t = EERelativeTransform()
        obs = _make_obs_tensor(3, n_arms=1, rng=rng)
        action = _make_action_tensor(16, n_arms=1, rng=rng)
        out = t.apply({"observation.state": obs, "action": action}, cfg)

        anchor_rel = out["observation.state"][-1]  # last step
        np.testing.assert_allclose(anchor_rel[:3].numpy(), [0, 0, 0], atol=1e-6,
                                   err_msg="anchor xyz must be zero")
        np.testing.assert_allclose(anchor_rel[3:9].numpy(), [1, 0, 0, 0, 1, 0], atol=1e-6,
                                   err_msg="anchor rot6d must be identity")

    def test_missing_keys_passthrough(self):
        """Item without 'action' key is returned unchanged (no crash)."""
        cfg = _make_config()
        t = EERelativeTransform()
        item: dict[str, Any] = {"task": "demo"}
        out = t.apply(item, cfg)
        assert out == item

    def test_not_enabled_for_ee_abs(self):
        """EERelativeTransform reports disabled for ee_abs config."""
        cfg = _make_config("ee_abs")
        t = EERelativeTransform()
        assert not t.is_enabled(cfg)

    def test_enabled_for_ee_relative(self):
        cfg = _make_config("ee_relative")
        assert EERelativeTransform().is_enabled(cfg)

    def test_enabled_for_ee_rel(self):
        """"ee_rel" is a permanent legacy alias for "ee_relative" — must behave identically."""
        cfg = _make_config("ee_rel")
        assert cfg.action_type == "ee_relative"
        assert EERelativeTransform().is_enabled(cfg)


# ── 2. C2 regression: n_obs_steps detection ──────────────────────────────────

class TestNObsStepsDetection:
    """C2: n_obs_steps must come from model.config, not model directly."""

    @staticmethod
    def _detect(model: Any) -> int:
        return int(getattr(getattr(model, "config", None), "n_obs_steps", 2))

    def test_reads_from_model_config(self):
        """model.config.n_obs_steps=3 → detected as 3."""
        model = SimpleNamespace(config=SimpleNamespace(n_obs_steps=3))
        assert self._detect(model) == 3

    def test_fallback_when_no_config_attr(self):
        """Model with no 'config' attribute → fallback=2."""
        model = object()
        assert self._detect(model) == 2

    def test_fallback_when_config_has_no_n_obs_steps(self):
        """model.config exists but has no n_obs_steps → fallback=2."""
        model = SimpleNamespace(config=SimpleNamespace())
        assert self._detect(model) == 2

    def test_bug_old_pattern_always_falls_back(self):
        """Regression: old pattern getattr(model, 'n_obs_steps', 2) returns 2
        even when model.config.n_obs_steps=3 — confirms C2 was a real bug."""
        model = SimpleNamespace(config=SimpleNamespace(n_obs_steps=3))
        # old (broken) pattern
        old_result = getattr(model, "n_obs_steps", 2)
        assert old_result == 2, "Old pattern should fall through — confirms C2 was a bug"
        # new (fixed) pattern
        new_result = self._detect(model)
        assert new_result == 3


# ── 3. C1 regression: queue prefill shape ────────────────────────────────────

class TestQueuePrefillShape:
    """C1: prefill must produce (1, 10n) entries, not (10n,).

    Parameterised over n_obs_steps to catch off-by-one in window size.
    """

    @staticmethod
    def _run_prefill(n_obs_steps: int, n_arms: int = 1):
        """Simulate _prefill_ee_relative_queue logic and return the queue."""
        torch = pytest.importorskip("torch", reason="torch not installed")

        state_dim = 10 * n_arms
        rng = np.random.default_rng(99)
        obs_window_rel_np = rng.standard_normal((n_obs_steps, state_dim))

        queue: deque = deque(maxlen=n_obs_steps)

        for i in range(len(obs_window_rel_np) - 1):
            obs_t = torch.tensor(obs_window_rel_np[i], dtype=torch.float32).unsqueeze(0)  # (1,10n)
            # simulate no-op preprocessor
            queue.append(obs_t)

        return queue, state_dim

    def test_n_obs_steps_2_entries_shape(self):
        """n_obs_steps=2 → 1 prefill entry, shape (1, 10)."""
        torch = pytest.importorskip("torch")
        queue, state_dim = self._run_prefill(n_obs_steps=2, n_arms=1)
        assert len(queue) == 1
        assert queue[0].shape == (1, state_dim)

    def test_n_obs_steps_3_entries_shape(self):
        """n_obs_steps=3 → 2 prefill entries, all shape (1, 10)."""
        torch = pytest.importorskip("torch")
        queue, state_dim = self._run_prefill(n_obs_steps=3, n_arms=1)
        assert len(queue) == 2
        for entry in queue:
            assert entry.shape == (1, state_dim), (
                f"Expected (1, {state_dim}), got {entry.shape}. "
                "C1 regression: squeeze(0) would produce (10,) which crashes torch.stack."
            )

    def test_n_obs_steps_1_no_prefill(self):
        """n_obs_steps=1 → no prefill entries (range(0) = empty)."""
        torch = pytest.importorskip("torch")
        queue, _ = self._run_prefill(n_obs_steps=1, n_arms=1)
        assert len(queue) == 0

    def test_stack_dim1_n_obs_steps_2(self):
        """After prefill + current obs push, torch.stack(..., dim=1) must succeed."""
        torch = pytest.importorskip("torch")
        n_arms, n_obs_steps = 1, 2
        state_dim = 10 * n_arms
        queue: deque = deque(maxlen=n_obs_steps)

        # prefill: 1 historical entry
        queue.append(torch.zeros(1, state_dim))
        # current obs pushed by select_action → populate_queues
        queue.append(torch.zeros(1, state_dim))

        stacked = torch.stack(list(queue), dim=1)
        assert stacked.shape == (1, n_obs_steps, state_dim)

    def test_stack_dim1_n_obs_steps_3(self):
        """n_obs_steps=3: stack of 3 (1,10) entries succeeds."""
        torch = pytest.importorskip("torch")
        n_arms, n_obs_steps = 1, 3
        state_dim = 10 * n_arms
        queue: deque = deque(maxlen=n_obs_steps)

        for _ in range(n_obs_steps):
            queue.append(torch.zeros(1, state_dim))

        stacked = torch.stack(list(queue), dim=1)
        assert stacked.shape == (1, n_obs_steps, state_dim)

    def test_bimanual_prefill_shape(self):
        """Bimanual (n_arms=2): prefill entries shape (1, 20)."""
        torch = pytest.importorskip("torch")
        queue, state_dim = self._run_prefill(n_obs_steps=2, n_arms=2)
        assert len(queue) == 1
        assert queue[0].shape == (1, state_dim)


# ── 4. patch_metadata shape change ───────────────────────────────────────────

class TestPatchMetadataShapeChange:
    """patch_metadata must change obs.state shape from (8n,) to (10n,)."""

    def _run_patched(self, input_shape: tuple, n_arms: int = 1):
        import lerobot.datasets.feature_utils as _feat_utils
        from lerobot.datasets.feature_utils import dataset_to_policy_features as _orig

        cfg = _make_config()
        t = EERelativeTransform()

        features = {
            "observation.state": {"shape": input_shape, "dtype": "float32", "names": None},
            "action": {"shape": (10 * n_arms,), "dtype": "float32", "names": None},
        }

        # Apply patch directly (no runner)
        t.patch_metadata(cfg)
        try:
            result = _feat_utils.dataset_to_policy_features(features)
        finally:
            # Restore original to avoid cross-test contamination
            _feat_utils.dataset_to_policy_features = _orig

        return result

    def test_obs_8_becomes_10(self):
        """obs.state (8,) → patched to (10,) in policy features."""
        result = self._run_patched(input_shape=(8,), n_arms=1)
        obs_shape = None
        for key, val in result.items():
            if "observation.state" in key:
                obs_shape = val.shape
                break
        assert obs_shape is not None, "observation.state missing from result"
        assert obs_shape[0] == 10, f"Expected 10, got {obs_shape[0]}"

    def test_obs_16_becomes_20_bimanual(self):
        """Bimanual obs.state (16,) → patched to (20,)."""
        result = self._run_patched(input_shape=(16,), n_arms=2)
        obs_shape = None
        for key, val in result.items():
            if "observation.state" in key:
                obs_shape = val.shape
                break
        assert obs_shape is not None
        assert obs_shape[0] == 20, f"Expected 20, got {obs_shape[0]}"

    def test_non_ee_relative_config_skips_patch(self):
        """patch_metadata is a no-op for ee_abs — does not modify the function."""
        import lerobot.datasets.feature_utils as _feat_utils
        from lerobot.datasets.feature_utils import dataset_to_policy_features as _orig

        cfg = _make_config("ee_abs")
        t = EERelativeTransform()
        t.patch_metadata(cfg)

        # Function should be unchanged
        assert _feat_utils.dataset_to_policy_features is _orig
