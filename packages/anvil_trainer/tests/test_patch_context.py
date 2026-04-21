"""Tests for the patched_lerobot context manager + patch restoration."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner


# =============================================================================
# _patch / restore_all_patches
# =============================================================================


class TestPatchInfrastructure:
    def _make_runner(self) -> TransformRunner:
        return TransformRunner(TrainingConfig())

    def test_patch_saves_original_and_sets_new(self):
        runner = self._make_runner()
        mod = types.SimpleNamespace(foo=1)
        runner._patch(mod, "foo", 2)
        assert mod.foo == 2
        assert len(runner._saved_originals) == 1
        assert runner._saved_originals[0] == (mod, "foo", 1)

    def test_restore_reverts_single_patch(self):
        runner = self._make_runner()
        mod = types.SimpleNamespace(foo=1)
        runner._patch(mod, "foo", 2)
        runner.restore_all_patches()
        assert mod.foo == 1
        assert runner._saved_originals == []

    def test_duplicate_patch_is_noop(self):
        runner = self._make_runner()
        mod = types.SimpleNamespace(foo=1)
        runner._patch(mod, "foo", 2)
        # Second call MUST NOT re-wrap / save the wrapped value as "original"
        runner._patch(mod, "foo", 3)
        assert mod.foo == 2            # second patch was skipped
        assert len(runner._saved_originals) == 1
        runner.restore_all_patches()
        assert mod.foo == 1            # restored to the true original

    def test_restore_is_lifo(self):
        runner = self._make_runner()
        mod = types.SimpleNamespace(a=1, b=10)
        runner._patch(mod, "a", 2)
        runner._patch(mod, "b", 20)
        runner.restore_all_patches()
        assert mod.a == 1 and mod.b == 10

    def test_restore_idempotent(self):
        """Calling restore twice is safe."""
        runner = self._make_runner()
        mod = types.SimpleNamespace(foo=1)
        runner._patch(mod, "foo", 2)
        runner.restore_all_patches()
        runner.restore_all_patches()  # must not raise
        assert mod.foo == 1

    def test_patches_across_multiple_modules(self):
        runner = self._make_runner()
        m1 = types.SimpleNamespace(x="orig1")
        m2 = types.SimpleNamespace(x="orig2")
        runner._patch(m1, "x", "new1")
        runner._patch(m2, "x", "new2")
        assert m1.x == "new1" and m2.x == "new2"
        runner.restore_all_patches()
        assert m1.x == "orig1" and m2.x == "orig2"


# =============================================================================
# Transform.patch_metadata interaction with runner
# =============================================================================


class TestTransformPatchMetadataUsesRunner:
    def test_exclude_observation_patches_via_runner(self, monkeypatch):
        """ExcludeObservationTransform.patch_metadata should go through runner._patch
        so its lerobot patches are reverted on context exit."""
        from anvil_trainer.transforms import ExcludeObservationTransform

        # Build nested fake lerobot modules so `import lerobot.datasets.feature_utils`
        # resolves and the `lerobot.datasets.feature_utils` attribute chain works.
        orig_fu = lambda f: f  # noqa: E731
        orig_pf = lambda f: f  # noqa: E731

        fake_lerobot = types.ModuleType("lerobot")
        fake_datasets = types.ModuleType("lerobot.datasets")
        fake_policies = types.ModuleType("lerobot.policies")
        fake_lerobot.datasets = fake_datasets
        fake_lerobot.policies = fake_policies

        feature_utils = types.ModuleType("lerobot.datasets.feature_utils")
        feature_utils.dataset_to_policy_features = orig_fu
        fake_datasets.feature_utils = feature_utils

        policies_factory = types.ModuleType("lerobot.policies.factory")
        policies_factory.dataset_to_policy_features = orig_pf
        fake_policies.factory = policies_factory

        monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
        monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
        monkeypatch.setitem(sys.modules, "lerobot.datasets.feature_utils", feature_utils)
        monkeypatch.setitem(sys.modules, "lerobot.policies", fake_policies)
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", policies_factory)

        cfg = TrainingConfig(exclude_observation=["images.chest"])
        runner = TransformRunner(cfg)
        transform = ExcludeObservationTransform()

        transform.patch_metadata(cfg, runner=runner)
        # Both module attrs replaced
        assert feature_utils.dataset_to_policy_features is not orig_fu
        assert policies_factory.dataset_to_policy_features is not orig_pf

        # Reverted when runner restores
        runner.restore_all_patches()
        assert feature_utils.dataset_to_policy_features is orig_fu
        assert policies_factory.dataset_to_policy_features is orig_pf
