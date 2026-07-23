"""Tests for the patched_lerobot context manager + patch restoration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

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
        assert mod.foo == 2  # second patch was skipped
        assert len(runner._saved_originals) == 1
        runner.restore_all_patches()
        assert mod.foo == 1  # restored to the true original

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


class TestConfigSequencePatch:
    def test_ordered_lists_reach_lerobot_cli_and_patch_restores(self):
        import lerobot.configs.parser as parser

        original = parser._flatten_to_cli_args
        runner = TransformRunner(TrainingConfig())
        runner.apply_config_sequence_patch()
        try:
            args = parser._flatten_to_cli_args(
                {
                    "path": "ignored/checkpoint",
                    "action_feature_names": [
                        "right_joint_2.pos",
                        "right_joint_1.pos",
                        "left_joint_1.pos",
                    ],
                    "relative_exclude_joints": ["gripper"],
                    "optimizer_betas": [0.9, 0.95],
                }
            )
            assert args == [
                '--action_feature_names=["right_joint_2.pos","right_joint_1.pos","left_joint_1.pos"]',
                '--relative_exclude_joints=["gripper"]',
                "--optimizer_betas=[0.9,0.95]",
            ]
        finally:
            runner.restore_all_patches()

        assert parser._flatten_to_cli_args is original

    def test_pretrained_yaml_lists_survive_path_extraction(self, tmp_path):
        import lerobot.configs.parser as parser

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(
            """
policy:
  path: example/checkpoint
  action_feature_names: [right_joint_2.pos, right_joint_1.pos, left_joint_1.pos]
  relative_exclude_joints: [gripper]
  optimizer_betas: [0.9, 0.95]
""".lstrip()
        )
        runner = TransformRunner(TrainingConfig())
        runner.apply_config_sequence_patch()
        cleaned_path = None
        parser._config_path_args.clear()
        parser._config_yaml_overrides.clear()
        try:
            cleaned_path = Path(parser.extract_path_fields_from_config(str(recipe), ["policy"]))
            assert parser.get_path_arg("policy") == "example/checkpoint"
            assert parser.get_yaml_overrides("policy") == [
                '--action_feature_names=["right_joint_2.pos","right_joint_1.pos","left_joint_1.pos"]',
                '--relative_exclude_joints=["gripper"]',
                "--optimizer_betas=[0.9,0.95]",
            ]
        finally:
            runner.restore_all_patches()
            parser._config_path_args.clear()
            parser._config_yaml_overrides.clear()
            if cleaned_path is not None:
                cleaned_path.unlink(missing_ok=True)


class TestVLAJEPAInputPatch:
    def test_stacked_state_is_replaced_with_current_timestep(self, monkeypatch):
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        def upstream_prepare(_policy, batch, training=True):
            del training
            return {"state": batch["observation.state"][:, -1:, :].float()}

        monkeypatch.setattr(VLAJEPAPolicy, "_prepare_model_inputs", upstream_prepare)
        runner = TransformRunner(TrainingConfig())
        runner.apply_vla_jepa_input_patch()
        state = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)

        result = VLAJEPAPolicy._prepare_model_inputs(
            object(), {"observation.state": state}, training=True
        )

        assert torch.equal(result["state"][:, 0], state[:, 0])
        assert not torch.equal(result["state"][:, 0], state[:, -1])
        runner.restore_all_patches()
        restored = VLAJEPAPolicy._prepare_model_inputs(
            object(), {"observation.state": state}, training=True
        )
        assert torch.equal(restored["state"][:, 0], state[:, -1])


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


# =============================================================================
# apply_processor_compat_aliases / unregister on restore
# =============================================================================


class TestProcessorCompatAliases:
    """Tests for relative_actions_processor / delta_actions_processor aliases."""

    def _make_runner(self) -> TransformRunner:
        return TransformRunner(TrainingConfig())

    @staticmethod
    def _canonical_and_alias():
        from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

        canonical = RelativeActionsProcessorStep._registry_name
        alias = (
            "delta_actions_processor"
            if canonical == "relative_actions_processor"
            else "relative_actions_processor"
        )
        return canonical, alias, RelativeActionsProcessorStep

    def test_alias_registered_after_apply(self):
        """The non-canonical processor name should resolve after alias setup."""
        from lerobot.processor.pipeline import ProcessorStepRegistry

        canonical, alias, step_cls = self._canonical_and_alias()
        runner = self._make_runner()
        ProcessorStepRegistry.unregister(alias)

        runner.apply_processor_compat_aliases()
        try:
            assert canonical in ProcessorStepRegistry.list()
            assert ProcessorStepRegistry.get(alias) is step_cls
        finally:
            runner.restore_all_patches()

    def test_canonical_registry_name_preserved(self):
        """_registry_name must stay on the installed release's canonical name."""
        from lerobot.processor.pipeline import ProcessorStepRegistry

        canonical, alias, step_cls = self._canonical_and_alias()
        runner = self._make_runner()
        ProcessorStepRegistry.unregister(alias)

        runner.apply_processor_compat_aliases()
        try:
            assert step_cls._registry_name == canonical
        finally:
            runner.restore_all_patches()

    def test_alias_unregistered_after_restore(self):
        """restore_all_patches must remove only the compat alias."""
        from lerobot.processor.pipeline import ProcessorStepRegistry

        canonical, alias, _ = self._canonical_and_alias()
        runner = self._make_runner()
        ProcessorStepRegistry.unregister(alias)

        runner.apply_processor_compat_aliases()
        runner.restore_all_patches()

        assert alias not in ProcessorStepRegistry.list()
        assert canonical in ProcessorStepRegistry.list()

    def test_noop_when_already_registered(self):
        """Calling apply_processor_compat_aliases twice must not raise ValueError."""
        from lerobot.processor.pipeline import ProcessorStepRegistry

        _, alias, _ = self._canonical_and_alias()
        runner = self._make_runner()
        ProcessorStepRegistry.unregister(alias)

        runner.apply_processor_compat_aliases()
        try:
            # Second call — alias already present, must be a no-op.
            runner.apply_processor_compat_aliases()
            assert len(runner._registered_aliases) == 1  # registered only once
        finally:
            runner.restore_all_patches()
