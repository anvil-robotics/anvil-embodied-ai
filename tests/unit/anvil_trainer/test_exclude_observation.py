"""
Tests for ExcludeObservationTransform and --exclude-observation CLI parsing.

Covers:
  1. ExcludeObservationTransform._full_keys() suffix expansion
  2. ExcludeObservationTransform.apply() removes correct keys
  3. ExcludeObservationTransform.apply() is a no-op when exclude_observation is None
  4. ExcludeObservationTransform.is_enabled() reflects config
  5. patch_metadata() wraps dataset_to_policy_features to exclude keys
  6. warn_unknown_exclude_keys() warns on unknown keys, silent on valid keys
  7. CLI parsing: --exclude-observation=images.chest,velocity sets config correctly
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anvil_trainer.train import ExcludeObservationTransform, TrainingConfig


# =============================================================================
# 1. _full_keys() suffix expansion
# =============================================================================

class TestFullKeys:
    def test_image_suffix_expansion(self):
        cfg = TrainingConfig(exclude_observation=["images.chest", "images.wrist_l"])
        result = ExcludeObservationTransform._full_keys(cfg)
        assert result == {"observation.images.chest", "observation.images.wrist_l"}

    def test_non_image_suffix_expansion(self):
        cfg = TrainingConfig(exclude_observation=["velocity", "effort"])
        result = ExcludeObservationTransform._full_keys(cfg)
        assert result == {"observation.velocity", "observation.effort"}

    def test_mixed_suffix_expansion(self):
        cfg = TrainingConfig(exclude_observation=["images.chest", "velocity", "effort"])
        result = ExcludeObservationTransform._full_keys(cfg)
        assert result == {
            "observation.images.chest",
            "observation.velocity",
            "observation.effort",
        }

    def test_single_suffix(self):
        cfg = TrainingConfig(exclude_observation=["velocity"])
        result = ExcludeObservationTransform._full_keys(cfg)
        assert result == {"observation.velocity"}


# =============================================================================
# 2. apply() removes correct keys
# =============================================================================

class TestApply:
    def _make_item(self):
        return {
            "observation.images.chest": "chest_tensor",
            "observation.images.waist": "waist_tensor",
            "observation.images.wrist_r": "wrist_r_tensor",
            "observation.state": [0.1, 0.2],
            "observation.velocity": [0.01, 0.02],
            "observation.effort": [1.0, 2.0],
            "action": [0.3, 0.4],
        }

    def test_removes_image_keys(self):
        cfg = TrainingConfig(exclude_observation=["images.chest", "images.wrist_r"])
        item = self._make_item()
        result = ExcludeObservationTransform().apply(item, cfg)
        assert "observation.images.chest" not in result
        assert "observation.images.wrist_r" not in result
        # other keys untouched
        assert "observation.images.waist" in result
        assert "observation.state" in result
        assert "action" in result

    def test_removes_non_image_keys(self):
        cfg = TrainingConfig(exclude_observation=["velocity", "effort"])
        item = self._make_item()
        result = ExcludeObservationTransform().apply(item, cfg)
        assert "observation.velocity" not in result
        assert "observation.effort" not in result
        assert "observation.state" in result
        assert "action" in result

    def test_removes_mixed_keys(self):
        cfg = TrainingConfig(exclude_observation=["images.chest", "velocity"])
        item = self._make_item()
        result = ExcludeObservationTransform().apply(item, cfg)
        assert "observation.images.chest" not in result
        assert "observation.velocity" not in result
        assert "observation.images.waist" in result
        assert "observation.effort" in result

    def test_nonexistent_key_is_silent(self):
        """pop() on non-existent key should not raise."""
        cfg = TrainingConfig(exclude_observation=["nonexistent_key"])
        item = self._make_item()
        result = ExcludeObservationTransform().apply(item, cfg)
        # All original keys should still be present
        assert "observation.state" in result
        assert "action" in result

    def test_returns_dict(self):
        cfg = TrainingConfig(exclude_observation=["velocity"])
        item = self._make_item()
        result = ExcludeObservationTransform().apply(item, cfg)
        assert isinstance(result, dict)


# =============================================================================
# 3. apply() is a no-op when exclude_observation is None
# =============================================================================

class TestApplyNoOp:
    def test_none_config_is_disabled(self):
        """When exclude_observation=None, is_enabled() is False so apply() is never called."""
        cfg = TrainingConfig(exclude_observation=None)
        transform = ExcludeObservationTransform()
        # is_enabled() must be False — TransformRunner never calls apply() in this state
        assert transform.is_enabled(cfg) is False

    def test_empty_list_is_disabled(self):
        """When exclude_observation=[], is_enabled() is False so apply() is never called."""
        cfg = TrainingConfig(exclude_observation=[])
        transform = ExcludeObservationTransform()
        assert transform.is_enabled(cfg) is False


# =============================================================================
# 4. is_enabled() reflects config
# =============================================================================

class TestIsEnabled:
    def test_enabled_with_list(self):
        cfg = TrainingConfig(exclude_observation=["velocity"])
        assert ExcludeObservationTransform().is_enabled(cfg) is True

    def test_disabled_with_none(self):
        cfg = TrainingConfig(exclude_observation=None)
        assert ExcludeObservationTransform().is_enabled(cfg) is False

    def test_disabled_with_empty_list(self):
        cfg = TrainingConfig(exclude_observation=[])
        assert ExcludeObservationTransform().is_enabled(cfg) is False


# =============================================================================
# 5. patch_metadata() wraps dataset_to_policy_features
# =============================================================================

class TestPatchMetadata:
    """
    patch_metadata() patches lerobot.datasets.feature_utils.dataset_to_policy_features
    AND lerobot.policies.factory.dataset_to_policy_features (where it is actually called).
    """

    def _run_patch_test(self, exclude_observation, features):
        """Helper: install patch, call via factory module, return what the original received."""
        import lerobot.datasets.feature_utils as feature_utils_mod
        import lerobot.policies.factory as factory_mod

        original_feature_utils = feature_utils_mod.dataset_to_policy_features
        original_factory = factory_mod.dataset_to_policy_features
        captured = {}

        def mock_original(feats: dict) -> dict:
            captured["received"] = dict(feats)
            return feats

        # Set the mock as the "original" that will be captured by the closure
        feature_utils_mod.dataset_to_policy_features = mock_original
        factory_mod.dataset_to_policy_features = mock_original

        try:
            cfg = TrainingConfig(exclude_observation=exclude_observation)
            transform = ExcludeObservationTransform()
            transform.patch_metadata(cfg)

            # Call via the factory module (where lerobot actually calls it)
            factory_mod.dataset_to_policy_features(features)
            return captured.get("received", {})
        finally:
            feature_utils_mod.dataset_to_policy_features = original_feature_utils
            factory_mod.dataset_to_policy_features = original_factory

    def test_excluded_keys_filtered_from_features(self):
        """patch_metadata() should exclude specified keys before calling original."""
        features = {
            "observation.state": "state_spec",
            "observation.velocity": "vel_spec",
            "observation.effort": "effort_spec",
            "action": "action_spec",
        }
        received = self._run_patch_test(["velocity", "effort"], features)

        assert "observation.velocity" not in received
        assert "observation.effort" not in received
        assert "observation.state" in received
        assert "action" in received

    def test_image_keys_filtered_from_features(self):
        """patch_metadata() should filter image observation keys."""
        features = {
            "observation.images.chest": "chest_spec",
            "observation.images.wrist_l": "wrist_l_spec",
            "observation.images.waist": "waist_spec",
            "action": "action_spec",
        }
        received = self._run_patch_test(["images.chest", "images.wrist_l"], features)

        assert "observation.images.chest" not in received
        assert "observation.images.wrist_l" not in received
        assert "observation.images.waist" in received


# =============================================================================
# 6. warn_unknown_exclude_keys()
# =============================================================================

class TestWarnUnknownExcludeKeys:
    def _write_info_json(self, tmp_dir: Path, features: list[str]) -> str:
        """Write a minimal meta/info.json and return dataset root path."""
        meta_dir = tmp_dir / "meta"
        meta_dir.mkdir()
        info = {"features": {k: {} for k in features}}
        (meta_dir / "info.json").write_text(json.dumps(info))
        return str(tmp_dir)

    def test_warns_on_unknown_key(self, caplog):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_info_json(
                Path(tmp),
                features=["observation.state", "action"],
            )
            cfg = TrainingConfig(
                exclude_observation=["velocity"],  # not in dataset
                dataset_root=root,
            )
            import logging
            with caplog.at_level(logging.WARNING, logger="anvil_trainer.train"):
                cfg.warn_unknown_exclude_keys()

            assert any("observation.velocity" in r.message for r in caplog.records)

    def test_silent_on_valid_key(self, caplog):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_info_json(
                Path(tmp),
                features=["observation.state", "observation.velocity", "action"],
            )
            cfg = TrainingConfig(
                exclude_observation=["velocity"],  # IS in dataset
                dataset_root=root,
            )
            import logging
            with caplog.at_level(logging.WARNING, logger="anvil_trainer.train"):
                cfg.warn_unknown_exclude_keys()

            # No warning about velocity
            assert not any("observation.velocity" in r.message for r in caplog.records)

    def test_skips_when_no_exclude(self):
        """warn_unknown_exclude_keys() returns early when exclude_observation is None."""
        cfg = TrainingConfig(exclude_observation=None, dataset_root="/does/not/exist")
        # Should not raise even if dataset_root is invalid
        cfg.warn_unknown_exclude_keys()

    def test_skips_when_no_dataset_root(self):
        """warn_unknown_exclude_keys() returns early when dataset_root is None."""
        cfg = TrainingConfig(exclude_observation=["velocity"], dataset_root=None)
        cfg.warn_unknown_exclude_keys()

    def test_warns_when_info_not_found(self, caplog):
        """warn_unknown_exclude_keys() warns when info.json is missing."""
        cfg = TrainingConfig(
            exclude_observation=["velocity"],
            dataset_root="/nonexistent/path",
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="anvil_trainer.train"):
            cfg.warn_unknown_exclude_keys()

        assert any("Cannot validate" in r.message for r in caplog.records)


# =============================================================================
# 7. CLI parsing of --exclude-observation
# =============================================================================

class TestCLIParsing:
    def _run_parsing(self, extra_argv: list[str]) -> TrainingConfig:
        """Call from_env_and_args() with controlled argv."""
        original_argv = sys.argv[:]
        original_env = os.environ.copy()
        try:
            sys.argv = [
                "anvil-trainer",
                "--dataset.root=/tmp/fake",
                "--policy.type=diffusion",
                "--dataset.repo_id=local",
            ] + extra_argv
            # Clear env vars that might interfere
            os.environ.pop("LEROBOT_EXCLUDE_OBSERVATION", None)
            return TrainingConfig.from_env_and_args()
        finally:
            sys.argv = original_argv
            # Restore env
            for k, v in original_env.items():
                os.environ[k] = v
            for k in list(os.environ.keys()):
                if k not in original_env:
                    del os.environ[k]

    def test_parse_single_value(self):
        cfg = self._run_parsing(["--exclude-observation=velocity"])
        assert cfg.exclude_observation == ["velocity"]

    def test_parse_multiple_values(self):
        cfg = self._run_parsing(["--exclude-observation=images.chest,velocity,effort"])
        assert set(cfg.exclude_observation) == {"images.chest", "velocity", "effort"}

    def test_parse_empty_gives_none(self):
        cfg = self._run_parsing([])
        assert cfg.exclude_observation is None

    def test_env_var_fallback(self):
        original_argv = sys.argv[:]
        original_env = os.environ.copy()
        try:
            sys.argv = [
                "anvil-trainer",
                "--dataset.root=/tmp/fake",
                "--policy.type=diffusion",
                "--dataset.repo_id=local",
            ]
            os.environ["LEROBOT_EXCLUDE_OBSERVATION"] = "velocity,effort"
            cfg = TrainingConfig.from_env_and_args()
            assert set(cfg.exclude_observation) == {"velocity", "effort"}
        finally:
            sys.argv = original_argv
            for k, v in original_env.items():
                os.environ[k] = v
            for k in list(os.environ.keys()):
                if k not in original_env:
                    del os.environ[k]

    def test_cli_arg_removed_from_argv(self):
        """--exclude-observation must be stripped from sys.argv so lerobot doesn't choke."""
        original_argv = sys.argv[:]
        original_env = os.environ.copy()
        try:
            sys.argv = [
                "anvil-trainer",
                "--dataset.root=/tmp/fake",
                "--policy.type=diffusion",
                "--dataset.repo_id=local",
                "--exclude-observation=velocity",
            ]
            os.environ.pop("LEROBOT_EXCLUDE_OBSERVATION", None)
            TrainingConfig.from_env_and_args()
            assert not any(a.startswith("--exclude-observation") for a in sys.argv)
        finally:
            sys.argv = original_argv
            for k, v in original_env.items():
                os.environ[k] = v
            for k in list(os.environ.keys()):
                if k not in original_env:
                    del os.environ[k]
