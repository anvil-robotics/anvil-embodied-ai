"""Tests for TrainingConfig.validate_action_space().

Covers:
  1. EE dataset + ee_abs    → passes
  2. EE dataset + ee_rel    → passes ("ee_rel" legacy alias for "ee_relative")
  3. EE dataset + joint_abs → DataIntegrityError
  4. Joint dataset + ee_abs → DataIntegrityError
  5. Joint dataset + joint_abs → passes
  6. EE dataset with state dim mismatched for its observation_encoding → DataIntegrityError
  7. EE dataset with bad action dim (not 10 * n_arms) → DataIntegrityError
  8. Missing info.json → passes silently (logged warning)
  9. Missing dataset_root (None) → passes silently
  10. EE dataset + ee_delta → passes; joint dataset + ee_delta → DataIntegrityError
  11. Per-observation_encoding validation (quaternion/rot6d/axis_angle), both via the
      dataset's conversion_config.yaml (ground truth) and via the marker-suffix fallback
      for datasets that predate it — this is the gap-3 fix: the old check hardcoded
      quaternion's 8-per-arm layout and silently misclassified rot6d/axis_angle datasets.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import DataIntegrityError


# ── info.json factory helpers ─────────────────────────────────────────────────

def _write_info(tmp_dir: Path, state_names: list[str], action_names: list[str]) -> None:
    """Write a minimal meta/info.json into tmp_dir."""
    meta_dir = tmp_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "features": {
            "observation.state": {
                "names": state_names,
                "shape": [len(state_names)],
            },
            "action": {
                "names": action_names,
                "shape": [len(action_names)],
            },
        }
    }
    (meta_dir / "info.json").write_text(json.dumps(info))


def _ee_state_names(n_arms: int = 1) -> list[str]:
    """EE state names: [x,y,z,qx,qy,qz,qw,gripper] per arm."""
    dims = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    prefix = ["left", "right"]
    return [f"{prefix[arm]}_{d}" for arm in range(n_arms) for d in dims]


def _ee_action_names(n_arms: int = 1) -> list[str]:
    """EE action names: [x,y,z,r0..r5,gripper] per arm."""
    dims = ["x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "gripper"]
    prefix = ["left", "right"]
    return [f"{prefix[arm]}_{d}" for arm in range(n_arms) for d in dims]


def _joint_state_names(n_joints: int = 8) -> list[str]:
    return [f"joint{i}" for i in range(n_joints)]


def _joint_action_names(n_joints: int = 8) -> list[str]:
    return [f"joint{i}" for i in range(n_joints)]


_ROTATION_SUFFIXES = {
    "quaternion": ["qx", "qy", "qz", "qw"],
    "rot6d": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "axis_angle": ["ax", "ay", "az"],
}


def _ee_state_names_encoded(observation_encoding: str, n_arms: int = 1) -> list[str]:
    """EE state names for a given observation_encoding: [x,y,z,<rot>,gripper] per arm."""
    dims = ["x", "y", "z", *_ROTATION_SUFFIXES[observation_encoding], "gripper"]
    prefix = ["left", "right"]
    return [f"{prefix[arm]}_{d}" for arm in range(n_arms) for d in dims]


def _write_conversion_config(
    tmp_dir: Path,
    *,
    data_space: str,
    action_encoding: str = "absolute",
    observation_encoding: str = "quaternion",
) -> None:
    """Write a minimal conversion_config.yaml, as mcap-convert would."""
    cfg = {
        "data_space": data_space,
        "action_encoding": action_encoding,
        "observation_encoding": observation_encoding,
    }
    Path(tmp_dir, "conversion_config.yaml").write_text(yaml.safe_dump(cfg))


def _make_config(dataset_root: str | Path, action_type: str) -> TrainingConfig:
    return TrainingConfig(
        dataset_root=str(dataset_root),
        action_type=action_type,
        split_ratio=[8.0, 1.0, 1.0],
    )


# ── 1–5: Dataset/action_type matching ────────────────────────────────────────

class TestActionTypeDatasetMatch:
    def test_ee_dataset_ee_abs_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names(), _ee_action_names())
            cfg = _make_config(tmp, "ee_abs")
            cfg.validate_action_space()  # must not raise

    def test_ee_dataset_ee_rel_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names(), _ee_action_names())
            cfg = _make_config(tmp, "ee_rel")
            cfg.validate_action_space()  # must not raise

    def test_ee_dataset_joint_abs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names(), _ee_action_names())
            cfg = _make_config(tmp, "joint_abs")
            with pytest.raises(DataIntegrityError, match="joint_abs.*EE-space"):
                cfg.validate_action_space()

    def test_joint_dataset_ee_abs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _joint_state_names(), _joint_action_names())
            cfg = _make_config(tmp, "ee_abs")
            with pytest.raises(DataIntegrityError, match="ee_abs.*joint-space"):
                cfg.validate_action_space()

    def test_joint_dataset_ee_rel_raises(self):
        """"ee_rel" legacy alias is normalized to "ee_relative" before validation,
        so the error message reports the canonical name."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _joint_state_names(), _joint_action_names())
            cfg = _make_config(tmp, "ee_rel")
            assert cfg.action_type == "ee_relative"
            with pytest.raises(DataIntegrityError, match="ee_relative.*joint-space"):
                cfg.validate_action_space()

    def test_joint_dataset_joint_abs_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _joint_state_names(), _joint_action_names())
            cfg = _make_config(tmp, "joint_abs")
            cfg.validate_action_space()  # must not raise


# ── 6–7: EE dimension validation ─────────────────────────────────────────────

class TestEEDimensionValidation:
    def _write_raw_info(
        self,
        tmp_dir: Path,
        state_dim: int,
        action_dim: int,
        state_names: list[str],
        action_names: list[str],
    ) -> None:
        """Write info.json with explicit shape override (for bad-dim tests)."""
        meta_dir = tmp_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "features": {
                "observation.state": {
                    "names": state_names,
                    "shape": [state_dim],
                },
                "action": {
                    "names": action_names,
                    "shape": [action_dim],
                },
            }
        }
        (meta_dir / "info.json").write_text(json.dumps(info))

    def test_bad_state_dim_mismatch(self):
        """EE state dim 9 doesn't match quaternion's per-arm dim (8) → error."""
        with tempfile.TemporaryDirectory() as tmp:
            # Use EE marker names but force a bad shape
            state_names = _ee_state_names(1) + ["extra"]  # 9 names
            action_names = _ee_action_names(1)
            self._write_raw_info(
                Path(tmp),
                state_dim=9, action_dim=10,
                state_names=state_names, action_names=action_names,
            )
            cfg = _make_config(tmp, "ee_rel")
            with pytest.raises(DataIntegrityError, match="observation.state dim 9"):
                cfg.validate_action_space()

    def test_bad_action_dim_mismatch(self):
        """EE dataset with 1 arm (state_dim=8) but action_dim=12 → error."""
        with tempfile.TemporaryDirectory() as tmp:
            state_names = _ee_state_names(1)
            action_names = _ee_action_names(1)[:10]  # correct names but wrong shape
            self._write_raw_info(
                Path(tmp),
                state_dim=8, action_dim=12,
                state_names=state_names, action_names=action_names,
            )
            cfg = _make_config(tmp, "ee_abs")
            with pytest.raises(DataIntegrityError, match="action dim"):
                cfg.validate_action_space()

    def test_bimanual_ee_passes(self):
        """Bimanual EE (state=16, action=20) passes."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names(2), _ee_action_names(2))
            cfg = _make_config(tmp, "ee_rel")
            cfg.validate_action_space()  # must not raise


# ── 10: ee_delta action_type ─────────────────────────────────────────────────

class TestEEDeltaActionType:
    def test_ee_dataset_ee_delta_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names(), _ee_action_names())
            cfg = _make_config(tmp, "ee_delta")
            cfg.validate_action_space()  # must not raise

    def test_joint_dataset_ee_delta_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _joint_state_names(), _joint_action_names())
            cfg = _make_config(tmp, "ee_delta")
            with pytest.raises(DataIntegrityError, match="ee_delta.*joint-space"):
                cfg.validate_action_space()


# ── 11: observation_encoding-aware validation (gap-3 fix) ────────────────────

class TestObservationEncodingAware:
    """validate_action_space must derive the expected per-arm state dim from the
    dataset's own observation_encoding (quaternion=8, rot6d=10, axis_angle=7), not
    assume quaternion — the pre-existing bug this fix addresses would silently
    misclassify rot6d/axis_angle EE datasets as joint-space."""

    @pytest.mark.parametrize("encoding", ["quaternion", "rot6d", "axis_angle"])
    def test_ee_dataset_passes_with_conversion_config(self, encoding):
        """conversion_config.yaml declares the encoding — the ground-truth path."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names_encoded(encoding), _ee_action_names())
            _write_conversion_config(Path(tmp), data_space="ee", observation_encoding=encoding)
            cfg = _make_config(tmp, "ee_abs")
            cfg.validate_action_space()  # must not raise

    @pytest.mark.parametrize("encoding", ["quaternion", "rot6d", "axis_angle"])
    def test_ee_dataset_passes_via_suffix_fallback(self, encoding):
        """No conversion_config.yaml — falls back to marker-suffix detection, which
        must recognize rot6d/axis_angle markers too, not just quaternion's."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names_encoded(encoding), _ee_action_names())
            cfg = _make_config(tmp, "ee_abs")
            cfg.validate_action_space()  # must not raise

    def test_state_dim_mismatched_for_declared_encoding_raises(self):
        """conversion_config.yaml says rot6d (10/arm) but state is quaternion-shaped
        (8/arm) → error, not a silent pass."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names_encoded("quaternion"), _ee_action_names())
            _write_conversion_config(Path(tmp), data_space="ee", observation_encoding="rot6d")
            cfg = _make_config(tmp, "ee_abs")
            with pytest.raises(DataIntegrityError, match="observation.state dim"):
                cfg.validate_action_space()

    def test_conversion_config_data_space_is_ground_truth(self):
        """conversion_config.yaml's data_space=ee is trusted directly, independent of
        feature-name suffix detection (the design intent: read the on-disk config
        rather than guess)."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_info(Path(tmp), _ee_state_names_encoded("rot6d"), _ee_action_names())
            _write_conversion_config(Path(tmp), data_space="ee", observation_encoding="rot6d")
            cfg = _make_config(tmp, "ee_delta")
            cfg.validate_action_space()  # must not raise


# ── 8–9: Missing dataset / info.json ─────────────────────────────────────────

class TestMissingInfo:
    def test_missing_info_json_skips_silently(self):
        """No info.json → warning logged, no exception."""
        with tempfile.TemporaryDirectory() as tmp:
            # meta/ directory exists but no info.json
            (Path(tmp) / "meta").mkdir()
            cfg = _make_config(tmp, "ee_rel")
            cfg.validate_action_space()  # must not raise

    def test_no_dataset_root_skips(self):
        """dataset_root=None → validation skipped silently."""
        cfg = TrainingConfig(
            dataset_root=None,
            action_type="ee_rel",
            split_ratio=[8.0, 1.0, 1.0],
        )
        cfg.validate_action_space()  # must not raise
