"""Tests for TrainingConfig.validate_action_space().

Covers:
  1. EE dataset + ee_abs    → passes
  2. EE dataset + ee_rel    → passes ("ee_rel" legacy alias for "ee_relative")
  3. EE dataset + joint_abs → DataIntegrityError
  4. Joint dataset + ee_abs → DataIntegrityError
  5. Joint dataset + joint_abs → passes
  6. EE dataset with bad state dim (not multiple of 8) → DataIntegrityError
  7. EE dataset with bad action dim (not 10 * n_arms) → DataIntegrityError
  8. Missing info.json → passes silently (logged warning)
  9. Missing dataset_root (None) → passes silently
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

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

    def test_bad_state_dim_not_multiple_of_8(self):
        """EE state dim 9 is not a multiple of 8 → error."""
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
            with pytest.raises(DataIntegrityError, match="multiple of 8"):
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
