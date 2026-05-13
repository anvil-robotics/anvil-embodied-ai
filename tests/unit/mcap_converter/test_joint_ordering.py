"""Tests for the slimmed-down mcap_converter config + joint parsing.

Verifies:
1. parse_joint_name() with hardcoded conventions
2. Quest reorder permutation maps Float64MultiArray data order to canonical order
3. Pydantic DataConfig validation (action_source variants, arms required)
4. Loader round-trips the slim YAML shape
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from mcap_converter.config import ActionSource, DataConfig, load_config
from mcap_converter.core.constants import (
    QUEST_JOINT_ORDER,
    quest_command_topic,
)
from mcap_converter.core.extractor import (
    _QUEST_CANONICAL_JOINT_NAMES,
    _QUEST_REORDER,
    parse_joint_name,
)


# =============================================================================
# parse_joint_name
# =============================================================================


class TestParseJointName:
    def test_follower_left(self):
        assert parse_joint_name("follower_l_joint1") == ("observation", "left", "joint1")

    def test_follower_right(self):
        assert parse_joint_name("follower_r_joint5") == ("observation", "right", "joint5")

    def test_leader_right(self):
        assert parse_joint_name("leader_r_joint3") == ("action", "right", "joint3")

    def test_finger_joint(self):
        assert parse_joint_name("follower_l_finger_joint1") == (
            "observation", "left", "finger_joint1",
        )

    def test_unknown_source_returns_none(self):
        assert parse_joint_name("teacher_r_joint1") is None

    def test_unknown_arm_returns_none(self):
        assert parse_joint_name("follower_x_joint1") is None

    def test_too_few_parts_returns_none(self):
        assert parse_joint_name("follower_r") is None
        assert parse_joint_name("joint1") is None


# =============================================================================
# Quest reorder permutation
# =============================================================================


class TestQuestReorder:
    def test_canonical_names_are_sorted(self):
        assert _QUEST_CANONICAL_JOINT_NAMES == sorted(QUEST_JOINT_ORDER)
        assert _QUEST_CANONICAL_JOINT_NAMES[0] == "finger_joint1"

    def test_reorder_finger_last_to_first(self):
        # Float64MultiArray.data arrives in QUEST_JOINT_ORDER.
        data = np.array(
            [0.01, -0.35, -0.10, 1.69, -0.27, -0.46, -0.49, 0.05],
            dtype=np.float32,
        )
        reordered = data[_QUEST_REORDER]
        assert reordered[0] == pytest.approx(0.05)   # finger_joint1
        assert reordered[1] == pytest.approx(0.01)   # joint1
        assert reordered[-1] == pytest.approx(-0.49)  # joint7


# =============================================================================
# Pydantic schema validation
# =============================================================================


class TestDataConfig:
    def _base(self, **overrides):
        d = dict(
            camera_topic_mapping={"/cam/x/compressed": "head"},
            action_source=ActionSource.leader,
            arms=["right"],
        )
        d.update(overrides)
        return d

    def test_leader_valid(self):
        c = DataConfig(**self._base())
        assert c.action_source is ActionSource.leader
        assert c.arms == ["right"]

    def test_quest_teleop_valid(self):
        c = DataConfig(**self._base(action_source=ActionSource.quest_teleop, arms=["left", "right"]))
        assert c.action_source is ActionSource.quest_teleop

    def test_future_observations_requires_n_step(self):
        with pytest.raises(ValueError, match="action_n_step"):
            DataConfig(**self._base(action_source=ActionSource.future_observations))

    def test_future_observations_valid(self):
        c = DataConfig(**self._base(action_source=ActionSource.future_observations, action_n_step=10))
        assert c.action_n_step == 10

    def test_n_step_rejected_outside_future_observations(self):
        with pytest.raises(ValueError, match="action_n_step"):
            DataConfig(**self._base(action_n_step=5))

    def test_empty_arms_rejected(self):
        with pytest.raises(ValueError, match="arms"):
            DataConfig(**self._base(arms=[]))

    def test_duplicate_arms_rejected(self):
        with pytest.raises(ValueError, match="arms"):
            DataConfig(**self._base(arms=["right", "right"]))

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            DataConfig(**self._base(unknown_field="x"))

    def test_empty_camera_mapping_rejected(self):
        with pytest.raises(ValueError, match="camera_topic_mapping"):
            DataConfig(**self._base(camera_topic_mapping={}))


# =============================================================================
# Loader round-trip on production YAMLs
# =============================================================================


PROD_CONFIGS = [
    "openarm_bimanual.yaml",
    "openarm_bimanual_quest.yaml",
    "openarm_single_quest.yaml",
    "openarm_single_quest_afo.yaml",
]


@pytest.mark.parametrize("filename", PROD_CONFIGS)
def test_production_yaml_parses(filename):
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "configs" / "mcap_converter" / filename
    cfg = load_config(path)
    assert isinstance(cfg, DataConfig)
    assert cfg.camera_topic_mapping
    assert cfg.arms


def test_quest_command_topic_template():
    assert quest_command_topic("right") == "/follower_r_forward_position_controller/commands"
    assert quest_command_topic("left") == "/follower_l_forward_position_controller/commands"


def test_loader_yaml_roundtrip(tmp_path):
    cfg = DataConfig(
        camera_topic_mapping={"/cam/x/compressed": "head"},
        action_source=ActionSource.future_observations,
        arms=["right"],
        action_n_step=8,
    )
    y = tmp_path / "c.yaml"
    y.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    loaded = load_config(y)
    assert loaded == cfg
