from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from lerobot.configs.parser import _flatten_to_cli_args

from anvil_trainer.patches import (
    _flatten_config_to_cli_args,
    _vla_jepa_current_state,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs" / "training"
SHIRT_FOLD_CONFIGS = {
    "shirt_fold_pi05_hf_phase_aligned.yaml": (
        "lerobot-data-collection/folding_final",
        "695abe40dbf3aac04efda59c1501d748681fa0fb",
    ),
    "shirt_fold_pi05_base_phase_aligned.yaml": (
        "lerobot/pi05_base",
        "7de663972b7817d2c4cf2d84c821153dfea772e9",
    ),
}
SHIRT_FOLD_ACTION_NAMES = [
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
]
VLA_JEPA_REVISION = "e946c3e5b538d760f4b4ff239d1b1c12090c041d"
SMOLVLA_REVISION = "967623a0f38c7e1236c66b3893c830398d793ff7"


@pytest.mark.parametrize(
    ("filename", "expected_horizon", "expected_steps"),
    (
        ("lego_in_cup_vla_jepa_world_model.yaml", 32, 10000),
        ("lego_in_cup_smolvla.yaml", 50, 20000),
    ),
)
def test_lego_finetuning_configs_use_policy_specific_eight_dimensional_chunks(
    filename: str, expected_horizon: int, expected_steps: int
) -> None:
    config = yaml.safe_load((CONFIG_ROOT / filename).read_text())
    policy = config["policy"]

    assert policy["chunk_size"] == expected_horizon
    assert policy["n_action_steps"] == expected_horizon
    assert config["steps"] == expected_steps
    assert config["log_freq"] == 500

    input_features = json.loads(policy["input_features"])
    output_features = json.loads(policy["output_features"])
    assert input_features["observation.state"]["shape"] == [8]
    assert output_features["action"]["shape"] == [8]


def test_vla_jepa_world_model_reinitializes_only_dimension_dependent_heads() -> None:
    config = yaml.safe_load((CONFIG_ROOT / "lego_in_cup_vla_jepa_world_model.yaml").read_text())
    policy = config["policy"]

    assert policy["enable_world_model"] is True
    assert policy["pretrained_revision"] == VLA_JEPA_REVISION
    assert policy["freeze_qwen"] is False
    assert policy["action_dim"] == 8
    assert policy["state_dim"] == 8
    assert set(json.loads(policy["reinit_modules"])) == {
        "model.action_model.action_encoder",
        "model.action_model.action_decoder",
        "model.action_model.state_encoder",
    }
    overrides = _flatten_to_cli_args({key: value for key, value in policy.items() if key != "path"})
    assert f"--reinit_modules={policy['reinit_modules']}" in overrides


def test_smolvla_config_preserves_pretrained_expert_and_freezes_vision() -> None:
    config = yaml.safe_load((CONFIG_ROOT / "lego_in_cup_smolvla.yaml").read_text())
    policy = config["policy"]

    assert policy["path"] == "lerobot/smolvla_robotwin"
    assert policy["pretrained_revision"] == SMOLVLA_REVISION
    assert policy["load_vlm_weights"] is True
    assert policy["freeze_vision_encoder"] is True
    assert policy["train_expert_only"] is True
    assert policy["train_state_proj"] is True


@pytest.mark.parametrize(("filename", "expected_model"), SHIRT_FOLD_CONFIGS.items())
def test_shirt_fold_pi05_configs_pin_the_openarm2_dataset_contract(
    filename: str, expected_model: tuple[str, str]
) -> None:
    config = yaml.safe_load((CONFIG_ROOT / filename).read_text())
    policy = config["policy"]

    assert config["dataset"] == {
        "repo_id": "local",
        "root": "datasets/shirt-fold/lerobot-hf-phase-aligned",
        "video_backend": "pyav",
        "image_transforms": {"enable": False},
    }
    assert (policy["path"], policy["pretrained_revision"]) == expected_model
    assert policy["chunk_size"] == 30
    assert policy["n_action_steps"] == 30
    assert policy["max_state_dim"] == 32
    assert policy["max_action_dim"] == 32
    assert policy["use_relative_actions"] is True
    assert policy["relative_exclude_joints"] == ["gripper"]
    assert policy["action_feature_names"] == SHIRT_FOLD_ACTION_NAMES
    assert policy["dtype"] == "bfloat16"
    assert policy["gradient_checkpointing"] is True
    assert policy["compile_model"] is False
    assert policy["train_expert_only"] is True

    assert json.loads(policy["normalization_mapping"]) == {
        "VISUAL": "IDENTITY",
        "STATE": "QUANTILES",
        "ACTION": "QUANTILES",
    }
    input_features = json.loads(policy["input_features"])
    assert input_features == {
        "observation.images.left_wrist": {
            "type": "VISUAL",
            "shape": [3, 270, 480],
        },
        "observation.images.right_wrist": {
            "type": "VISUAL",
            "shape": [3, 270, 480],
        },
        "observation.images.base": {
            "type": "VISUAL",
            "shape": [3, 270, 480],
        },
        "observation.state": {"type": "STATE", "shape": [16]},
    }
    assert json.loads(policy["output_features"]) == {"action": {"type": "ACTION", "shape": [16]}}
    assert config["steps"] == 5000
    assert config["log_freq"] == 100
    assert config["save_freq"] == 500
    assert config["wandb"] == {
        "enable": True,
        "disable_artifact": True,
        "project": "openarm2-shirt-folding",
    }

    overrides = _flatten_config_to_cli_args(
        {key: value for key, value in policy.items() if key != "path"}
    )
    assert "--use_relative_actions=true" in overrides
    action_names_json = json.dumps(policy["action_feature_names"], separators=(",", ":"))
    assert f"--action_feature_names={action_names_json}" in overrides
    assert '--relative_exclude_joints=["gripper"]' in overrides
    assert "--optimizer_betas=[0.9,0.95]" in overrides


def test_shirt_fold_pi05_configs_differ_only_by_initialization_and_run_identity() -> None:
    configs = [
        yaml.safe_load((CONFIG_ROOT / filename).read_text()) for filename in SHIRT_FOLD_CONFIGS
    ]
    comparable = []
    for config in configs:
        candidate = deepcopy(config)
        candidate["policy"].pop("path")
        candidate["policy"].pop("pretrained_revision")
        candidate.pop("output_dir")
        candidate.pop("job_name")
        comparable.append(candidate)

    assert comparable[0] == comparable[1]


def test_vla_jepa_inference_uses_opt_in_full_rtc() -> None:
    config = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "lerobot_control" / "inference_lego_in_cup_vla_jepa.yaml"
        ).read_text()
    )
    sync = config["inference_tuning"]["sync"]
    rtc = config["inference_tuning"]["rtc"]

    assert sync["n_action_steps"] is None
    assert sync["async_prefetch"] is False
    assert rtc["enabled"] is True
    assert rtc["queue_trigger_threshold"] == 32
    assert rtc["execution_horizon"] == 12
    assert rtc["prefix_attention_schedule"] == "EXP"


def test_vla_jepa_stacked_state_uses_current_observation_not_future_state() -> None:
    state = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)

    selected = _vla_jepa_current_state(state)

    assert selected.shape == (2, 1, 8)
    assert torch.equal(selected[:, 0], state[:, 0])
    assert not torch.equal(selected[:, 0], state[:, -1])
