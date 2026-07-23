import json

import pytest
from anvil_shared.embodiment import (
    EmbodimentContract,
    EmbodimentError,
    ExperimentContract,
    PolicyBinding,
    normalize_from_limits,
)


def _scene(tmp_path):
    root = tmp_path / "scene.anvilscene"
    root.mkdir()
    embodiment = {
        "schema_version": 1,
        "robot": "anvil_openarm_2",
        "control_hz": 30,
        "arms": {
            "l": {
                "command_topic": "/left/commands",
                "command_order": [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"],
                "tcp_site": "left_tcp",
            }
        },
        "joint_ranges": {f"joint{i}": [-1, 1] for i in range(1, 8)},
        "action_surfaces": ["joint_position", "relative_end_effector"],
    }
    (root / "embodiment.json").write_text(json.dumps(embodiment))
    (root / "scene_manifest.json").write_text(
        json.dumps({"schema_version": 1, "embodiment_manifest": "embodiment.json"})
    )
    return root


def test_pi05_binding_accepts_active_eight_dimensional_arm(tmp_path):
    embodiment = EmbodimentContract.load(_scene(tmp_path) / "embodiment.json")
    binding = PolicyBinding.from_dict(
        {
            "model_type": "pi05",
            "arm": "l",
            "action_surface": "joint_position",
            "normalization_source": "embodiment_limits",
            "active_state_dim": 8,
            "active_action_dim": 8,
            "padded_capacity": 32,
            "camera_roles": {"base": "waist"},
        }
    )
    binding.validate(embodiment)


def test_trained_baseline_requires_checkpoint_normalization(tmp_path):
    embodiment = EmbodimentContract.load(_scene(tmp_path) / "embodiment.json")
    binding = PolicyBinding.from_dict(
        {
            "model_type": "act",
            "arm": "l",
            "action_surface": "joint_position",
            "normalization_source": "embodiment_limits",
            "active_state_dim": 8,
            "active_action_dim": 8,
            "camera_roles": {"base": "waist"},
        }
    )
    with pytest.raises(EmbodimentError, match="checkpoint normalization"):
        binding.validate(embodiment)


def test_limit_normalization_clamps_and_preserves_gripper():
    assert normalize_from_limits([-2, 0.25], [(-1, 1), None]) == [-1.0, 0.25]


def test_one_shot_requires_one_existing_demo(tmp_path):
    scene = _scene(tmp_path)
    raw = {
        "scene_bundle": str(scene),
        "instruction": "put the Lego in the cup",
        "success_threshold": 0.8,
        "seeds": list(range(20)),
        "one_shot": {"demonstration": None},
        "models": [
            {
                "model_type": "smolvla",
                "arm": "l",
                "action_surface": "joint_position",
                "normalization_source": "embodiment_limits",
                "active_state_dim": 8,
                "active_action_dim": 8,
                "camera_roles": {"base": "waist"},
            }
        ],
    }
    experiment = ExperimentContract.from_dict(raw, tmp_path)
    experiment.validate(mode="zero")
    with pytest.raises(EmbodimentError, match="exactly one"):
        experiment.validate(mode="one")


def test_hybrid_renderer_covers_policy_cameras(tmp_path):
    scene = _scene(tmp_path)
    raw = {
        "scene_bundle": str(scene),
        "instruction": "put the Lego in the cup",
        "success_threshold": 0.8,
        "seeds": list(range(20)),
        "observation_renderer": {
            "type": "gaussian_mujoco_hybrid",
            "cameras": ["waist"],
            "gaussian": {"render_mode": "RGB+ED"},
        },
        "models": [
            {
                "model_type": "smolvla",
                "arm": "l",
                "action_surface": "joint_position",
                "normalization_source": "embodiment_limits",
                "active_state_dim": 8,
                "active_action_dim": 8,
                "camera_roles": {"base": "waist", "wrist": "wrist_l"},
            }
        ],
    }
    with pytest.raises(EmbodimentError, match="missing policy cameras"):
        ExperimentContract.from_dict(raw, tmp_path).validate(mode="zero")
    raw["observation_renderer"]["cameras"].append("wrist_l")
    ExperimentContract.from_dict(raw, tmp_path).validate(mode="zero")
