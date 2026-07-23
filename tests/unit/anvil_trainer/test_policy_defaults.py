from __future__ import annotations

import os
import sys

from anvil_trainer.config import TrainingConfig


def _parse_with_policy(policy_type: str) -> list[str]:
    original_argv = sys.argv[:]
    original_env = os.environ.copy()
    try:
        sys.argv = [
            "anvil-trainer",
            "--dataset.root=/tmp/fake",
            f"--policy.type={policy_type}",
        ]
        for key in (
            "LEROBOT_EXCLUDE_OBSERVS",
            "LEROBOT_EXCLUDE_OBSERVATION",
            "LEROBOT_TASK_OVERRIDE",
        ):
            os.environ.pop(key, None)

        TrainingConfig.from_env_and_args()
        return sys.argv[:]
    finally:
        sys.argv = original_argv
        os.environ.clear()
        os.environ.update(original_env)


def test_act_gets_default_backbone_flags():
    argv = _parse_with_policy("act")

    assert "--policy.vision_backbone=resnet18" in argv
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in argv


def test_selected_foundation_policy_skips_backbone_flags():
    argv = _parse_with_policy("molmoact2")

    assert not any(arg.startswith("--policy.vision_backbone=") for arg in argv)
    assert not any(arg.startswith("--policy.pretrained_backbone_weights=") for arg in argv)


def test_sync_foundation_policy_skips_backbone_flags():
    argv = _parse_with_policy("multi_task_dit")

    assert not any(arg.startswith("--policy.vision_backbone=") for arg in argv)
    assert not any(arg.startswith("--policy.pretrained_backbone_weights=") for arg in argv)


def test_yaml_config_values_are_not_replaced_by_cli_defaults(tmp_path):
    recipe = tmp_path / "train.yaml"
    recipe.write_text(
        """
dataset:
  repo_id: local
  root: /workspace/datasets/lego-in-cup
  video_backend: pyav
policy:
  path: lerobot/VLA-JEPA-Pretrain
  push_to_hub: false
output_dir: model_zoo/lego-in-cup/chunk20
job_name: lego-in-cup-chunk20
steps: 30000
env_eval_freq: 0
save_freq: 2500
wandb:
  project: lego-in-cup
  disable_artifact: true
""".strip()
    )

    original_argv = sys.argv[:]
    try:
        sys.argv = ["anvil-trainer", f"--config_path={recipe}"]
        config = TrainingConfig.from_env_and_args()

        assert config.dataset_root == "/workspace/datasets/lego-in-cup"
        assert config.output_dir == "model_zoo/lego-in-cup/chunk20"
        assert not any(arg.startswith("--output_dir=") for arg in sys.argv)
        assert not any(arg.startswith("--job_name=") for arg in sys.argv)
        assert "--steps=100000" not in sys.argv
        assert "--save_freq=10000" not in sys.argv
        assert "--policy.push_to_hub=false" not in sys.argv
        assert "--wandb.disable_artifact=true" not in sys.argv
        assert not any(arg.startswith("--policy.vision_backbone=") for arg in sys.argv)
    finally:
        sys.argv = original_argv
