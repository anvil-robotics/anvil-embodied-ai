from __future__ import annotations

from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

from anvil_trainer.patches import reconcile_vla_jepa_postprocessor


def _pretrained_postprocessor():
    base_config = VLAJEPAConfig(
        pre_snap_gripper_action=True,
        binarize_gripper_action=True,
        gripper_dim=6,
    )
    _, postprocessor = make_vla_jepa_pre_post_processors(base_config)
    return postprocessor


def test_reconcile_removes_disabled_inherited_gripper_steps():
    postprocessor = _pretrained_postprocessor()
    effective_config = VLAJEPAConfig(
        pre_snap_gripper_action=False,
        binarize_gripper_action=False,
        gripper_dim=0,
    )

    removed = reconcile_vla_jepa_postprocessor(effective_config, postprocessor)
    registry_names = [step["registry_name"] for step in postprocessor.get_config()["steps"]]

    assert removed == ["PreSnapGripperProcessorStep", "BinarizeGripperProcessorStep"]
    assert "vla_jepa_pre_snap_gripper" not in registry_names
    assert "vla_jepa_binarize_gripper" not in registry_names


def test_reconcile_serializes_effective_gripper_dimension_when_enabled():
    postprocessor = _pretrained_postprocessor()
    effective_config = VLAJEPAConfig(
        pre_snap_gripper_action=True,
        binarize_gripper_action=True,
        gripper_dim=0,
        gripper_threshold=0.25,
    )

    reconcile_vla_jepa_postprocessor(effective_config, postprocessor)
    steps = postprocessor.get_config()["steps"]
    gripper_steps = [
        step
        for step in steps
        if step["registry_name"]
        in {"vla_jepa_pre_snap_gripper", "vla_jepa_binarize_gripper"}
    ]

    assert len(gripper_steps) == 2
    assert all(step["config"] == {"gripper_dim": 0, "threshold": 0.25} for step in gripper_steps)
