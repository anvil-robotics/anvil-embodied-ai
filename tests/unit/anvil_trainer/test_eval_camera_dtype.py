"""Regression coverage for Anvil training compatibility patches."""

from types import SimpleNamespace

import pytest
import torch

from anvil_trainer.patches import (
    _make_pre_post_processors_with_compat,
    _normalize_uint8_camera_images,
    _remap_molmoact2_processor_overrides,
)


def test_custom_eval_normalizes_only_uint8_camera_images():
    batch = {
        "observation.images.waist": torch.tensor([0, 255], dtype=torch.uint8),
        "observation.state": torch.tensor([1, 2], dtype=torch.uint8),
    }

    result = _normalize_uint8_camera_images(batch, ("observation.images.waist",))

    assert result is batch
    assert batch["observation.images.waist"].dtype == torch.float32
    assert torch.equal(batch["observation.images.waist"], torch.tensor([0.0, 1.0]))
    assert batch["observation.state"].dtype == torch.uint8


def test_molmoact2_uses_saved_masked_normalizer_step_names():
    generic_stats_override = {"stats": {"action": {"mean": [0.0]}}}
    kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": generic_stats_override,
        },
        "postprocessor_overrides": {
            "unnormalizer_processor": generic_stats_override,
        },
    }

    result = _remap_molmoact2_processor_overrides(
        SimpleNamespace(type="molmoact2"), kwargs
    )

    assert "normalizer_processor" not in result["preprocessor_overrides"]
    assert (
        result["preprocessor_overrides"]["molmoact2_masked_normalizer"]
        is generic_stats_override
    )
    assert "unnormalizer_processor" not in result["postprocessor_overrides"]
    assert (
        result["postprocessor_overrides"]["molmoact2_masked_unnormalizer"]
        is generic_stats_override
    )
    assert "normalizer_processor" in kwargs["preprocessor_overrides"]


def test_legacy_delta_action_checkpoint_remaps_relative_action_override():
    relative_override = {
        "enabled": True,
        "exclude_joints": ["gripper"],
        "action_names": ["right_joint_1.pos"],
    }
    kwargs = {
        "policy_cfg": SimpleNamespace(type="pi05"),
        "pretrained_path": "lerobot-data-collection/folding_final",
        "preprocessor_overrides": {
            "device_processor": {"device": "cuda"},
            "relative_actions_processor": relative_override,
        },
    }
    calls = []

    def make_processors(**call_kwargs):
        calls.append(call_kwargs)
        overrides = call_kwargs["preprocessor_overrides"]
        if "relative_actions_processor" in overrides:
            raise KeyError(
                "Override keys ['relative_actions_processor'] do not match any step in "
                "the saved configuration. Available step keys: "
                "['delta_actions_processor', 'normalizer_processor']."
            )
        return "preprocessor", "postprocessor"

    result = _make_pre_post_processors_with_compat(make_processors, **kwargs)

    assert result == ("preprocessor", "postprocessor")
    assert len(calls) == 2
    assert "relative_actions_processor" in calls[0]["preprocessor_overrides"]
    assert "relative_actions_processor" not in calls[1]["preprocessor_overrides"]
    assert calls[1]["preprocessor_overrides"]["delta_actions_processor"] is relative_override
    assert "relative_actions_processor" in kwargs["preprocessor_overrides"]


def test_unrelated_processor_key_error_is_not_retried():
    calls = 0

    def make_processors(**_kwargs):
        nonlocal calls
        calls += 1
        raise KeyError("unrelated processor failure")

    with pytest.raises(KeyError, match="unrelated processor failure"):
        _make_pre_post_processors_with_compat(
            make_processors,
            policy_cfg=SimpleNamespace(type="pi05"),
            preprocessor_overrides={
                "relative_actions_processor": {"enabled": True},
            },
        )

    assert calls == 1
