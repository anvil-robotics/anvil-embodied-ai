from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_registry():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "ros2" / "src" / "lerobot_control" / "lerobot_control" / "policy_registry.py"
    spec = importlib.util.spec_from_file_location("policy_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_lerobot_policies_are_supported():
    registry = _load_registry()
    expected = {
        "act",
        "diffusion",
        "smolvla",
        "pi0",
        "pi05",
        "molmoact2",
        "groot",
        "multi_task_dit",
        "evo1",
        "fastwam",
        "vla_jepa",
    }

    assert expected == registry.SUPPORTED_POLICIES
    assert all(registry.is_supported_policy(policy) for policy in expected)
    assert not registry.is_supported_policy("openvla_oft")


def test_language_and_runtime_capabilities_are_separate():
    registry = _load_registry()

    assert registry.is_language_conditioned("multi_task_dit")
    assert registry.is_language_conditioned("fastwam")
    assert registry.is_language_conditioned("vla_jepa")
    assert not registry.uses_rtc_inference("multi_task_dit")
    assert not registry.uses_rtc_inference("fastwam")
    assert not registry.uses_rtc_inference("vla_jepa")
    assert registry.supports_rtc_inference("vla_jepa")
    assert not registry.resolve_rtc_inference("vla_jepa")
    assert registry.resolve_rtc_inference("vla_jepa", enabled=True)
    assert not registry.resolve_rtc_inference("vla_jepa", enabled=False)
    assert registry.uses_sync_chunk_inference("multi_task_dit")
    assert registry.uses_sync_chunk_inference("fastwam")
    assert registry.uses_sync_chunk_inference("vla_jepa")

    assert registry.uses_rtc_inference("pi05")
    assert registry.uses_sync_chunk_inference("pi05")
    assert registry.uses_rtc_inference("molmoact2")
    assert registry.uses_rtc_inference("groot")
    assert registry.uses_rtc_inference("evo1")
    assert not registry.uses_sync_chunk_inference("groot")

    assert not registry.is_language_conditioned("act")
    assert not registry.uses_rtc_inference("diffusion")
    assert not registry.supports_rtc_inference("diffusion")
    with pytest.raises(ValueError, match="does not support RTC"):
        registry.resolve_rtc_inference("diffusion", enabled=True)
