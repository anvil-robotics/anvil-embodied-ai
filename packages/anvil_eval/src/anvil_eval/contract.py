"""Static checkpoint/dataset/inference contract validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def resolve_pretrained_model(checkpoint_path: Path) -> Path:
    """Resolve either a checkpoint step or a pretrained_model directory."""
    resolved = checkpoint_path.resolve()
    if (resolved / "config.json").exists():
        return resolved
    candidate = resolved / "pretrained_model"
    if (candidate / "config.json").exists():
        return candidate
    raise FileNotFoundError(
        f"No config.json or pretrained_model/config.json under {checkpoint_path}"
    )


def audit_policy_contract(
    checkpoint_path: Path,
    dataset_path: Path,
    inference_config_path: Path,
) -> dict[str, Any]:
    """Compare vector, camera, processor, and scheduler contracts."""
    model_dir = resolve_pretrained_model(checkpoint_path)
    model_config = _load_json(model_dir / "config.json")
    preprocessor = _load_json(model_dir / "policy_preprocessor.json")
    postprocessor = _load_json(model_dir / "policy_postprocessor.json")
    dataset_info = _load_json(dataset_path / "meta" / "info.json")
    inference_config = yaml.safe_load(inference_config_path.read_text())
    conversion_path = dataset_path / "conversion_config.yaml"
    conversion = yaml.safe_load(conversion_path.read_text()) if conversion_path.exists() else {}

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    dataset_features = dataset_info.get("features", {})
    dataset_action = dataset_features.get("action", {})
    dataset_state = dataset_features.get("observation.state", {})
    action_names = list(dataset_action.get("names") or [])
    state_names = list(dataset_state.get("names") or [])
    checkpoint_names = list(model_config.get("action_feature_names") or [])
    relative_step = _processor_step(preprocessor, "relative_actions_processor")
    processor_names = list((relative_step or {}).get("config", {}).get("action_names") or [])

    _expect_equal(errors, "dataset action/state names", action_names, state_names)
    _expect_equal(errors, "checkpoint/dataset action names", checkpoint_names, action_names)
    _expect_equal(errors, "relative processor/dataset action names", processor_names, action_names)

    state_dim = _feature_dim(model_config, "input_features", "observation.state")
    action_dim = _feature_dim(model_config, "output_features", "action")
    _expect_equal(errors, "checkpoint state/action dimensions", state_dim, action_dim)
    _expect_equal(errors, "checkpoint/dataset action dimension", action_dim, len(action_names))

    arm_mapping = inference_config.get("joint_names", {}).get("arm_mapping", {})
    runtime_arm_order = list(arm_mapping.values())
    checkpoint_arm_order = _arm_order(checkpoint_names)
    _expect_equal(errors, "runtime/checkpoint arm order", runtime_arm_order, checkpoint_arm_order)

    arms = inference_config.get("arms", {})
    expected_start = 0
    arm_slices: dict[str, list[int]] = {}
    for arm_name in runtime_arm_order:
        arm = arms.get(arm_name)
        if arm is None:
            errors.append(f"Missing runtime arm configuration for {arm_name}")
            continue
        start = int(arm.get("action_start", -1))
        end = int(arm.get("action_end", -1))
        arm_slices[arm_name] = [start, end]
        if start != expected_start or end <= start:
            errors.append(
                f"Non-contiguous action slice for {arm_name}: [{start}, {end}), expected start {expected_start}"
            )
        expected_start = end
    if expected_start != action_dim:
        errors.append(
            f"Runtime arm slices cover {expected_start} actions, checkpoint expects {action_dim}"
        )

    joint_order = inference_config.get("joint_names", {}).get("model_joint_order", [])
    if len(joint_order) * len(runtime_arm_order) != action_dim:
        errors.append(
            "Runtime model_joint_order multiplied by arm count does not match checkpoint action dimension"
        )

    checkpoint_cameras = {
        key.removeprefix("observation.images.")
        for key in model_config.get("input_features", {})
        if key.startswith("observation.images.")
    }
    dataset_cameras = {
        key.removeprefix("observation.images.")
        for key in dataset_features
        if key.startswith("observation.images.")
    }
    runtime_cameras = set(inference_config.get("cameras", {}).get("mapping", {}).values())
    _expect_equal(errors, "checkpoint/dataset camera roles", checkpoint_cameras, dataset_cameras)
    _expect_equal(errors, "runtime/checkpoint camera roles", runtime_cameras, checkpoint_cameras)

    camera_shapes: dict[str, Any] = {}
    for camera in sorted(checkpoint_cameras | dataset_cameras):
        checkpoint_shape = (
            model_config.get("input_features", {})
            .get(f"observation.images.{camera}", {})
            .get("shape")
        )
        dataset_shape = dataset_features.get(f"observation.images.{camera}", {}).get("shape")
        camera_shapes[camera] = {
            "checkpoint": checkpoint_shape,
            "dataset": dataset_shape,
        }
        if checkpoint_shape and dataset_shape and checkpoint_shape != dataset_shape:
            warnings.append(
                f"Camera {camera} metadata shape differs: checkpoint={checkpoint_shape}, dataset={dataset_shape}"
            )
        if (
            checkpoint_shape
            and dataset_shape
            and _aspect_ratio(checkpoint_shape) != _aspect_ratio(dataset_shape)
        ):
            warnings.append(
                f"Camera {camera} aspect ratio differs between checkpoint and dataset metadata"
            )

    relative_enabled = bool((relative_step or {}).get("config", {}).get("enabled", False))
    absolute_step = _processor_step(postprocessor, "absolute_actions_processor")
    absolute_enabled = bool((absolute_step or {}).get("config", {}).get("enabled", False))
    if model_config.get("use_relative_actions") and not relative_enabled:
        errors.append(
            "Checkpoint requests relative actions but preprocessor conversion is disabled"
        )
    if relative_enabled and not absolute_enabled:
        errors.append("Relative preprocessor is enabled but absolute postprocessor is disabled")

    conversion_order = conversion.get("robot_order")
    if conversion_order:
        _expect_equal(
            errors, "conversion/checkpoint arm order", conversion_order, checkpoint_arm_order
        )
    conversion_names = conversion.get("feature_names")
    if conversion_names:
        _expect_equal(
            errors, "conversion/checkpoint feature names", conversion_names, checkpoint_names
        )

    sync_config = inference_config.get("inference_tuning", {}).get("sync", {})
    rtc_config = inference_config.get("inference_tuning", {}).get("rtc", {})
    if rtc_config.get("enabled"):
        warnings.append(
            "RTC is enabled; use non-RTC sync prefetch for the initial Pi0.5 sanity test"
        )
    if not sync_config.get("async_prefetch"):
        warnings.append("Synchronous prefetch is disabled")

    checks.update(
        {
            "checkpoint_type": model_config.get("type"),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_names": checkpoint_names,
            "runtime_arm_order": runtime_arm_order,
            "checkpoint_arm_order": checkpoint_arm_order,
            "arm_slices": arm_slices,
            "camera_roles": sorted(checkpoint_cameras),
            "camera_shapes": camera_shapes,
            "relative_actions": {
                "model_enabled": bool(model_config.get("use_relative_actions")),
                "preprocessor_enabled": relative_enabled,
                "postprocessor_enabled": absolute_enabled,
                "exclude_joints": (relative_step or {}).get("config", {}).get("exclude_joints", []),
            },
            "scheduler": {
                "chunk_size": model_config.get("chunk_size"),
                "n_action_steps": model_config.get("n_action_steps"),
                "rtc_enabled": bool(rtc_config.get("enabled")),
                "async_prefetch": bool(sync_config.get("async_prefetch")),
                "prefetch_threshold": sync_config.get("prefetch_threshold"),
                "replace_pending_actions": bool(sync_config.get("replace_pending_actions")),
            },
            "max_position_delta": inference_config.get("safety", {}).get("max_position_delta"),
        }
    )

    return {
        "status": "error" if errors else "pass_with_warnings" if warnings else "pass",
        "checkpoint": str(model_dir),
        "dataset": str(dataset_path.resolve()),
        "inference_config": str(inference_config_path.resolve()),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def inspect_debug_images(debug_dir: Path, camera_roles: list[str]) -> dict[str, Any]:
    """Report the latest saved image shape and mode for each required camera role."""
    report: dict[str, Any] = {}
    for camera in camera_roles:
        camera_dir = debug_dir / camera
        candidates = sorted(camera_dir.glob("frame_*.png")) if camera_dir.exists() else []
        if not candidates:
            report[camera] = {"status": "missing"}
            continue
        path = candidates[-1]
        with Image.open(path) as image:
            report[camera] = {
                "status": "found",
                "path": str(path.resolve()),
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "aspect_ratio": image.width / image.height,
            }
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _processor_step(config: dict[str, Any], registry_name: str) -> dict[str, Any] | None:
    for step in config.get("steps", []):
        if step.get("registry_name") == registry_name:
            return step
    return None


def _feature_dim(config: dict[str, Any], group: str, key: str) -> int:
    shape = config.get(group, {}).get(key, {}).get("shape") or []
    if len(shape) != 1:
        raise ValueError(f"Expected one-dimensional {group}.{key}, got {shape}")
    return int(shape[0])


def _arm_order(names: list[str]) -> list[str]:
    order: list[str] = []
    for name in names:
        arm = name.split("_", maxsplit=1)[0]
        if arm not in order:
            order.append(arm)
    return order


def _aspect_ratio(shape: list[int]) -> float:
    return round(float(shape[-1]) / float(shape[-2]), 6)


def _expect_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} mismatch: actual={actual!r}, expected={expected!r}")
