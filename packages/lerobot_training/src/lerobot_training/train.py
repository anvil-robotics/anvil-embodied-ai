#!/usr/bin/env python3
"""
LeRobot Training with Pluggable Customizations

This module wraps lerobot-train to provide extensible training customizations.
The plugin architecture supports various customization types:

Current implementations (Dataset Transforms):
    - Camera filtering: Train with subset of cameras
    - Task override: Override dataset task for SmolVLA
    - Delta actions: Convert actions to relative (action - observation.state)

Future extension areas:
    - Model architecture modifications
    - Image preprocessing pipelines
    - Custom loss functions
    - Data augmentation strategies
    - Training callbacks

Architecture:
    TrainingConfig  - Central configuration for all customizations
    Transform       - Abstract base class for dataset transforms
    TransformRunner - Applies transforms via monkey-patching

Adding new dataset transforms:
    1. Create a new Transform subclass
    2. Add configuration field to TrainingConfig
    3. Register in TransformRunner.TRANSFORMS

Usage:
    # CLI
    lerobot-train [lerobot args] [--use-delta-actions]

    # Python
    from lerobot_training import train, TrainingConfig
    config = TrainingConfig(cameras=["chest", "waist"])
    train(config)

Environment variables:
    LEROBOT_CAMERA_FILTER: Comma-separated list of cameras to include
    LEROBOT_TASK_OVERRIDE: Override task string for all samples
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Configuration for custom training transformations.

    Attributes:
        cameras: List of camera names to include (None = all cameras)
        task_override: Override task string for all samples (for SmolVLA)
        use_delta_actions: Convert actions to delta (action - observation.state)
        dataset_root: Path to local dataset (for validation)
    """

    cameras: list[str] | None = None
    task_override: str | None = None
    use_delta_actions: bool = False
    dataset_root: str | None = None

    @classmethod
    def from_env_and_args(cls) -> TrainingConfig:
        """
        Parse configuration from environment variables and command line args.

        Environment variables:
            LEROBOT_CAMERA_FILTER: Comma-separated camera names
            LEROBOT_TASK_OVERRIDE: Task string override

        Command line args:
            --use-delta-actions: Enable delta action transform
        """
        # Camera filter from environment
        camera_env = os.environ.get("LEROBOT_CAMERA_FILTER", "")
        cameras = [c.strip() for c in camera_env.split(",") if c.strip()] or None

        # Task override from environment
        task_override = os.environ.get("LEROBOT_TASK_OVERRIDE", "") or None

        # Delta actions from command line (remove to avoid lerobot arg parsing error)
        use_delta_actions = "--use-delta-actions" in sys.argv
        if use_delta_actions:
            sys.argv.remove("--use-delta-actions")

        # Try to extract dataset root from args for validation
        dataset_root = None
        for arg in sys.argv:
            if arg.startswith("--dataset.root="):
                dataset_root = arg.split("=", 1)[1]
                break

        return cls(
            cameras=cameras,
            task_override=task_override,
            use_delta_actions=use_delta_actions,
            dataset_root=dataset_root,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TrainingConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(
            cameras=data.get("cameras"),
            task_override=data.get("task_override"),
            use_delta_actions=data.get("use_delta_actions", False),
            dataset_root=data.get("dataset_root"),
        )

    def validate_cameras(self) -> list[str]:
        """
        Validate camera names against dataset metadata.

        Returns:
            List of invalid camera names (empty if all valid)

        Raises:
            FileNotFoundError: If dataset metadata not found
        """
        if not self.cameras or not self.dataset_root:
            return []

        info_path = Path(self.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset info not found: {info_path}")

        with open(info_path) as f:
            info = json.load(f)

        features = info.get("features", {})
        available_cameras = [
            key.replace("observation.images.", "")
            for key in features
            if key.startswith("observation.images.")
        ]

        invalid = [c for c in self.cameras if c not in available_cameras]
        return invalid


# =============================================================================
# Dataset Transform Base Class
# =============================================================================


class Transform(ABC):
    """
    Abstract base class for dataset transforms.

    Subclasses implement specific transformations applied to dataset items
    during training. Each transform can optionally patch LeRobot internals
    for metadata filtering.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @abstractmethod
    def is_enabled(self, config: TrainingConfig) -> bool:
        """Check if this transform should be applied."""

    @abstractmethod
    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        """
        Apply transform to a single dataset item.

        Args:
            item: Dataset item from LeRobotDataset.__getitem__
            config: Training configuration

        Returns:
            Transformed item
        """

    def patch_metadata(self, config: TrainingConfig) -> None:  # noqa: B027
        """
        Optional: Patch LeRobot metadata/utils before training.

        Override this method if the transform needs to modify how
        LeRobot builds the policy (e.g., filtering input features).
        """


# =============================================================================
# Dataset Transform Implementations
# =============================================================================


class CameraFilterTransform(Transform):
    """Filter dataset to include only specified cameras."""

    @property
    def name(self) -> str:
        return "camera_filter"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.cameras is not None

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if not config.cameras:
            return item

        keys_to_remove = [
            key
            for key in item
            if key.startswith("observation.images.")
            and key.replace("observation.images.", "") not in config.cameras
        ]
        for key in keys_to_remove:
            del item[key]

        return item

    def patch_metadata(self, config: TrainingConfig) -> None:
        """Patch dataset_to_policy_features to exclude filtered cameras."""
        if not config.cameras:
            return

        import lerobot.datasets.utils
        from lerobot.datasets.utils import dataset_to_policy_features

        original_func = dataset_to_policy_features
        selected_cameras = config.cameras
        transform_name = self.name

        def filtered_func(features: dict) -> dict:
            filtered = {}
            for key, value in features.items():
                if key.startswith("observation.images."):
                    cam_name = key.replace("observation.images.", "")
                    if cam_name in selected_cameras:
                        filtered[key] = value
                    else:
                        print(f"[{transform_name}] Excluding: {key}")
                else:
                    filtered[key] = value
            return original_func(filtered)

        lerobot.datasets.utils.dataset_to_policy_features = filtered_func


class TaskOverrideTransform(Transform):
    """Override the task field for all dataset items."""

    @property
    def name(self) -> str:
        return "task_override"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.task_override is not None

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if config.task_override:
            item["task"] = config.task_override
        return item


class DeltaActionTransform(Transform):
    """Convert absolute actions to delta actions (action - observation.state)."""

    @property
    def name(self) -> str:
        return "delta_actions"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.use_delta_actions

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if "action" in item and "observation.state" in item:
            item["action"] = item["action"] - item["observation.state"]
        return item


# =============================================================================
# Transform Runner
# =============================================================================


class TransformRunner:
    """
    Manages and applies dataset transforms.

    Handles:
    - Registration of transforms
    - Metadata patching (before lerobot import)
    - Dataset patching (after lerobot import)
    """

    # Registry of available transforms (add new transforms here)
    TRANSFORMS: list[Transform] = [
        CameraFilterTransform(),
        TaskOverrideTransform(),
        DeltaActionTransform(),
    ]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.active_transforms = [t for t in self.TRANSFORMS if t.is_enabled(config)]

    def log_config(self) -> None:
        """Log active transforms."""
        print("[lerobot_training] Active transforms:")
        if not self.active_transforms:
            print("  (none - pass-through mode)")
            return

        for transform in self.active_transforms:
            details = self._get_transform_details(transform)
            print(f"  - {transform.name}: {details}")

    def _get_transform_details(self, transform: Transform) -> str:
        """Get human-readable details for a transform."""
        if isinstance(transform, CameraFilterTransform):
            return str(self.config.cameras)
        elif isinstance(transform, TaskOverrideTransform):
            return f"'{self.config.task_override}'"
        elif isinstance(transform, DeltaActionTransform):
            return "action = action - observation.state"
        return "enabled"

    def apply_metadata_patches(self) -> None:
        """Apply metadata patches before importing lerobot training."""
        for transform in self.active_transforms:
            transform.patch_metadata(self.config)

    def apply_dataset_patches(self) -> None:
        """Patch LeRobotDataset.__getitem__ to apply transforms."""
        if not self.active_transforms:
            return

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        original_getitem = LeRobotDataset.__getitem__
        transforms = self.active_transforms
        config = self.config

        def patched_getitem(self, idx):
            item = original_getitem(self, idx)
            for transform in transforms:
                item = transform.apply(item, config)
            return item

        LeRobotDataset.__getitem__ = patched_getitem
        print(f"[lerobot_training] Patched LeRobotDataset with {len(transforms)} transform(s)")


# =============================================================================
# Training Functions
# =============================================================================


def train(config: TrainingConfig | None = None) -> None:
    """
    Run LeRobot training with custom transforms.

    Args:
        config: Training configuration. If None, parsed from env/args.
    """
    # Parse configuration if not provided
    if config is None:
        config = TrainingConfig.from_env_and_args()

    # Validate camera names if specified
    if config.cameras and config.dataset_root:
        try:
            invalid = config.validate_cameras()
            if invalid:
                print(f"[ERROR] Invalid camera names: {invalid}")
                sys.exit(1)
        except FileNotFoundError:
            print("[WARN] Could not validate cameras - dataset metadata not found")

    # Initialize transform runner
    runner = TransformRunner(config)
    runner.log_config()

    # Apply metadata patches BEFORE importing lerobot training
    # (required for camera filtering to affect policy input features)
    runner.apply_metadata_patches()

    # Import lerobot training module
    from lerobot.scripts.lerobot_train import train as lerobot_train

    # Apply dataset transforms
    runner.apply_dataset_patches()

    # Run training
    lerobot_train()


def main() -> None:
    """CLI entry point for lerobot-train."""
    train()


if __name__ == "__main__":
    main()
