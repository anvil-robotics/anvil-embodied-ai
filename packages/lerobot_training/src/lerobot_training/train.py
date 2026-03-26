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
    anvil-trainer [lerobot args] [--use-delta-actions] [--task-description="..."] [--camera-filter=chest,waist]

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
from datetime import datetime
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
    output_dir: str | None = None

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
        # Camera filter from --camera-filter arg (takes precedence over env var)
        camera_str = None
        for arg in sys.argv:
            if arg.startswith("--camera-filter="):
                camera_str = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break
        # Fall back to environment variable
        if camera_str is None:
            camera_str = os.environ.get("LEROBOT_CAMERA_FILTER", "")
        cameras = [c.strip() for c in camera_str.split(",") if c.strip()] or None

        # Task description from --task-description arg (takes precedence over env var)
        task_override = None
        for arg in sys.argv:
            if arg.startswith("--task-description="):
                task_override = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break
        # Fall back to environment variable
        if task_override is None:
            task_override = os.environ.get("LEROBOT_TASK_OVERRIDE", "") or None

        # Delta actions from command line (remove to avoid lerobot arg parsing error)
        use_delta_actions = "--use-delta-actions" in sys.argv
        if use_delta_actions:
            sys.argv.remove("--use-delta-actions")

        # Default push_to_hub=false unless explicitly set
        if not any(arg.startswith("--policy.push_to_hub") for arg in sys.argv):
            sys.argv.append("--policy.push_to_hub=false")

        # Try to extract dataset root from args for validation
        dataset_root = None
        for arg in sys.argv:
            if arg.startswith("--dataset.root="):
                dataset_root = arg.split("=", 1)[1]
                break

        # Extract job_name if provided (passed through to lerobot as-is)
        job_name = None
        for arg in sys.argv:
            if arg.startswith("--job_name="):
                job_name = arg.split("=", 1)[1]
                break

        # Resolve output_dir:
        #   explicit --output_dir  → use as-is
        #   --job_name             → model_zoo/<job_name>
        #   neither                → model_zoo/<policy_type>_<YYYYMMDD_HHMMSS>
        output_dir = None
        for arg in sys.argv:
            if arg.startswith("--output_dir="):
                output_dir = arg.split("=", 1)[1]
                break
        if output_dir is None:
            if job_name:
                output_dir = f"model_zoo/{job_name}"
            else:
                dataset_name = Path(dataset_root).name if dataset_root else "dataset"
                policy_type = "run"
                for arg in sys.argv:
                    if arg.startswith("--policy.type="):
                        policy_type = arg.split("=", 1)[1]
                        break
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"model_zoo/{dataset_name}_{policy_type}_{timestamp}"
            sys.argv.append(f"--output_dir={output_dir}")

        return cls(
            cameras=cameras,
            task_override=task_override,
            use_delta_actions=use_delta_actions,
            dataset_root=dataset_root,
            output_dir=output_dir,
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


class GrootMetaDevicePatch(Transform):
    """Patch GR00TN15.from_pretrained to disable meta device initialization.

    lerobot v0.5.0 bug: transformers initializes GROOT on meta device, causing
    FlowmatchingActionHead to call Beta(...).item() which fails on meta tensors.
    Fix: force device_map=None to skip meta device init entirely.
    """

    @property
    def name(self) -> str:
        return "groot_meta_device_patch"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return "--policy.type=groot" in sys.argv

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        return item

    def patch_metadata(self, config: TrainingConfig) -> None:
        from lerobot.policies.groot import groot_n1

        _orig = groot_n1.GR00TN15.from_pretrained.__func__

        @classmethod  # type: ignore[misc]
        def _patched(cls, *args, **kwargs):
            kwargs.setdefault("device_map", None)
            return _orig(cls, *args, **kwargs)

        groot_n1.GR00TN15.from_pretrained = _patched
        print(f"[{self.name}] Patched GR00TN15.from_pretrained (device_map=None)")


class XVLAConfigPatch(Transform):
    """Patch XVLAConfig.get_florence_config to inject default empty sub-configs.

    lerobot v0.5.0 bug: XVLAConfig.florence_config defaults to {}, but
    get_florence_config() raises ValueError if vision_config is absent.
    Fix: setdefault vision_config and text_config to {} before delegating.
    """

    @property
    def name(self) -> str:
        return "xvla_config_patch"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return "--policy.type=xvla" in sys.argv

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        return item

    def patch_metadata(self, config: TrainingConfig) -> None:
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig

        _orig = XVLAConfig.get_florence_config

        def _patched(self):
            self.florence_config.setdefault("vision_config", {})
            self.florence_config.setdefault("text_config", {})
            return _orig(self)

        XVLAConfig.get_florence_config = _patched
        print(f"[{self.name}] Patched XVLAConfig.get_florence_config (default sub-configs)")


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
        GrootMetaDevicePatch(),
        XVLAConfigPatch(),
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
        elif isinstance(transform, GrootMetaDevicePatch):
            return "device_map=None"
        elif isinstance(transform, XVLAConfigPatch):
            return "inject default florence sub-configs"
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

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to write anvil_config.json at each checkpoint save."""
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg: dict = {"use_delta_actions": self.config.use_delta_actions}
        if self.config.task_override:
            anvil_cfg["task_description"] = self.config.task_override
        anvil_cfg_content = json.dumps(anvil_cfg, indent=2)

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            original_save_checkpoint(checkpoint_dir, **kwargs)
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                (pretrained_dir / "anvil_config.json").write_text(anvil_cfg_content)
                print(f"[lerobot_training] Saved anvil_config.json to {pretrained_dir}")

        # Patch both the module and the already-imported reference in lerobot_train
        train_utils_mod.save_checkpoint = patched_save_checkpoint
        lerobot_train_mod.save_checkpoint = patched_save_checkpoint
        print("[lerobot_training] Patched save_checkpoint to write anvil_config.json per checkpoint")


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

    # Monkey-patch save_checkpoint to write anvil_config.json at each checkpoint save
    runner.apply_checkpoint_patch()

    # Run training
    lerobot_train()


_ANVIL_HELP = """\
anvil-trainer — LeRobot training with Anvil customizations
===========================================================

Examples:

  # Train ACT (basic)
  anvil-trainer --dataset.repo_id=local --dataset.root=data/datasets/my-dataset \\
    --policy.type=act --job_name=grabbing-w1

  # Train SmolVLA with task description
  anvil-trainer --dataset.repo_id=local --dataset.root=data/datasets/my-dataset \\
    --policy.type=smolvla --job_name=grabbing-w1 \\
    --task-description="Grab the gray doll and put it in the bucket"

  # Train with delta actions and camera subset
  anvil-trainer --dataset.repo_id=local --dataset.root=data/datasets/my-dataset \\
    --policy.type=act --job_name=grabbing-w1 \\
    --camera-filter=chest,waist --use-delta-actions

  # Resume a stopped run
  anvil-trainer --resume=true --output_dir=model_zoo/grabbing-w1

===============================================================================

Anvil-specific flags (stripped before passing to LeRobot):

  --use-delta-actions
      Convert actions to delta form (action - observation.state).
      Persisted to anvil_config.json in each checkpoint so inference
      can read it automatically.

  --task-description=TEXT
      Task prompt for SmolVLA. Overrides LEROBOT_TASK_OVERRIDE env var.
      Example: --task-description="Grab the gray doll and put it in the bucket"

  --camera-filter=CAM1,CAM2,...
      Train using only the specified cameras. Overrides LEROBOT_CAMERA_FILTER.
      Example: --camera-filter=chest,waist

  --job_name=NAME
      Human-readable run name. Checkpoints saved to model_zoo/<name>/.
      Auto-generated from policy type + timestamp if omitted.

  --output_dir=PATH
      Override the default checkpoint directory (default: model_zoo/).

Output:
  Checkpoints  →  model_zoo/<job_name>/checkpoints/<step>/pretrained_model/
  anvil_config →  model_zoo/<job_name>/checkpoints/<step>/pretrained_model/anvil_config.json

===============================================================================
LeRobot flags (passed through):
===============================================================================
"""


def _capture_lerobot_help() -> str:
    """Capture lerobot's help output without exiting."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    buf = io.StringIO()
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "--help"]
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            from lerobot.scripts.lerobot_train import train as lerobot_train
            lerobot_train()
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv
    return buf.getvalue()


def main() -> None:
    """CLI entry point for anvil-trainer."""
    import pydoc

    if "-h" in sys.argv or "--help" in sys.argv:
        lerobot_help = _capture_lerobot_help()
        pydoc.pager(_ANVIL_HELP + lerobot_help)
        sys.exit(0)
    train()


if __name__ == "__main__":
    main()
