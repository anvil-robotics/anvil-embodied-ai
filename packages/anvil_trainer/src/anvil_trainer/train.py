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
    from anvil_trainer import train, TrainingConfig
    config = TrainingConfig(cameras=["chest", "waist"])
    train(config)

Environment variables:
    LEROBOT_CAMERA_FILTER: Comma-separated list of cameras to include
    LEROBOT_TASK_OVERRIDE: Override task string for all samples
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)
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
    val_split_ratio: float = 0.2  # Fraction of episodes held out for validation loss (0.0 = disabled)
    # Vision backbone for ACT/Diffusion: resnet18 | resnet34 | resnet50 (VLA models ignore this)
    backbone: str = "resnet18"

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

        # Validation split ratio from --val-split-ratio arg (remove to avoid lerobot arg parsing error)
        val_split_ratio = 0.2  # default
        for arg in sys.argv:
            if arg.startswith("--val-split-ratio="):
                val_split_ratio = float(arg.split("=", 1)[1])
                sys.argv.remove(arg)
                break

        # Extract dataset_root and policy_type early (needed for naming and backbone injection)
        dataset_root = None
        for arg in sys.argv:
            if arg.startswith("--dataset.root="):
                dataset_root = arg.split("=", 1)[1]
                break
        dataset_name = Path(dataset_root).name if dataset_root else "dataset"

        policy_type = "run"
        for arg in sys.argv:
            if arg.startswith("--policy.type="):
                policy_type = arg.split("=", 1)[1]
                break

        # Backbone selection from --backbone= arg (for ACT/Diffusion; VLAs use their own vision)
        backbone = "resnet18"
        for arg in sys.argv:
            if arg.startswith("--backbone="):
                backbone = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # Default push_to_hub=false unless explicitly set
        if not any(arg.startswith("--policy.push_to_hub") for arg in sys.argv):
            sys.argv.append("--policy.push_to_hub=false")

        # Default dataset.repo_id=local for local dataset training
        if not any(arg.startswith("--dataset.repo_id") for arg in sys.argv):
            sys.argv.append("--dataset.repo_id=local")

        # Disable eval by default — no gym env available for Anvil datasets
        if not any(arg.startswith("--eval_freq") for arg in sys.argv):
            sys.argv.append("--eval_freq=0")

        # Default total training steps
        if not any(arg.startswith("--steps") for arg in sys.argv):
            sys.argv.append("--steps=100000")

        # Default checkpoint save frequency
        if not any(arg.startswith("--save_freq") for arg in sys.argv):
            sys.argv.append("--save_freq=10000")

        # Inject backbone settings for non-VLA policies (ACT, Diffusion).
        # Pi0.5 / SmolVLA use their own vision encoders and ignore these flags.
        _VLA_POLICIES = {"pi05", "smolvla", "pi0"}
        if policy_type not in _VLA_POLICIES:
            _BACKBONE_MAP = {
                "resnet18": ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
                "resnet34": ("resnet34", "ResNet34_Weights.IMAGENET1K_V1"),
                "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V1"),
            }
            _vb, _pw = _BACKBONE_MAP.get(backbone, ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"))
            if not any(a.startswith("--policy.vision_backbone=") for a in sys.argv):
                sys.argv.append(f"--policy.vision_backbone={_vb}")
            if not any(a.startswith("--policy.pretrained_backbone_weights=") for a in sys.argv):
                sys.argv.append(f"--policy.pretrained_backbone_weights={_pw}")
            # Diffusion's use_group_norm=True replaces BatchNorm with GroupNorm, which is
            # incompatible with pretrained ImageNet weights. Disable it so the pretrained
            # BatchNorm statistics are preserved. GroupNorm is only needed for very small batches.
            if policy_type == "diffusion":
                if not any(a.startswith("--policy.use_group_norm=") for a in sys.argv):
                    sys.argv.append("--policy.use_group_norm=false")

        # Extract job_name if provided (passed through to lerobot as-is)
        job_name = None
        for arg in sys.argv:
            if arg.startswith("--job_name="):
                job_name = arg.split("=", 1)[1]
                break

        # Resolve output_dir: model_zoo/{dataset_name}/{run_name}
        #   run_name = job_name if provided, else {policy_type}_{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = job_name if job_name else f"{policy_type}_{timestamp}"

        output_dir = None
        for arg in sys.argv:
            if arg.startswith("--output_dir="):
                output_dir = arg.split("=", 1)[1]
                break
        if output_dir is None:
            output_dir = f"model_zoo/{dataset_name}/{run_name}"
            sys.argv.append(f"--output_dir={output_dir}")

        # Auto-inject job_name if not provided (used as wandb run name)
        if not job_name:
            sys.argv.append(f"--job_name={run_name}")

        # Set wandb project = dataset_name (not hardcoded "anvil")
        if not any(a.startswith("--wandb.project=") for a in sys.argv):
            sys.argv.append(f"--wandb.project={dataset_name}")

        return cls(
            cameras=cameras,
            task_override=task_override,
            use_delta_actions=use_delta_actions,
            dataset_root=dataset_root,
            output_dir=output_dir,
            val_split_ratio=val_split_ratio,
            backbone=backbone,
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
            backbone=data.get("backbone", "resnet18"),
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
                        log.info("[camera_filter] Excluding: %s", key)
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
        self._val_dataloader = None  # set by apply_val_loss_patch when make_dataset is called

    def log_config(self) -> None:
        """Log active transforms."""
        if not self.active_transforms:
            log.info("[anvil_trainer] Active transforms: (none - pass-through mode)")
            return

        for transform in self.active_transforms:
            details = self._get_transform_details(transform)
            log.info("[anvil_trainer] Active transform: %s — %s", transform.name, details)

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
        log.info("[anvil_trainer] Patched LeRobotDataset with %d transform(s)", len(transforms))

    def apply_val_loss_patch(self) -> None:
        """Monkey-patch make_dataset to create a val split, then compute val loss at each checkpoint."""
        if self.config.val_split_ratio <= 0.0:
            return

        import lerobot.datasets.factory as factory_mod
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        from lerobot.datasets.factory import make_dataset as original_make_dataset
        import torch

        val_split_ratio = self.config.val_split_ratio
        val_state = self
        _patched = {"done": False}

        def patched_make_dataset(cfg):
            # Only intercept the first call (main process dataset creation).
            # Subsequent calls (non-main process or internal re-calls) pass through.
            if _patched["done"]:
                return original_make_dataset(cfg)
            _patched["done"] = True

            # Full dataset to determine total episode count
            full_dataset = original_make_dataset(cfg)
            total_ep = full_dataset.num_episodes
            n_val = max(1, round(total_ep * val_split_ratio))
            train_ep = list(range(0, total_ep - n_val))
            val_ep = list(range(total_ep - n_val, total_ep))

            # Val dataset
            cfg.dataset.episodes = val_ep
            val_dataset = original_make_dataset(cfg)

            # Train dataset — leave cfg.dataset.episodes = train_ep so the rest of
            # the training pipeline (sampler, logging) sees only the train set.
            cfg.dataset.episodes = train_ep
            train_dataset = original_make_dataset(cfg)

            # Val dataloader — sequential, no episode-aware sampler.
            # EpisodeAwareSampler uses absolute frame indices from the full dataset, which
            # are out-of-bounds for the subset val dataset. For loss computation we just
            # iterate all val frames in order; drop_n_last_frames is a training-only concern.
            val_state._val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=2 if cfg.num_workers > 0 else None,
            )

            log.info(
                "[val_loss] Val split: train=%d ep, val=%d ep (idx %d–%d, %d frames)",
                len(train_ep), len(val_ep), val_ep[0], val_ep[-1], val_dataset.num_frames,
            )
            return train_dataset

        factory_mod.make_dataset = patched_make_dataset
        lerobot_train_mod.make_dataset = patched_make_dataset
        log.info("[val_loss] Patched make_dataset (val_split_ratio=%.2f)", val_split_ratio)

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log validation loss (if val split is active).
        2. Write anvil_config.json into each checkpoint's pretrained_model/ directory.
        """
        import time

        import torch
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg: dict = {"use_delta_actions": self.config.use_delta_actions}
        if self.config.task_override:
            anvil_cfg["task_description"] = self.config.task_override
        anvil_cfg_content = json.dumps(anvil_cfg, indent=2)

        val_state = self  # captures self._val_dataloader set by apply_val_loss_patch

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            # --- Validation loss ---
            if val_state._val_dataloader is not None:
                policy = kwargs.get("policy")
                preprocessor = kwargs.get("preprocessor")
                step = kwargs.get("step", "?")

                if policy is not None:
                    t0 = time.perf_counter()
                    total_loss = 0.0
                    n_batches = 0

                    with torch.no_grad():
                        for batch in val_state._val_dataloader:
                            if preprocessor is not None:
                                batch = preprocessor(batch)
                            else:
                                device = next(policy.parameters()).device
                                batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                            loss, _ = policy.forward(batch)
                            total_loss += loss.item()
                            n_batches += 1

                    val_loss = total_loss / max(n_batches, 1)
                    val_s = time.perf_counter() - t0
                    log.info("[val_loss] Step %s: loss=%.4f (%.1fs)", step, val_loss, val_s)

            # --- Original save ---
            original_save_checkpoint(checkpoint_dir, **kwargs)

            # --- Write anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                (pretrained_dir / "anvil_config.json").write_text(anvil_cfg_content)
                log.info("[anvil_trainer] Saved anvil_config.json to %s", pretrained_dir)

        # Patch both the module and the already-imported reference in lerobot_train
        train_utils_mod.save_checkpoint = patched_save_checkpoint
        lerobot_train_mod.save_checkpoint = patched_save_checkpoint
        log.info("[anvil_trainer] Patched save_checkpoint to write anvil_config.json per checkpoint")


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
                log.error("[anvil_trainer] Invalid camera names: %s", invalid)
                sys.exit(1)
        except FileNotFoundError:
            log.warning("[anvil_trainer] Could not validate cameras - dataset metadata not found")

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

    # Patch make_dataset to create val split (must be before apply_checkpoint_patch)
    runner.apply_val_loss_patch()

    # Monkey-patch save_checkpoint to compute val loss + write anvil_config.json
    runner.apply_checkpoint_patch()

    # Run training
    lerobot_train()


_ANVIL_HELP = """\
anvil-trainer — LeRobot training with Anvil customizations
===========================================================

Examples:

  # Train ACT (basic)
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
    --policy.type=act --job_name=grabbing-w1

  # Train SmolVLA with task description
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
    --policy.type=smolvla --job_name=grabbing-w1 \\
    --task-description="Grab the gray doll and put it in the bucket"

  # Train with delta actions and camera subset
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
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

  --val-split-ratio=FLOAT
      Fraction of episodes (last N%) held out for validation loss (default: 0.2).
      Val loss is computed at every --save_freq step and logged to console + wandb.
      Set to 0 to disable: --val-split-ratio=0

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
