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
from dataclasses import dataclass, field
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
    delta_exclude_joints: list[str] | None = None  # Joint names to keep in absolute space when use_delta_actions=True
    dataset_root: str | None = None
    output_dir: str | None = None
    resume_job_path: str | None = None
    split_ratio: list[float] = field(default_factory=lambda: [8.0, 1.0, 1.0])  # train/val/test episode split ratios
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

        # Joints to exclude from delta transform (kept in absolute space)
        delta_exclude_joints: list[str] | None = None
        for arg in sys.argv:
            if arg.startswith("--delta-exclude-joints="):
                raw = arg.split("=", 1)[1]
                delta_exclude_joints = [j.strip() for j in raw.split(",") if j.strip()]
                sys.argv.remove(arg)
                break

        # Episode split ratios from --split-ratio arg (remove to avoid lerobot arg parsing error)
        split_ratio = [8.0, 1.0, 1.0]  # default: train/val/test
        for arg in sys.argv:
            if arg.startswith("--split-ratio="):
                parts = [float(x) for x in arg.split("=", 1)[1].split(",")]
                if len(parts) == 2:
                    parts.append(0.0)  # no test set
                split_ratio = parts
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

        # Resume job from --resume-job=PATH arg
        resume_job_path = None
        for arg in sys.argv:
            if arg.startswith("--resume-job="):
                resume_job_path = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # If --resume or --output_dir is passed directly, warn and tell to use --resume-job
        if any(a.startswith("--resume") or a.startswith("--output_dir") for a in sys.argv):
            log.warning("[anvil_trainer] Manual --resume or --output_dir detected. Please use --resume-job=PATH instead for better compatibility.")

        is_resume = resume_job_path is not None
        
        if is_resume:
            # Inject lerobot resume flags
            if not any(a.startswith("--resume=") for a in sys.argv) and "--resume" not in sys.argv:
                sys.argv.append("--resume=true")
            if not any(a.startswith("--output_dir=") for a in sys.argv):
                sys.argv.append(f"--output_dir={resume_job_path}")
            
            # Extract output_dir for our internal config
            output_dir = resume_job_path
        else:
            # Resolve output_dir for NEW job: model_zoo/{dataset_name}/{run_name}
            # Extract job_name if provided (passed through to lerobot as-is)
            job_name = None
            for arg in sys.argv:
                if arg.startswith("--job_name="):
                    job_name = arg.split("=", 1)[1]
                    break

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

        # Backbone selection from --backbone= arg (for ACT/Diffusion; VLAs use their own vision)
        backbone = "resnet18"
        for arg in sys.argv:
            if arg.startswith("--backbone="):
                backbone = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # Defaults injection — skip if resuming to avoid draccus decoding errors
        if not is_resume:
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

            # If --policy.path is given (loading from checkpoint), lerobot rejects --policy.type.
            # Strip --policy.type from sys.argv; we've already captured the value for naming purposes.
            # Also skip backbone injection — the checkpoint already contains backbone config.
            has_policy_path = any(a.startswith("--policy.path=") for a in sys.argv)
            if has_policy_path:
                sys.argv = [a for a in sys.argv if not a.startswith("--policy.type=")]

            # Inject backbone settings for non-VLA policies (ACT, Diffusion).
            # Pi0.5 / SmolVLA use their own vision encoders and ignore these flags.
            _VLA_POLICIES = {"pi05", "smolvla", "pi0"}
            if policy_type not in _VLA_POLICIES and not has_policy_path:
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
                if policy_type == "diffusion":
                    if not any(a.startswith("--policy.use_group_norm=") for a in sys.argv):
                        sys.argv.append("--policy.use_group_norm=false")

        return cls(
            cameras=cameras,
            task_override=task_override,
            use_delta_actions=use_delta_actions,
            delta_exclude_joints=delta_exclude_joints,
            dataset_root=dataset_root,
            output_dir=output_dir,
            resume_job_path=resume_job_path,
            split_ratio=split_ratio,
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
            split_ratio=data.get("split_ratio", [8.0, 1.0, 1.0]),
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
    """Convert absolute actions to delta actions (action - observation.state).

    Joints listed in config.delta_exclude_joints are kept in absolute space
    (their delta is not applied). Joint names are resolved from the dataset's
    meta/info.json on first call and cached.
    """

    def __init__(self):
        self._exclude_indices: list[int] | None = None  # cached after first lookup

    @property
    def name(self) -> str:
        return "delta_actions"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.use_delta_actions

    def _resolve_exclude_indices(self, config: TrainingConfig) -> list[int]:
        """Resolve joint names → action tensor indices from dataset metadata (cached)."""
        if self._exclude_indices is not None:
            return self._exclude_indices

        if not config.delta_exclude_joints or not config.dataset_root:
            self._exclude_indices = []
            return self._exclude_indices

        info_path = Path(config.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[delta_actions] %s not found — no joints excluded", info_path)
            self._exclude_indices = []
            return self._exclude_indices

        with open(info_path) as f:
            info = json.load(f)

        names = info.get("features", {}).get("action", {}).get("names", [])
        # Flatten if nested list (e.g. [{"motor_names": [...]}])
        if names and isinstance(names[0], dict):
            names = [n for group in names for n in group.get("motor_names", [])]

        indices = []
        for joint in config.delta_exclude_joints:
            if joint in names:
                idx = names.index(joint)
                indices.append(idx)
                log.info("[delta_actions] Excluding joint '%s' (index %d) from delta", joint, idx)
            else:
                log.warning("[delta_actions] Joint '%s' not found in action names %s", joint, names)

        self._exclude_indices = indices
        return self._exclude_indices

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if "action" not in item or "observation.state" not in item:
            return item

        original_action = item["action"].clone()
        item["action"] = item["action"] - item["observation.state"]

        # Restore excluded joints to their original absolute values
        for idx in self._resolve_exclude_indices(config):
            item["action"][..., idx] = original_action[..., idx]

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

    # Registry of available transforms (add new transforms here).
    # Instantiated fresh per TransformRunner so stateful transforms (e.g. DeltaActionTransform
    # which caches joint indices) do not share state across runs.
    TRANSFORMS: list[Transform] = []  # populated in __init__

    def __init__(self, config: TrainingConfig):
        self.config = config
        transforms: list[Transform] = [
            CameraFilterTransform(),
            TaskOverrideTransform(),
            DeltaActionTransform(),
        ]
        self.active_transforms = [t for t in transforms if t.is_enabled(config)]
        self._val_dataloader = None   # set by apply_val_loss_patch when make_dataset is called
        self._test_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._split_info: dict = {}   # populated by patched_make_dataset
        self._preprocessor = None     # captured from make_pre_post_processors
        self._val_freq = 0            # set from cfg.log_freq * 5 inside patched_make_dataset
        self._resume_step = 0         # for absolute step tracking in wandb

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
        """Patch LeRobotDataset.__getitem__ to apply transforms and fix index mapping.

        This patch is always installed (even without active_transforms) because
        EpisodeAwareSampler yields absolute frame indices that must be remapped to
        relative indices for filtered (split) datasets. The mapping is only applied
        to the train dataset instance (flagged via _anvil_uses_abs_sampler).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # We must capture the original __getitem__ to use it in our patch.
        # LeRobotDataset.__getitem__ in v0.5.1 does not perform index mapping,
        # but EpisodeAwareSampler yields absolute indices. We add the mapping
        # logic here to support filtered datasets (splits).
        original_getitem = LeRobotDataset.__getitem__
        transforms = self.active_transforms
        config = self.config

        def patched_getitem(self, idx):
            # 1. Resolve relative index if the dataset is filtered by episodes.
            # Only the train dataset uses EpisodeAwareSampler (absolute indices).
            # Val/test datasets use DataLoader without a sampler (relative indices
            # 0..N-1) and must NOT be remapped — doing so would corrupt reads when
            # relative indices overlap with the absolute frame index space.
            reader = self._ensure_reader()
            if getattr(self, '_anvil_uses_abs_sampler', False) and reader._absolute_to_relative_idx is not None:
                # Map from absolute HF frame index to relative filtered index
                idx = reader._absolute_to_relative_idx.get(idx, idx)

            # 2. Call original __getitem__ (which calls reader.get_item)
            item = original_getitem(self, idx)

            # 3. Apply transforms (no-op when transforms list is empty)
            for transform in transforms:
                item = transform.apply(item, config)
            return item

        LeRobotDataset.__getitem__ = patched_getitem
        log.info("[anvil_trainer] Patched LeRobotDataset.__getitem__ (%d transform(s))", len(transforms))

    def apply_val_loss_patch(self) -> None:
        """Monkey-patch make_dataset to create train/val/test splits, and capture preprocessor."""
        s = self.config.split_ratio
        total_r = sum(s)
        if total_r <= 0 or (s[1] <= 0 and (len(s) < 3 or s[2] <= 0)):
            return  # no val or test, skip patching

        import lerobot.datasets.factory as factory_mod
        import lerobot.policies.factory as policy_factory_mod
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        from lerobot.datasets.factory import make_dataset as original_make_dataset
        import torch
        import random

        val_state = self
        _patched = {"done": False}

        def patched_make_dataset(cfg):
            # Only intercept the first call (main process dataset creation).
            if _patched["done"]:
                return original_make_dataset(cfg)
            _patched["done"] = True

            # Capture val_freq and resume_step from lerobot cfg
            val_state._val_freq = cfg.log_freq * 5 if cfg.log_freq > 0 else 0
            if cfg.resume and hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
                try:
                    step_file = Path(cfg.checkpoint_path) / "training_state" / "training_step.json"
                    if step_file.exists():
                        val_state._resume_step = json.loads(step_file.read_text()).get("step", 0)
                except Exception:
                    val_state._resume_step = 0

            # Full dataset to determine total episode count
            full_dataset = original_make_dataset(cfg)
            total_ep = full_dataset.num_episodes

            # Check if split_info.json already exists in last checkpoint (for resume)
            split_info_path = Path(cfg.output_dir) / "checkpoints" / "last" / "pretrained_model" / "split_info.json"
            if split_info_path.exists():
                try:
                    loaded_split = json.loads(split_info_path.read_text())
                    train_ep = loaded_split.get("train_episodes", [])
                    val_ep = loaded_split.get("val_episodes", [])
                    test_ep = loaded_split.get("test_episodes", [])
                    log.info("[split] Loaded random splits from %s", split_info_path)
                except Exception as e:
                    log.warning("[split] Failed to load %s: %s. Re-splitting...", split_info_path, e)
                    train_ep = val_ep = test_ep = None
            else:
                train_ep = val_ep = test_ep = None

            if train_ep is None:
                # Random three-way split
                n_test = max(1, round(total_ep * s[2] / total_r)) if len(s) > 2 and s[2] > 0 else 0
                n_val = max(1, round(total_ep * s[1] / total_r)) if s[1] > 0 else 0
                n_train = total_ep - n_val - n_test

                if n_train < 1:
                    log.warning("[split] Not enough episodes (%d) for split %s, using all for training", total_ep, s)
                    return full_dataset

                all_eps = list(range(total_ep))
                # Use a fixed seed for splitting to ensure consistency if re-run without split_info.json
                rng = random.Random(cfg.seed)
                rng.shuffle(all_eps)

                train_ep = sorted(all_eps[:n_train])
                val_ep = sorted(all_eps[n_train : n_train + n_val])
                test_ep = sorted(all_eps[n_train + n_val :])
                log.info("[split] Generated random splits")

            # Store split info for anvil_config.json (as full lists now)
            val_state._split_info = {
                "split_ratio": list(s),
                "total_episodes": total_ep,
                "train_episodes": train_ep,
                "val_episodes": val_ep,
                "test_episodes": test_ep,
            }

            def _make_dataloader(dataset):
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    prefetch_factor=2 if cfg.num_workers > 0 else None,
                )

            # Val dataloader
            if val_ep:
                cfg.dataset.episodes = val_ep
                val_dataset = original_make_dataset(cfg)
                val_state._val_dataloader = _make_dataloader(val_dataset)
                log.info("[split] val=%d ep (randomly selected, %d frames)", len(val_ep), val_dataset.num_frames)

            # Test dataloader
            if test_ep:
                cfg.dataset.episodes = test_ep
                test_dataset = original_make_dataset(cfg)
                val_state._test_dataloader = _make_dataloader(test_dataset)
                log.info("[split] test=%d ep (randomly selected, %d frames)", len(test_ep), test_dataset.num_frames)

            # Train dataset
            cfg.dataset.episodes = train_ep
            train_dataset = original_make_dataset(cfg)
            # Flag this instance so patched_getitem applies absolute→relative mapping.
            # EpisodeAwareSampler (used by ACT and similar policies) yields absolute
            # frame indices; val/test dataloaders use relative indices and must NOT
            # be remapped.
            train_dataset._anvil_uses_abs_sampler = True
            log.info("[split] train=%d ep (randomly selected)", len(train_ep))
            return train_dataset

        factory_mod.make_dataset = patched_make_dataset
        lerobot_train_mod.make_dataset = patched_make_dataset
        log.info("[split] Patched make_dataset (split_ratio=%s, random=True)", s)

        # Capture preprocessor when it's created by lerobot
        original_make_processors = policy_factory_mod.make_pre_post_processors

        def capturing_make_processors(*args, **kwargs):
            preprocessor, postprocessor = original_make_processors(*args, **kwargs)
            val_state._preprocessor = preprocessor
            return preprocessor, postprocessor

        policy_factory_mod.make_pre_post_processors = capturing_make_processors
        lerobot_train_mod.make_pre_post_processors = capturing_make_processors
        log.info("[split] Patched make_pre_post_processors to capture preprocessor")


    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log test loss (if test split is active) at save_freq.
        2. Write anvil_config.json (with split info) into each checkpoint's pretrained_model/ directory.
        """
        import time

        import torch
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg_base: dict = {"use_delta_actions": self.config.use_delta_actions}
        if self.config.delta_exclude_joints:
            anvil_cfg_base["delta_exclude_joints"] = self.config.delta_exclude_joints
        if self.config.task_override:
            anvil_cfg_base["task_description"] = self.config.task_override

        val_state = self

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            # --- Test loss (computed at save_freq) ---
            if val_state._test_dataloader is not None:
                policy = kwargs.get("policy")
                preprocessor = kwargs.get("preprocessor") or val_state._preprocessor
                step = kwargs.get("step", "?")

                if policy is not None:
                    policy.eval()
                    t0 = time.perf_counter()
                    total_loss = 0.0
                    n_batches = 0

                    # ACTPolicy in evaluation mode has no VAE, but test_loss
                    # needs to calculate the full loss. We set back to train mode
                    # to get the VAE loss if needed.
                    is_act = "ACTPolicy" in str(type(policy))
                    if is_act:
                        policy.train()

                    with torch.no_grad():
                        for batch in val_state._test_dataloader:
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

                    if is_act:
                        policy.eval()

                    test_loss = total_loss / max(n_batches, 1)
                    test_s = time.perf_counter() - t0
                    log.info("[eval] test_loss=%.6f @ step %s (%.1fs)", test_loss, step, test_s)

                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            _wandb.log({"eval/test_loss": test_loss}, step=int(step))
                    except Exception:
                        pass

            # --- Original save ---
            original_save_checkpoint(checkpoint_dir, **kwargs)

            # --- Save split_info.json and anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                # 1. anvil_config.json: only non-split flags
                (pretrained_dir / "anvil_config.json").write_text(json.dumps(anvil_cfg_base, indent=2))
                
                # 2. split_info.json: all split metadata
                if val_state._split_info:
                    (pretrained_dir / "split_info.json").write_text(json.dumps(val_state._split_info, indent=2))
                
                log.info("[anvil_trainer] Saved configs to %s", pretrained_dir)

        # Patch both the module and the already-imported reference in lerobot_train
        train_utils_mod.save_checkpoint = patched_save_checkpoint
        lerobot_train_mod.save_checkpoint = patched_save_checkpoint
        log.info("[anvil_trainer] Patched save_checkpoint for test loss + anvil_config.json")

    def apply_val_loss_hook(self) -> None:
        """Monkey-patch update_policy for periodic val loss computation at val_freq intervals."""
        import time

        import torch
        import lerobot.scripts.lerobot_train as lerobot_train_mod

        original_update_policy = lerobot_train_mod.update_policy
        val_state = self
        _counter = {"n": 0}

        def patched_update_policy(
            train_metrics, policy, batch, optimizer, grad_clip_norm,
            accelerator=None, lr_scheduler=None, lock=None, rabc_weights_provider=None,
        ):
            result = original_update_policy(
                train_metrics, policy, batch, optimizer, grad_clip_norm,
                accelerator=accelerator, lr_scheduler=lr_scheduler,
                lock=lock, rabc_weights_provider=rabc_weights_provider,
            )

            _counter["n"] += 1
            val_freq = val_state._val_freq
            if not val_freq or val_freq <= 0 or val_state._val_dataloader is None:
                return result
            if _counter["n"] % val_freq != 0:
                return result

            abs_step = val_state._resume_step + _counter["n"]
            preprocessor = val_state._preprocessor

            # Unwrap accelerator-wrapped policy for eval
            if accelerator is not None:
                unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            else:
                unwrapped = policy

            unwrapped.eval()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_batches = 0

            # ACTPolicy in evaluation mode has no VAE, but val_loss
            # needs to calculate the full loss.
            is_act = "ACTPolicy" in str(type(unwrapped))
            if is_act:
                unwrapped.train()

            with torch.no_grad():
                for val_batch in val_state._val_dataloader:
                    if preprocessor is not None:
                        val_batch = preprocessor(val_batch)
                    else:
                        device = next(unwrapped.parameters()).device
                        val_batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()
                        }
                    loss, _ = unwrapped.forward(val_batch)
                    total_loss += loss.item()
                    n_batches += 1

            if is_act:
                unwrapped.eval()

            val_loss = total_loss / max(n_batches, 1)
            val_s = time.perf_counter() - t0
            log.info("[eval] val_loss=%.6f @ step %s (%.1fs)", val_loss, abs_step, val_s)

            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({"eval/val_loss": val_loss}, step=abs_step)
            except Exception:
                pass

            return result

        lerobot_train_mod.update_policy = patched_update_policy
        log.info("[eval] Patched update_policy for periodic val loss (val_freq will be log_freq*5)")


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

    # Patch make_dataset for train/val/test split + capture preprocessor
    runner.apply_val_loss_patch()

    # Patch save_checkpoint for test loss + anvil_config.json
    runner.apply_checkpoint_patch()

    # Patch update_policy for periodic val loss
    runner.apply_val_loss_hook()

    # Run training
    if config.resume_job_path:
        # LeRobot 0.5.1 saves train_config.json inside each checkpoint
        last_cfg_path = Path(config.resume_job_path) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if last_cfg_path.exists() and not any(a.startswith("--config_path=") for a in sys.argv):
            sys.argv.append(f"--config_path={last_cfg_path}")
            log.info("[anvil_trainer] Resuming with config from last checkpoint: %s", last_cfg_path)

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
  anvil-trainer --resume-job=model_zoo/my-dataset/grabbing-w1

===============================================================================

Anvil-specific flags (stripped before passing to LeRobot):

  --use-delta-actions
      Convert actions to delta form (action - observation.state).
      Persisted to anvil_config.json in each checkpoint so inference
      can read it automatically.

  --resume-job=PATH
      Resume a previously stopped training job.
      Shortcut for --resume=true --output_dir=PATH.

  --task-description=TEXT
      Task prompt for SmolVLA. Overrides LEROBOT_TASK_OVERRIDE env var.
      Example: --task-description="Grab the gray doll and put it in the bucket"

  --camera-filter=CAM1,CAM2,...
      Train using only the specified cameras. Overrides LEROBOT_CAMERA_FILTER.
      Example: --camera-filter=chest,waist

  --split-ratio=TRAIN,VAL,TEST
      Episode split ratios for train/val/test (default: 8,1,1).
      Two values also accepted: --split-ratio=8,2 means [8,2,0] (no test set).
      Val loss logged every log_freq*5 steps (eval/val_loss).
      Test loss logged every save_freq steps (eval/test_loss).
      Set to --split-ratio=1,0,0 to disable held-out sets.

  --job_name=NAME
      Human-readable run name. Checkpoints saved to model_zoo/<name>/.
      Auto-generated from policy type + timestamp if omitted.

  --output_dir=PATH
      Override the default checkpoint directory (default: model_zoo/).
      For resuming, use --resume-job instead.

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
