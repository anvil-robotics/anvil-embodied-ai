"""
Anvil Trainer Package

Training utilities for Anvil robotics workflows, supporting lerobot and other platforms.
Provides pluggable transforms for dataset preprocessing:
- Observation exclude: Drop cameras or non-image observations by suffix
- Task override: Override dataset task for SmolVLA
- EE relative: Convert EE absolute actions to SE(3) relative (action_type=ee_relative;
  "ee_rel" is a permanent legacy alias for existing checkpoints)

Usage:
    from anvil_trainer import train, TrainingConfig, TransformRunner

    # Or use CLI:
    # anvil-trainer --dataset.root=/path/to/dataset --action-type=joint_abs
"""

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner
from anvil_trainer.train import main, train
from anvil_trainer.transforms import (
    EERelativeTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

__version__ = "0.1.0"

__all__ = [
    "TrainingConfig",
    "Transform",
    "ExcludeObservationTransform",
    "TaskOverrideTransform",
    "EERelativeTransform",
    "TransformRunner",
    "train",
    "main",
]
