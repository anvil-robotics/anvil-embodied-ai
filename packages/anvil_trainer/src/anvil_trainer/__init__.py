"""
Anvil Trainer Package

Training utilities for Anvil robotics workflows, supporting lerobot and other platforms.
Provides pluggable transforms for dataset preprocessing:
- Observation exclude: Drop cameras or non-image observations by suffix
- Task override: Override dataset task for SmolVLA
- Delta actions: Convert actions to relative (action - observation.state)

Usage:
    from anvil_trainer import train, TrainingConfig, TransformRunner

    # Or use CLI:
    # anvil-trainer --dataset.repo_id=local --dataset.root=/path/to/dataset
"""

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner
from anvil_trainer.train import main, train
from anvil_trainer.transforms import (
    DeltaActionTransform,
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
    "DeltaActionTransform",
    "TransformRunner",
    "train",
    "main",
]
