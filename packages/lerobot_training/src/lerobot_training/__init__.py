"""
LeRobot Training Package

Custom LeRobot training utilities with pluggable transforms:
- Camera filtering: Train with a subset of cameras
- Task override: Override dataset task for SmolVLA
- Delta actions: Convert actions to relative (action - observation.state)

Usage:
    from lerobot_training import train, TrainingConfig, TransformRunner

    # Or use CLI:
    # lerobot-train --dataset.repo_id=local --dataset.root=/path/to/dataset
"""

from lerobot_training.train import (
    CameraFilterTransform,
    DeltaActionTransform,
    TaskOverrideTransform,
    TrainingConfig,
    Transform,
    TransformRunner,
    main,
    train,
)

__version__ = "0.1.0"

__all__ = [
    "TrainingConfig",
    "Transform",
    "CameraFilterTransform",
    "TaskOverrideTransform",
    "DeltaActionTransform",
    "TransformRunner",
    "train",
    "main",
]
