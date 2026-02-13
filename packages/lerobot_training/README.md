# LeRobot Training Package

Custom LeRobot training utilities with pluggable transforms for Anvil robotics workflows.

## Features

- **Camera filtering**: Train with a subset of available cameras
- **Task override**: Override dataset task for SmolVLA training
- **Delta actions**: Convert actions to relative (action - observation.state)

## Installation

```bash
# From repository root (installs all workspace packages)
uv sync --all-packages
```

## Usage

### Training

```bash
# Basic training with local dataset
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=act

# Train with camera filtering
LEROBOT_CAMERA_FILTER=chest,waist,wrist_l lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=act

# Train with delta actions
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=act \
    --use-delta-actions

# Train SmolVLA with language instruction
LEROBOT_TASK_OVERRIDE="Pick up the red cube" lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=smolvla
```

### Python API

```python
from lerobot_training import train, TrainingConfig

config = TrainingConfig(
    cameras=["chest", "waist", "wrist_l"],
    task_override="Pick up the object",
    use_delta_actions=True,
)
train(config)
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LEROBOT_CAMERA_FILTER` | Comma-separated camera names to include |
| `LEROBOT_TASK_OVERRIDE` | Override task string for all samples |

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--use-delta-actions` | Convert actions to delta (action - state) |

## Adding Custom Transforms

1. Create a new `Transform` subclass in `train.py`
2. Add configuration field to `TrainingConfig`
3. Register in `TransformRunner.TRANSFORMS`

```python
class MyTransform(Transform):
    @property
    def name(self) -> str:
        return "my_transform"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.my_option is not None

    def apply(self, item: dict, config: TrainingConfig) -> dict:
        # Modify item
        return item
```
