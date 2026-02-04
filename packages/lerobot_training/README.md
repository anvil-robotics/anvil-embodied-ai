# LeRobot Training Package

Custom LeRobot training utilities with pluggable transforms for Anvil robotics workflows.

## Features

- **Camera filtering**: Train with a subset of available cameras
- **Task override**: Override dataset task for SmolVLA training
- **Delta actions**: Convert actions to relative (action - observation.state)
- **Video utilities**: Convert MCAP recordings to MP4, preview AV1 videos

## Installation

```bash
# From the anvil-embodied-ai repository root
uv pip install -e packages/lerobot_training

# With video utilities
uv pip install -e "packages/lerobot_training[video]"
```

## Usage

### Training

```bash
# Basic training with local dataset
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=act \
    --output_dir=outputs/my_model

# Train with camera filtering (via environment variable)
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

# Train SmolVLA with language instruction override
LEROBOT_TASK_OVERRIDE="Pick up the red cube" lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/path/to/dataset \
    --policy.type=smolvla
```

### Python API

```python
from lerobot_training import train, TrainingConfig

# Configure training
config = TrainingConfig(
    cameras=["chest", "waist", "wrist_l"],
    task_override="Pick up the object",
    use_delta_actions=True,
    dataset_root="/path/to/dataset",
)

# Validate configuration
invalid_cameras = config.validate_cameras()
if invalid_cameras:
    print(f"Invalid cameras: {invalid_cameras}")

# Run training
train(config)
```

### Video Utilities

```bash
# Convert MCAP to MP4
mcap2mp4 -i recording.mcap -o ./videos

# Scan MCAP topics only
mcap2mp4 -i recording.mcap --scan-only

# Convert AV1 videos to H.264 for preview
preview-videos -i ./dataset -o ./preview
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

## License

Apache-2.0
