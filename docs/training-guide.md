# Training Guide

This guide covers the complete workflow for training imitation learning models using the Anvil-Embodied-AI platform.

## Overview

The training pipeline consists of three stages:

1. **Convert**: Transform MCAP recordings into LeRobot dataset format
2. **Train**: Train a policy using the LeRobot framework
3. **Upload**: Optionally upload models to HuggingFace Hub

## Converting MCAP to LeRobot Format

### The mcap-convert Tool

The `mcap-convert` tool transforms raw MCAP recordings into LeRobot v3.0 datasets:

```bash
uv run mcap-convert \
  --input data/raw/my-session \
  --output data/datasets/my-dataset \
  --config configs/mcap_converter/openarm_bimanual.yaml
```

### Command Options

| Option | Description |
|--------|-------------|
| `--input` | Directory containing MCAP files (one per episode) |
| `--output` | Output directory for LeRobot dataset |
| `--config` | Robot configuration YAML file |
| `--task` | Task description for the dataset |
| `--fps` | Target frame rate (default: 30) |

### Conversion Configuration

Create a YAML configuration file for your robot setup:

```yaml
# configs/mcap_converter/my_robot.yaml

robot_name: my_robot
fps: 30

# Camera configuration
cameras:
  cam_wrist:
    topic: /camera/wrist/image_raw
    resolution: [640, 480]
  cam_overhead:
    topic: /camera/overhead/image_raw
    resolution: [640, 480]

# Joint state configuration
joint_state:
  topic: /joint_states
  joints:
    - joint_1
    - joint_2
    - joint_3
    - joint_4
    - joint_5
    - joint_6
    - gripper

# Action mapping (which joints are controlled)
action_joints:
  - joint_1
  - joint_2
  - joint_3
  - joint_4
  - joint_5
  - joint_6
  - gripper
```

### LeRobot Dataset Structure

After conversion, the dataset has this structure:

```
data/datasets/my-dataset/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── episodes.jsonl     # Episode information
│   └── tasks.jsonl        # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.cam_wrist/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ...
```

### Validating the Dataset

Verify dataset integrity before training:

```bash
uv run mcap-validate --root data/datasets/my-dataset
```

## Training Configuration

### Training Config Directory

Training configurations are stored in `configs/lerobot_training/`. Create a YAML file for your training run:

```yaml
# configs/lerobot_training/my_training.yaml

# Custom training options
cameras:
  - cam_wrist
  - cam_overhead
task_override: null
use_delta_actions: false
```

### Environment Variables

Configure training behavior via environment variables:

```bash
# Filter cameras (comma-separated)
export LEROBOT_CAMERA_FILTER="cam_wrist,cam_overhead"

# Override task for all samples (useful for SmolVLA)
export LEROBOT_TASK_OVERRIDE="Pick up the red cube"
```

## Running Training

### ACT (Action Chunking Transformer)

ACT is a transformer-based policy that predicts action sequences:

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.chunk_size=100 \
  --training.batch_size=8 \
  --training.num_workers=4 \
  --training.steps=100000 \
  --wandb.enable=true \
  --wandb.project=anvil-training
```

#### Key ACT Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `policy.chunk_size` | Number of actions to predict | 50-100 |
| `policy.n_action_steps` | Actions to execute per inference | 50-100 |
| `policy.dim_model` | Transformer hidden dimension | 256-512 |
| `policy.n_heads` | Number of attention heads | 4-8 |
| `policy.n_encoder_layers` | Encoder layers | 4 |
| `policy.n_decoder_layers` | Decoder layers | 1-2 |

### SmolVLA (Vision-Language-Action)

SmolVLA is a vision-language-action model for language-conditioned policies:

```bash
# Set task override for language conditioning
export LEROBOT_TASK_OVERRIDE="Pick up the red cube and place it in the bin"

uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --training.batch_size=4 \
  --training.steps=50000 \
  --wandb.enable=true
```

### Using Delta Actions

Convert absolute actions to relative (delta) actions:

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --use-delta-actions \
  --training.steps=100000
```

Delta actions compute: `action = action - observation.state`, which can improve generalization for some tasks.

### Camera Filtering

Train with a subset of cameras to reduce model complexity:

```bash
export LEROBOT_CAMERA_FILTER="cam_wrist"

uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act
```

## Monitoring with Weights & Biases

### Setup

1. Create a [Weights & Biases](https://wandb.ai/) account
2. Login to wandb:
   ```bash
   wandb login
   ```

### Enable Logging

Add wandb parameters to your training command:

```bash
uv run lerobot-train \
  ... \
  --wandb.enable=true \
  --wandb.project=my-project \
  --wandb.entity=my-team \
  --wandb.name=experiment-001
```

### Tracked Metrics

Training automatically logs:

- **Loss curves**: Policy loss, reconstruction loss
- **Learning rate**: Current learning rate
- **Gradient norms**: For debugging training stability
- **Evaluation metrics**: Success rate on held-out episodes
- **Model checkpoints**: Periodic weight saves

## Training Outputs

### Checkpoint Structure

Training saves checkpoints to `outputs/<run-id>/`:

```
outputs/2024-01-15_12-00-00/
├── config.json              # Training configuration
├── checkpoints/
│   ├── checkpoint_10000/    # Periodic checkpoints
│   ├── checkpoint_20000/
│   └── ...
├── last/                    # Latest checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── optimizer.pt
└── logs/                    # Training logs
```

### Resuming Training

Resume from a checkpoint:

```bash
uv run lerobot-train \
  --resume=outputs/2024-01-15_12-00-00/checkpoints/checkpoint_50000
```

## Uploading Models to HuggingFace Hub

### Authentication

Login to HuggingFace:

```bash
huggingface-cli login
```

### Upload Trained Model

Upload your trained model for deployment or sharing:

```bash
uv run mcap-upload \
  --model-path outputs/2024-01-15_12-00-00/last \
  --repo-id your-username/my-robot-policy \
  --commit-message "Trained ACT policy for pick and place"
```

### Manual Upload

Alternatively, upload manually:

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="outputs/2024-01-15_12-00-00/last",
    repo_id="your-username/my-robot-policy",
    repo_type="model",
)
```

## Training Tips

### Hyperparameter Tuning

1. **Start small**: Begin with fewer training steps to validate pipeline
2. **Learning rate**: Start with 1e-4, reduce if loss is unstable
3. **Batch size**: Larger is generally better, limited by GPU memory
4. **Chunk size**: Match to typical task duration in timesteps

### Data Quality

- **More data**: 50-100 demonstrations per task recommended
- **Consistent data**: Remove failed or interrupted episodes
- **Balanced data**: Include variation in initial conditions

### Debugging Training

```bash
# Verbose logging
uv run lerobot-train ... --training.log_freq=100

# Smaller batch for debugging
uv run lerobot-train ... --training.batch_size=2 --training.steps=100
```

## Next Steps

After training:

1. **Evaluate locally**: Test the model on held-out episodes
2. **Deploy for inference**: See [Deployment Guide](deployment-guide.md)
3. **Iterate**: Collect more data if performance is insufficient
