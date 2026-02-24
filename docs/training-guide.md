# Training Guide

End-to-end workflow for training imitation learning models.

## Pipeline

1. **Convert**: `mcap-convert` — MCAP recordings → LeRobot dataset
2. **Validate**: `dataset-validate` — verify dataset integrity
3. **Train**: `lerobot-train` — train a policy

## Step 1: Convert MCAP to LeRobot Format

```bash
uv run mcap-convert -i data/raw/my-session -o data/datasets/my-dataset --config configs/mcap_converter/openarm_bimanual.yaml
```

See [mcap_converter CLI reference](../packages/mcap_converter/README.md#cli-tools) for all options.

### Output Dataset Structure

```
data/datasets/my-dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── observation.images.<camera>/
            └── episode_000000.mp4
```

## Step 2: Validate Dataset

```bash
uv run dataset-validate --root data/datasets/my-dataset
```

## Step 3: Train

### ACT (Action Chunking Transformer)

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --training.batch_size=8 \
  --training.steps=100000 \
  --wandb.enable=true \
  --wandb.project=anvil-training
```

### SmolVLA (Vision-Language-Action)

```bash
LEROBOT_TASK_OVERRIDE="Pick up the red cube" uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --training.batch_size=4 \
  --training.steps=50000
```

### With Camera Filtering

```bash
LEROBOT_CAMERA_FILTER="waist,wrist_r" uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act
```

### With Delta Actions

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --use-delta-actions
```

See [lerobot_training reference](../packages/lerobot_training/README.md) for all transforms and environment variables.

## Monitoring with Weights & Biases

```bash
wandb login
uv run lerobot-train ... --wandb.enable=true --wandb.project=my-project
```

## Resuming Training

```bash
uv run lerobot-train --resume=outputs/<run-id>/checkpoints/checkpoint_50000
```

## Tips

- **Start small**: Validate with `--training.steps=100 --training.batch_size=2` first
- **Data quantity**: 50-100 demonstrations per task recommended
- **Chunk size**: Match `policy.chunk_size` to typical task duration in timesteps
- **Learning rate**: Start with 1e-4, reduce if loss is unstable
