# Anvil-Embodied-AI

Infrastructure for training ML models and deploying them on Anvil robot platforms.

## Overview

Anvil-Embodied-AI provides a pipeline for training and deploying ML models on Anvil robots:

1. **Data Collection**: Record teleoperation demonstrations as ROS2 MCAP files
2. **Data Conversion**: Convert MCAP recordings to LeRobot v3.0 dataset format
3. **Model Training**: Train policies via LeRobot
4. **Inference**: Deploy trained models on a GPU PC communicating with the Robot PC via CycloneDDS

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker

### Installation

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --all-packages
```

### 1. Convert Data (ETL)

Convert MCAP recordings from teleoperation sessions into LeRobot v3.0 datasets.

Two teleop modes are supported — pick the config that matches your recording:

```bash
# For data recorded with leader-follower
uv run mcap-convert --input data/raw/my-sessions --output data/datasets/my-dataset --config configs/mcap_converter/openarm_bimanual.yaml

# For data recorded with Quest
uv run mcap-convert --input data/raw/my-sessions --output data/datasets/my-dataset --config configs/mcap_converter/openarm_bimanual_quest.yaml
```

Then validate the converted dataset:

```bash
uv run dataset-validate --root data/datasets/my-dataset
```

Expected output: 5 checks (load, info, features, read, batch) all showing `[OK]`.

### 2. Train a Model

Train a policy on the converted dataset:

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.repo_id=my-policy \
  --output_dir=data/training-output
```

Optional flags:
- `--steps=100000` — total training steps (default 100k)
- `--batch_size=8` — adjust based on GPU memory
- `--save_freq=10000` — checkpoint frequency
- `LEROBOT_CAMERA_FILTER=chest,waist` — train with a subset of cameras
- `--use-delta-actions` — convert actions to relative (action - state)

Checkpoints are saved to `--output_dir`.

### 3. Run Inference

```bash
cp .env.example .env              # configure MODEL_PATH, ROS_DOMAIN_ID, CycloneDDS
docker compose up                  # run inference on GPU PC
```

## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/            # MCAP to LeRobot conversion
│   └── lerobot_training/          # Training utilities & transforms
├── ros2/
│   └── src/lerobot_control/       # ROS2 inference node (Jazzy)
├── configs/
│   ├── cyclonedds/                # CycloneDDS peer configs (GPU PC, Robot PC)
│   ├── lerobot_control/           # Inference node config (cameras, joints, arms)
│   └── mcap_converter/            # Data conversion config
├── docker/
│   └── inference/                 # Dockerfile + entrypoint
├── docker-compose.yml             # Production inference (GPU PC)
├── docker-compose.fake-hardware.yml # Fake hardware test (no real hardware needed)
├── .env.example                   # Environment template
└── model_zoo/                     # Trained model weights (gitignored)
```

## CLI Tools

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `mcap-convert`     | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect`     | Inspect MCAP file structure and topics      |
| `mcap-to-video`    | Extract MCAP image topics to MP4 videos     |
| `dataset-validate` | Validate a converted LeRobot dataset        |
| `mcap-upload`      | Upload datasets to HuggingFace Hub          |
| `lerobot-train`    | Train ML models                             |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
