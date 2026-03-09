# Anvil-Embodied-AI

Infrastructure for deploying imitation learning models on Anvil robot platforms.

## Overview

Anvil-Embodied-AI provides a pipeline for imitation learning on Anvil robots:

1. **Data Collection**: Record teleoperation demonstrations as ROS2 MCAP files
2. **Data Conversion**: Convert MCAP recordings to LeRobot v3.0 dataset format
3. **Model Training**: Train ACT, SmolVLA, or other policies via LeRobot
4. **Inference**: Deploy trained models on a GPU PC communicating with the Robot PC via CycloneDDS

## Architecture

```
   Robot PC (anvil-workcell)        CycloneDDS          GPU PC (anvil-embodied-ai)
┌───────────────────────────┐   ┌────────────────┐   ┌───────────────────────────┐
│  ros2_control             │   │                │   │  lerobot_control          │
│  joint_states (500 Hz)    │◄──┤ Gigabit Switch ├──►│  inference (30 Hz)        │
│  cameras (4x 30 Hz)       │   │                │   │  action commands          │
└───────────────────────────┘   └────────────────┘   └───────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for package management
- Docker & Docker Compose (for inference tests)

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
# Leader-follower teleop (actions derived from leader joints)
uv run mcap-convert -i data/raw/my-session -o /tmp/my-dataset --config configs/mcap_converter/openarm_bimanual.yaml

# Quest VR teleop (actions from position command topics)
uv run mcap-convert -i data/raw/my-session -o /tmp/my-dataset --config configs/mcap_converter/openarm_bimanual_quest.yaml
```

Then validate the converted dataset:

```bash
uv run dataset-validate --root /tmp/my-dataset
```

Expected output: 5 checks (load, info, features, read, batch) all showing `[OK]`.

### 2. Train a Model

Train an ACT policy on the converted dataset:

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=/tmp/my-dataset \
  --policy.type=act \
  --policy.repo_id=my-policy \
  --output_dir=/tmp/my-training-output
```

Optional flags:
- `--steps=100000` — total training steps (default 100k)
- `--batch_size=8` — adjust based on GPU memory
- `--save_freq=10000` — checkpoint frequency
- `LEROBOT_CAMERA_FILTER=chest,waist` — train with a subset of cameras
- `--use-delta-actions` — convert actions to relative (action - state)

Checkpoints are saved to `--output_dir`. The trained model can be used for inference.

### 3. Run Inference

#### Production (GPU PC + Robot PC)

```bash
cp .env.example .env              # configure MODEL_PATH, ROS_DOMAIN_ID, CycloneDDS
docker compose up                  # run inference on GPU PC
```

#### Monitor-only (no model, verify data streams)

```bash
MONITOR_ONLY=true docker compose up
```

### 4. Test Without Hardware

No robot needed. Uses a fake hardware node that publishes dummy camera images and joint states over CycloneDDS.

```bash
# Monitor-only: verify data streams without a model (no GPU needed)
MONITOR_ONLY=true docker compose -f docker-compose.fake-hardware.yml up --build --abort-on-container-exit

# Full inference: load a trained model and verify action output (model_zoo/ is mounted)
MODEL_PATH=/workspace/model_zoo/test/pretrained_model docker compose -f docker-compose.fake-hardware.yml up --build --abort-on-container-exit
```

Expected output:
- `fake_hardware` publishes 4 cameras at ~30 Hz and joint states at ~500 Hz
- `inference-node` receives and logs matching rates in 5-second stat intervals
- With `MODEL_PATH`: inference node also publishes actions to per-arm controller topics, and the fake hardware node validates them
- `discovery-check` prints `=== Discovery check PASSED ===`

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
├── model_zoo/                     # Trained model weights (gitignored)
├── scripts/                       # Utility scripts
└── docs/                          # Documentation
```

## CLI Tools

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `mcap-convert`     | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect`     | Inspect MCAP file structure and topics      |
| `mcap-to-video`    | Extract MCAP image topics to MP4 videos     |
| `dataset-validate` | Validate a converted LeRobot dataset        |
| `mcap-upload`      | Upload datasets to HuggingFace Hub          |
| `lerobot-train`    | Train imitation learning models             |

## Documentation

- [Architecture](docs/architecture.md)
- [Data Collection Guide](docs/data-collection.md)
- [Training Guide](docs/training-guide.md)

## License

Apache License 2.0 - see [LICENSE](LICENSE).
