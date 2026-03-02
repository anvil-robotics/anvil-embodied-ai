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

### Installation

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --all-packages
```

### Convert Data

```bash
uv run mcap-convert -i data/raw/my-session -o /tmp/my-dataset --config configs/mcap_converter/openarm_bimanual.yaml
```

### Validate Dataset

```bash
uv run dataset-validate --root /tmp/my-dataset
```

### Train a Model

```bash
uv run lerobot-train --dataset.repo_id=local --dataset.root=/tmp/my-dataset --policy.type=act
```

### Run Inference (Docker)

```bash
cp .env.example .env              # configure model path, ROS_DOMAIN_ID, CycloneDDS
docker compose up                  # run inference on GPU PC
```

### Test Distributed Connectivity

```bash
# Monitor-only mode: verify DDS data streams without loading a model
MONITOR_ONLY=true docker compose up

# Mock distributed test: CycloneDDS discovery on Docker bridge (no hardware needed)
docker compose -f docker-compose.mockdist.yml up --build --abort-on-container-exit
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
├── docker-compose.mockdist.yml    # Mock CycloneDDS discovery test
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
