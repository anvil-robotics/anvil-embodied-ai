# Anvil-Embodied-AI

[![CI](https://github.com/anvil-robotics/anvil-embodied-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/anvil-robotics/anvil-embodied-ai/actions/workflows/ci.yml)
[![Docker](https://github.com/anvil-robotics/anvil-embodied-ai/actions/workflows/docker.yml/badge.svg)](https://github.com/anvil-robotics/anvil-embodied-ai/actions/workflows/docker.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A robust mission-control platform for Physical AI. Providing the essential infrastructure to deploy, orchestrate, and optimize advanced AI models on Anvil Robotics' robot platforms.

## Overview

Anvil-Embodied-AI provides an end-to-end pipeline for imitation learning on Anvil robots:

1. **Data Collection**: Record teleoperation demonstrations as MCAP files
2. **Data Conversion**: Convert MCAP recordings to LeRobot dataset format
3. **Model Training**: Train ACT, SmolVLA, or other imitation learning models
4. **Deployment**: Run real-time inference on your robot via ROS2

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for package management
- Docker (for ROS2 deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai

# Set up development environment
./scripts/setup-dev.sh
```

### Convert Data

```bash
# Convert MCAP recordings to LeRobot format
uv run mcap-convert \
  --input data/raw/my-session \
  --output data/datasets/my-dataset \
  --config configs/mcap_converter/openarm_bimanual.yaml
```

### Train a Model

```bash
# Train an ACT policy
uv run lerobot-train \
  --dataset data/datasets/my-dataset \
  --policy act \
  --output-dir outputs/my-model
```

### Deploy

```bash
# Build and run the inference container
./scripts/build-docker.sh
docker compose -f docker/inference/docker-compose.yml up
```

## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/     # MCAP to LeRobot conversion
│   └── lerobot_training/   # Model training utilities
├── ros2/src/
│   └── lerobot_control/    # ROS2 inference package
├── configs/                # Configuration files
├── docker/                 # Docker deployment
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Data Collection Guide](docs/data-collection.md)
- [Training Guide](docs/training-guide.md)
- [Deployment Guide](docs/deployment-guide.md)
- [Architecture](docs/architecture.md)
- [Extending to New Robots](docs/extending-to-new-robots.md)

## CLI Tools

| Command | Description |
|---------|-------------|
| `mcap-convert` | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect` | Inspect MCAP file contents |
| `mcap-validate` | Validate LeRobot dataset integrity |
| `mcap-upload` | Upload datasets to HuggingFace Hub |
| `lerobot-train` | Train imitation learning models |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- [Anvil Robotics](https://anvil.bot/)
- [LeRobot](https://github.com/huggingface/lerobot)
- [HuggingFace Hub](https://huggingface.co/)
