# Getting Started with Anvil-Embodied-AI

This guide will help you set up your development environment and get started with the Anvil-Embodied-AI platform.

## Prerequisites

Before you begin, ensure you have the following installed:

### Python 3.10+

The project requires Python 3.10 or higher. Check your version:

```bash
python3 --version
```

### uv Package Manager

[uv](https://github.com/astral-sh/uv) is used for fast, reliable Python package management. The setup script will install it automatically if not present, or you can install it manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Docker

Docker is required for ROS2 development and deployment. Install Docker Engine following the [official documentation](https://docs.docker.com/engine/install/).

Verify installation:

```bash
docker --version
docker compose version
```

### ROS2 Jazzy (for local development)

For local ROS2 development without Docker, install ROS2 Jazzy following the [ROS2 documentation](https://docs.ros.org/en/jazzy/Installation.html).

After installation, source the ROS2 setup:

```bash
source /opt/ros/jazzy/setup.bash
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
```

### 2. Run the Development Setup Script

The setup script automates the development environment configuration:

```bash
./scripts/setup-dev.sh
```

This script will:

1. Check for and install uv if needed
2. Create a virtual environment at `.venv/`
3. Install all workspace packages in development mode
4. Install pre-commit hooks for code quality

### 3. Verify the Installation

Run the test suite to ensure everything is working:

```bash
./scripts/test.sh
```

Run the linter to check code quality:

```bash
./scripts/lint.sh
```

## Quick Start Workflow

The Anvil-Embodied-AI platform follows a four-stage workflow for training and deploying robot policies:

```
[Data Collection] -> [Conversion] -> [Training] -> [Deployment]
```

### 1. Data Collection

Record robot demonstrations using teleoperation. The robot's sensors (cameras, joint encoders) are recorded to MCAP files.

See: [Data Collection Guide](data-collection.md)

### 2. Data Conversion

Convert raw MCAP recordings into LeRobot dataset format:

```bash
uv run mcap-convert \
  --input data/raw/my-session \
  --output data/datasets/my-dataset \
  --config configs/mcap_converter/openarm_bimanual.yaml
```

See: [Training Guide - Data Conversion](training-guide.md#converting-mcap-to-lerobot-format)

### 3. Model Training

Train imitation learning models using the LeRobot framework:

```bash
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act
```

See: [Training Guide](training-guide.md)

### 4. Deployment

Deploy trained models for real-time inference on your robot:

```bash
docker compose -f docker/inference/docker-compose.yml up
```

See: [Deployment Guide](deployment-guide.md)

## Project Structure

```
anvil-embodied-ai/
├── packages/                  # Python packages
│   ├── mcap_converter/        # MCAP to LeRobot conversion
│   └── lerobot_training/      # Custom training utilities
├── ros2/                      # ROS2 packages
│   └── src/lerobot_control/   # Inference ROS2 node
├── configs/                   # Configuration files
│   ├── mcap_converter/        # Conversion configs
│   ├── lerobot_training/      # Training configs
│   └── lerobot_control/       # Inference configs
├── data/                      # Dataset storage (gitignored)
│   ├── raw/                   # Raw MCAP recordings
│   └── datasets/              # Processed LeRobot datasets
├── model_zoo/                 # Trained models (gitignored)
├── docker/                    # Docker configurations
└── scripts/                   # Development scripts
```

## Activating the Virtual Environment

After setup, you can activate the virtual environment manually:

```bash
source .venv/bin/activate
```

Or use `uv run` to execute commands within the environment:

```bash
uv run python -c "import mcap_converter; print(mcap_converter.__version__)"
```

## Next Steps

- [Data Collection Guide](data-collection.md) - Learn how to record robot demonstrations
- [Training Guide](training-guide.md) - Train your first imitation learning model
- [Deployment Guide](deployment-guide.md) - Deploy models for real-time inference
- [Architecture Overview](architecture.md) - Understand the system design
- [Extending to New Robots](extending-to-new-robots.md) - Add support for your robot

## Getting Help

- Check the [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for bugs or feature requests
- Visit [https://anvil.bot/](https://anvil.bot/) for more information about Anvil Robotics
