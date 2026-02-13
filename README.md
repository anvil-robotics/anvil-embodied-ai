# Anvil-Embodied-AI

Infrastructure for deploying imitation learning models on Anvil robot platforms.

## Overview

Anvil-Embodied-AI provides a pipeline for imitation learning on Anvil robots:

1. **Data Collection**: Record teleoperation demonstrations as ROS2 MCAP files
2. **Data Conversion**: Convert MCAP recordings to LeRobot v3.0 dataset format
3. **Model Training**: Train ACT, SmolVLA, or other policies via LeRobot

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

## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/     # MCAP to LeRobot conversion
│   └── lerobot_training/   # Training utilities & transforms
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `mcap-convert` | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect` | Inspect MCAP file structure and topics |
| `mcap-to-video` | Extract MCAP image topics to MP4 videos |
| `dataset-validate` | Validate a converted LeRobot dataset |
| `mcap-upload` | Upload datasets to HuggingFace Hub |
| `lerobot-train` | Train imitation learning models |

## Documentation

- [Architecture](docs/architecture.md)
- [Data Collection Guide](docs/data-collection.md)
- [Training Guide](docs/training-guide.md)

## License

Apache License 2.0 - see [LICENSE](LICENSE).
