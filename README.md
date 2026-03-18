<p align="center">
  <a href="https://anvil.bot/">
    <img src="material/anvil.png" alt="Anvil" width="120" />
  </a>
</p>

<h1 align="center">Anvil-Embodied-AI</h1>

<p align="center">
  <a href="https://anvil.bot/"><img src="https://img.shields.io/badge/Website-anvil.bot-blue?style=for-the-badge" alt="Website" /></a>
  <a href="https://docs.anvil.bot/"><img src="https://img.shields.io/badge/Documentation-docs.anvil.bot-green?style=for-the-badge" alt="Docs" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge" alt="License" /></a>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-yellow?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://docs.ros.org/en/jazzy/"><img src="https://img.shields.io/badge/ROS2-Jazzy-22314E?style=flat-square&logo=ros&logoColor=white" alt="ROS2" /></a>
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-v0.5.0-ff69b4?style=flat-square&logo=huggingface&logoColor=white" alt="LeRobot" /></a>
</p>

---

## Overview

This repository is the embodied AI stack for the Anvil platform — data conversion, model training, and real-time inference for robot manipulation policies.

```
  Anvil Devbox (Data collection)          This repo (anvil-embodied-ai)
┌──────────────────────────────┐    ┌──────────────────────────────────────────────────────────┐
│  Teleoperation + Recording   │───>│  Convert      ───>  Train         ───>  Run Inference    │
│  MCAP files                  │    │  mcap-convert       anvil-trainer       ROS2 CycloneDDS  │
└──────────────────────────────┘    └──────────────────────────────────────────────────────────┘
```

### The Full Pipeline

| Stage                        | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **0. Data Collection** | Record teleoperation demos as ROS2 MCAP files through[ Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox). |
| **1. Data Conversion** | Convert MCAP recordings to LeRobot v3.0 datasets                                                                 |
| **2. Model Training**  | Train ACT, SmolVLA, or other policies via LeRobot                                                                |
| **3. Run Inference**   | Deploy trained models on a GPU PC via ROS2 CycloneDDS                                                            |

> **Don't have data yet?** The [Anvil OpenARM Quest Teleop Kit](https://shop.anvil.bot/products/openarm-quest-teleop-kit) gives you everything you need to start collecting teleoperation demonstrations out of the box — robot hardware, cameras, control software, and recording tools included. See our [data collection guide](https://docs.anvil.bot/software/collecting-data) for details.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker
- Collected MCAP dataset recordings from an [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox)

### Installation

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --all-packages
```

### 0. Data Collection

Record teleoperation demos as ROS2 MCAP files through an [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox). See the [data collection guide](https://docs.anvil.bot/software/collecting-data) for details.

### 1. Data Conversion

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

### 2. Model Training

Train a policy on the converted dataset:

```bash
uv run anvil-trainer \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --output_dir=data/training-output
```

Supported policy types: `act`, `diffusion`, `smolvla`, `pi0`, `pi0_fast`, `groot`, `xvla`

VLA-family policies (Pi0, Pi0-Fast, GROOT, XVLA, SmolVLA) require extra dependencies:

```bash
uv sync --all-packages --extra pi        # Pi0, Pi0-Fast
uv sync --all-packages --extra groot      # GROOT 1.5
uv sync --all-packages --extra xvla       # XVLA
uv sync --all-packages --extra smolvla    # SmolVLA
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

#### Distributed Inference Architecture

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4x 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

The Anvil Devbox runs [anvil-loader](https://docs.anvil.bot/software/starting-robot-operation) for real-time robot control, streaming joint states and camera feeds over CycloneDDS. This repo runs on a separate GPU PC, subscribing to those streams, running the trained policy, and publishing action commands back. See the [full documentation](https://docs.anvil.bot/) for setup details.

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
├── material/                      # Logo and visual assets
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
| `anvil-trainer`    | Train ML models                             |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
