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
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-v0.6.0-ff69b4?style=flat-square&logo=huggingface&logoColor=white" alt="LeRobot" /></a>
</p>

---

## 📢 News

- **2026-07-09** — Upgraded to LeRobot v0.6.0 and added selected native foundation-policy support.

_See full history in [CHANGELOG.md](CHANGELOG.md)._

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

| Stage | Description |
|-------|-------------|
| **0. Data Collection** | Record teleoperation demos as MCAP files via [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox) |
| **1. Data Conversion** | Convert MCAP recordings to LeRobot v3.0 datasets → [docs/data-conversion.md](docs/data-conversion.md) |
| **2. Model Training** | Train ACT, Diffusion, Pi/SmolVLA, and selected LeRobot foundation policies → [docs/training.md](docs/training.md); [OpenArm 2 shirt-fold A/B runbook](docs/shirt-fold-training.md) |
| **3. Offline Evaluation** | Validate model performance against ground-truth before deploying → [docs/evaluation.md](docs/evaluation.md) |
| **4. Run Inference** | Deploy trained models on a GPU PC via ROS2 CycloneDDS → [docs/inference.md](docs/inference.md) |

> **Don't have data yet?** The [Anvil OpenARM Quest Teleop Kit](https://shop.anvil.bot/products/openarm-quest-teleop-kit) gives you everything you need to start collecting demonstrations out of the box. See the [data collection guide](https://docs.anvil.bot/software/collecting-data).

---

## Table of Contents

- [Installation](#installation)
  - [Available Models](#available-models)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- FFmpeg (required by `torchcodec` for video decoding)
- Docker (for inference and ROS2 eval)

```bash
sudo apt install ffmpeg   # Ubuntu / Debian
```

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --all-packages
```

### Available Models

This repo supports these native LeRobot v0.6 policies for training, offline evaluation, and ROS2 inference:

| Model | `--policy.type` | Install extra | Notes |
|-------|----------------|---------------|-------|
| ACT | `act` | Base install | Action-chunking baseline |
| Diffusion | `diffusion` | Base install | Diffusion Policy baseline |
| SmolVLA | `smolvla` | `smolvla` | Language-conditioned VLA; RTC chunk inference |
| Pi0 | `pi0` | `pi` | Flow-matching VLA; RTC chunk inference |
| Pi0.5 | `pi05` | `pi` | Larger Pi0 variant; RTC chunk inference |
| MolmoAct2 | `molmoact2` | `molmoact2` | Language-conditioned foundation policy; RTC chunk inference |
| GR00T N1.7 | `groot` | `groot` | Language-conditioned foundation policy; RTC chunk inference |
| Multitask DiT | `multi_task_dit` | `multi_task_dit` | Language-conditioned foundation policy; synchronous chunk inference |
| EVO1 | `evo1` | `evo1` | Language-conditioned foundation policy; RTC chunk inference |
| FastWAM | `fastwam` | `fastwam` | Language-conditioned foundation policy; synchronous chunk inference |
| VLA-JEPA | `vla_jepa` | `vla_jepa` | Language-conditioned foundation policy; synchronous by default, RTC-capable |

Use `all` to install every optional policy extra:

```bash
uv sync --all-packages --extra smolvla
uv sync --all-packages --extra smolvla --extra pi   # multiple
uv sync --all-packages --extra all                  # all optional policies
```

> **GPU / CUDA note:** The root `pyproject.toml` pins torch to the `cu132` index. If your machine uses a different CUDA driver, change `pytorch-cu132` to the matching PyTorch CUDA index before syncing.

---

## Usage

### 0. Data Collection

Record teleoperation demonstrations as ROS2 MCAP files through an [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox). See the [data collection guide](https://docs.anvil.bot/software/collecting-data) for details.

### 1. Data Conversion ([doc](docs/data-conversion.md))

Convert MCAP recordings into LeRobot v3.0 datasets. Pick the config that matches your recording setup and run `mcap-convert`.

### 2. Model Training ([doc](docs/training.md))

Train ACT, Diffusion, Pi/SmolVLA, and selected LeRobot-native foundation policies. Checkpoints saved to `model_zoo/<dataset>/<job_name>/`.

### 3. Offline Evaluation ([doc](docs/evaluation.md))

Validate model performance before deploying. Two modes: dataset replay (`anvil-eval`) and ROS2 MCAP replay (`anvil-eval-ros`).

### 4. Run Inference ([doc](docs/inference.md))

Deploy trained models on a GPU PC via ROS2 CycloneDDS. All inference scenarios go through `scripts/run_inference.sh`.

---


## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/            # MCAP → LeRobot dataset conversion
│   ├── anvil_trainer/             # Training wrapper: transforms, splits, val loss
│   ├── anvil_eval/                # Offline evaluation: dataset replay
│   ├── anvil_eval_ros/            # Offline evaluation: ROS2 MCAP replay
│   └── anvil_shared/              # Shared utilities (pure-Python, no ML deps)
├── ros2/
│   └── src/lerobot_control/       # ROS2 inference node (Jazzy)
├── configs/
│   ├── cyclonedds/                # CycloneDDS peer configs
│   ├── lerobot_control/           # Inference YAML configs (cameras, joints, arms)
│   └── mcap_converter/            # Data conversion configs
├── docs/
│   ├── data-conversion.md         # Data conversion guide
│   ├── training.md                # Model training guide
│   ├── evaluation.md              # Offline evaluation guide
│   └── inference.md               # Inference deployment guide
├── docker/
│   └── inference/                 # Dockerfile + entrypoint
├── scripts/
│   ├── run_inference.sh           # Entry point for all inference scenarios
│   └── plot_monitor_csv.py        # Plot obs.state / raw_output / control_cmd from CSV
├── tests/
│   ├── smoke/                     # End-to-end smoke tests
│   └── unit/                      # Unit tests per package
├── docker-compose.yml                    # Production inference
├── docker-compose.fake-hardware.yml      # Simulate 2-PC setup locally
├── docker-compose.eval.yml               # ROS2 MCAP replay eval stack
├── docker-compose.monitor-smoke-test.yml # Monitor smoke test
├── .env.example                          # Environment variable template
└── model_zoo/                            # Trained checkpoints (gitignored)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
