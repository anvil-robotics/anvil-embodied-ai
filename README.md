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
| **2. Model Training**  | Train ACT, Diffusion, SmolVLA, Pi0, or Pi0.5 policies via LeRobot v0.5.0                                        |
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

ACT and Diffusion are included in the base install. For other policies:

| Extra | Policy |
|---|---|
| `smolvla` | SmolVLA |
| `pi` | Pi0 / Pi0.5 |

```bash
uv sync --all-packages --extra smolvla
uv sync --all-packages --extra smolvla --extra pi   # multiple
```

> **GPU / CUDA note:** The root `pyproject.toml` pins torch to the PyTorch `cu128` index so `uv sync` always installs the CUDA-enabled build. If your machine runs a different CUDA driver version, change `pytorch-cu128` → `pytorch-cu126` (or `cu124`) in `pyproject.toml` before syncing. SmolVLA and Pi-series models also require a HuggingFace account with access to `google/paligemma-3b-pt-224` — run `huggingface-hub login` once before training.

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
  --job_name=pick-and-place
```

Checkpoints are saved to `model_zoo/<job_name>/` by default. Run `anvil-trainer --help` for the full flag reference.

Optional flags:

- `--job_name=NAME` — run name; auto-generated from policy + timestamp if omitted
- `--task-description="..."` — task prompt for SmolVLA (see below)
- `--camera-filter=chest,waist` — train with a subset of cameras
- `--use-delta-actions` — convert actions to relative (action - state)
- `--steps=100000` — total training steps (default 100k)
- `--batch_size=8` — adjust based on GPU memory
- `--save_freq=10000` — save a checkpoint every N steps

**Training SmolVLA with a task description:**

SmolVLA is language-conditioned — it requires a task description. Pass the same string at both training and inference:

```bash
uv run anvil-trainer \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --job_name=grabbing-w1 \
  --task-description="Grab the gray doll and put it in the bucket" \
  --eval_freq=0
```

**Training Pi0 / Pi0.5:**

Pi0 and Pi0.5 use a PaliGemma-3B backbone (requires HuggingFace access to `google/paligemma-3b-pt-224`). Train only the action expert to reduce GPU memory:

```bash
# Pi0
uv run anvil-trainer \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi0 \
  --policy.push_to_hub=false \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --job_name=grabbing-pi0 \
  --task-description="Grab the gray doll and put it in the bucket"

# Pi0.5
uv run anvil-trainer \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi05 \
  --policy.push_to_hub=false \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=1 \
  --num_workers=0 \
  --job_name=grabbing-pi05 \
  --task-description="Grab the gray doll and put it in the bucket"
```

> Pi0.5 (4B params) requires `--policy.dtype=bfloat16 --batch_size=1 --num_workers=0` on a 24 GB GPU.
> The `normalization_mapping` override is needed because mcap-convert datasets don't include quantile stats — see [training tips](docs/training-tips.md#pi05) for details.

Mirror the task description in `configs/lerobot_control/inference_default.yaml`:

```yaml
model:
  task_description: "Grab the gray doll and put it in the bucket"
```


**Visualizing training progress with Weights & Biases:**

```bash
# Login once
uv run wandb login

# Train with W&B enabled
uv run anvil-trainer \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --job_name=grabbing-w1 \
  --wandb.enable=true \
  --wandb.project=my-project
```

Loss curves, action prediction visualizations, and eval metrics are streamed live to [wandb.ai](https://wandb.ai).

To resume a stopped or interrupted run:

```bash
uv run anvil-trainer --resume=true --output_dir=model_zoo/pick-and-place
```

LeRobot will pick up from the latest checkpoint. Only pass `--resume=true` and `--output_dir` — all other settings are restored from the saved `train_config.json`.

### 3. Run Inference

```bash
cp .env.example .env              # configure MODEL_PATH, ROS_DOMAIN_ID, CycloneDDS
docker compose up                  # run inference on GPU PC
```

Before running, review `configs/lerobot_control/inference_default.yaml`:

**Model**
```yaml
model:
  # Task prompt for SmolVLA — must match what was used during training.
  task_description: "Grab the gray doll and put it in the bucket"
```

**Inference Tuning**
```yaml
# null = use the value the model was trained with (recommended starting point)
inference_tuning:
  # Number of predicted actions to execute before running inference again.
  # The model predicts a full chunk (e.g. 100 steps) but only executes n_action_steps of them.
  n_action_steps: null

  # ACT only — re-infers every step and blends overlapping predictions with exponential weighting.
  # Smoother motion than raising n_action_steps. Use 0.01 (paper default) will forces n_action_steps=1.
  temporal_ensemble_coeff: null
```

**Safety**
```yaml
safety:
  # Maximum joint position change allowed per control step (radians). (lower = safer)
  # But may limit fast motions. Raise cautiously if the robot feels that require quick joint travel.
  max_position_delta: 0.2
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

## Training Tips

> Full guide: [docs/training-tips.md](docs/training-tips.md)

**ACT (TL;DR)**
- Match `chunk_size` and `n_action_steps` to your task speed (50 for precise, 100 for sweeping)
- Enable temporal ensemble at inference for smoother execution — no retraining needed
- Use `--camera-filter` to drop cameras that don't add signal
- 100k steps / batch 16 is a solid default; drop to 50k for small datasets

**SmolVLA (TL;DR)**
- Always fine-tune from `lerobot/smolvla_base` with `--policy.load_vlm_weights=true`
- Set a specific task description via `--task-description` — it matters
- Set `--eval_freq=0` (no live env available)
- 30k–50k steps is usually enough from a pretrained base

**Pi0 (TL;DR)**
- Use `--policy.train_expert_only=true` to freeze the PaliGemma backbone — faster and enough for most tasks
- Always pass `--task-description` — Pi0 is language-conditioned
- Requires HuggingFace access to `google/paligemma-3b-pt-224`

**Pi0.5 (TL;DR)**
- Same as Pi0 but 4B params — always add `--policy.dtype=bfloat16 --batch_size=1 --num_workers=0` on a 24 GB GPU
- `bfloat16` is required to fit in VRAM; `num_workers=0` prevents CPU RAM OOM during model load

**MODEL_PATH gotcha**

Point to the `pretrained_model` subdirectory inside a specific checkpoint, not the top-level model folder:
```
# Correct
MODEL_PATH=model_zoo/pick-and-place/checkpoints/100000/pretrained_model
```

## CLI Tools

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `anvil-trainer`    | Train ML models                             |
| `mcap-convert`     | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect`     | Inspect MCAP file structure and topics      |
| `mcap-to-video`    | Extract MCAP image topics to MP4 videos     |
| `dataset-validate` | Validate a converted LeRobot dataset        |
| `mcap-upload`      | Upload datasets to HuggingFace Hub          |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
