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
uv sync --all-packages --extra all                  # all policies
```

> **GPU / CUDA note:** The root `pyproject.toml` pins torch to the PyTorch `cu128` index so `uv sync` always installs the CUDA-enabled build. If your machine runs a different CUDA driver version, change `pytorch-cu128` → `pytorch-cu126` (or `cu124`) in `pyproject.toml` before syncing.

### 0. Data Collection

Record teleoperation demos as ROS2 MCAP files through an [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox). See the [data collection guide](https://docs.anvil.bot/software/collecting-data) for details.

### 1. Data Conversion

Convert MCAP recordings from teleoperation sessions into LeRobot v3.0 datasets.

Two teleop modes are supported — pick the config that matches your recording:

```bash
# For data recorded with leader-follower
uv run mcap-convert --input data/raw/my-sessions --output data/datasets --config configs/mcap_converter/openarm_bimanual.yaml

# For data recorded with Quest
uv run mcap-convert --input data/raw/my-sessions --output data/datasets --config configs/mcap_converter/openarm_bimanual_quest.yaml
```

> **Output path:** the dataset is always saved to `<output-dir>/<input-dir-name>/`. With the examples above, the result is `data/datasets/my-sessions/`.

**Common flags:**

| Flag | Description |
|---|---|
| `--resume` | Skip already-converted episodes and append new ones — safe to re-run after interruption |
| `--max-episodes N` | Convert only the first N episodes — useful for a quick sanity check before full conversion |
| `--fps N` | Override output FPS. FPS is auto-detected from MCAP metadata by default; upsampling beyond source FPS is blocked |
| `--vcodec` | Video codec: `h264` (default, widely viewable), `hevc` (H.265), or `libsvtav1` (best compression) |
| `--robot-type` | Robot configuration: `anvil_openarm` (default) or `anvil_yam` |

Run `uv run mcap-convert --help` for the full flag reference.

Then validate the converted dataset:

```bash
uv run dataset-validate --root data/datasets/my-sessions
```

Expected output: 5 checks (load, info, features, read, batch) all showing `[OK]`.

### 2. Model Training

Supported policies:

| Policy | `--policy.type` | Notes |
|--------|----------------|-------|
| ACT | `act` | Action Chunking Transformer — fast, reliable |
| Diffusion | `diffusion` | Diffusion Policy — smooth, multimodal |
| SmolVLA | `smolvla` | Language-conditioned VLA; requires task description |
| Pi0 | `pi0` | Flow-matching VLA; PaliGemma-3B backbone |
| Pi0.5 | `pi05` | Larger Pi0 variant (~4B params); higher VRAM |

Checkpoints are saved to `model_zoo/<job_name>/`. Run `anvil-trainer --help` for the full flag reference.

#### Common Parameters

| Flag | Description |
|------|-------------|
| `--dataset.root=PATH` | Path to converted LeRobot dataset |
| `--policy.type=TYPE` | Policy type (see table above) |
| `--job_name=NAME` | Run name; auto-generated if omitted |
| `--camera-filter=chest,waist` | Train with a subset of cameras |
| `--task-description="..."` | Task prompt — required for SmolVLA / Pi0 / Pi0.5 |

#### Common Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--steps=100000` | 100k | Total training steps |
| `--batch_size=8` | 8 | Reduce if GPU OOM |
| `--save_freq=10000` | 10k | Checkpoint interval |
| `--use-delta-actions` | off | Relative actions (target − state) |
| `--policy.normalization_mapping='{...}'` | policy default | e.g. `{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}`<br><br>Keys:<br>`ACTION` · `STATE` · `VISUAL`<br>Values:<br>`MEAN_STD`   — normalise by μ/σ<br>`MIN_MAX`    — normalise to [0,1]<br>`QUANTILE10` — normalise by p10/p90 (Pi0.5 default; requires quantile stats\*)<br>`IDENTITY`   — passthrough (always use for images)<br><br>mcap-convert datasets lack quantile stats — use `MEAN_STD` for `ACTION`/`STATE`. |
| `--resume=true` | off | Resume from `--output_dir` checkpoint |

\* quantile stats (`q01`/`q99`) are not produced by `mcap-convert`. See [Pi0.5 — Normalization mapping](docs/training-tips.md#normalization-mapping) for how to add them or switch normalization method.

#### Weights & Biases

[Weights & Biases](https://wandb.ai) streams training metrics in real time — useful for tracking loss curves, gradient norms, and comparing runs. One-time setup:

```bash
uv run wandb login
```

Then add to any training command:

| Flag | Description |
|---|---|
| `--wandb.enable=true` | Enable W&B logging (default: `false`) |
| `--wandb.project=NAME` | W&B project name — `--job_name` becomes the run name within it |

Key metrics: `train/loss` (should decrease steadily), `train/grad_norm` (spikes indicate instability — try lowering LR). All sample commands below include `--wandb.enable=false` — flip to `true` to start logging.

#### [ACT](docs/training-tips.md#act)

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
  --job_name=pick-and-place
```

#### [Diffusion](docs/training-tips.md#diffusion-policy)

Good at tasks with multimodal action distributions (e.g. the robot can complete a task via multiple valid paths). Produces smooth motions and requires no chunk tuning — at the cost of slower inference than ACT due to the denoising loop.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
  --job_name=pick-and-place
```

#### [SmolVLA](docs/training-tips.md#smolvla)

Language-conditioned — always pass `--task-description` and `--policy.pretrained_path`. Mirror the same description in the inference YAML.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
  --job_name=grabbing-smolvla \
  --task-description="Grab the gray doll and put it in the bucket"
```

#### Pi Series ([Pi0](docs/training-tips.md#pi0) / [Pi0.5](docs/training-tips.md#pi05))

Pi0 and Pi0.5 are flow-matching VLA policies from [Physical Intelligence](https://github.com/Physical-Intelligence/openpi), built on a PaliGemma-3B backbone. Both require HuggingFace access to [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) (request access on the model page) — then run [`huggingface-hub login`](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login) once to authenticate.

Key flags for both models:

| Flag | Recommendation |
|---|---|
| `--policy.train_expert_only=true` | Freeze backbone, train only action expert — lower memory, faster convergence, sufficient for most tasks |
| `--policy.compile_model=true` | Enables `torch.compile` for ~10–20% throughput gain on the denoising loop (one-time compilation cost on first forward pass) |
| `--policy.gradient_checkpointing=true` | Reduces VRAM during backprop — always enable |
| `--policy.dtype=bfloat16` | Halves VRAM — required for Pi0.5 on a 24 GB GPU |

**Pi0**

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
  --job_name=grabbing-pi0 \
  --task-description="Grab the gray doll and put it in the bucket"
```

**Pi0.5**

Same as Pi0 but ~4B params. Requires `--num_workers=0` to prevent CPU RAM OOM from forked workers copying the full model. Use `--batch_size=16` as a starting point — reduce if GPU OOM.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=16 \
  --num_workers=0 \
  --wandb.enable=false \  # set true to save training logs to W&B
  --job_name=grabbing-pi05 \
  --task-description="Grab the gray doll and put it in the bucket"
```

After training SmolVLA / Pi0 / Pi0.5, the task description is automatically read from `anvil_config.json` saved in the checkpoint — no manual copy needed. To override it at inference time, set it explicitly in `configs/lerobot_control/inference_default.yaml`:

```yaml
model:
  task_description: "Grab the gray doll and put it in the bucket"
```

#### Resume a run

```bash
uv run anvil-trainer --resume=true --output_dir=model_zoo/pick-and-place
```

Only pass `--resume=true` and `--output_dir` — all other settings are restored from the saved `train_config.json`.

### 3. Run Inference

#### Test with Fake Hardware First (Recommended)

Before deploying to a real robot, validate your setup using the fake hardware compose file.
It uses a bridge network + CycloneDDS to simulate real 2-PC cooperation — mock-robot acts as the Robot PC, monitor/inference acts as the GPU PC.

```bash
# 1. Validate DDS connectivity + camera FPS (no model, no GPU needed)
docker compose -f docker-compose.fake-hardware.yml --profile monitor up --build

# 2. Validate full inference pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
docker compose -f docker-compose.fake-hardware.yml --profile inference up --build
```

If `Control Loop` hits the target FPS (30 Hz) in the stats output, the setup is ready for real hardware.

#### Production (Real Robot)

```bash
# MODEL_PATH must be an absolute path or start with ./
# Use $(pwd)/ prefix so tab-completion works naturally on the path first, then wrap it:
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
docker compose up

# Or use explicit ./relative path:
MODEL_PATH=./model_zoo/my-task/checkpoints/last \
docker compose up
```

**Optional env overrides:**

| Variable | Description |
|---|---|
| `CONFIG_FILE` | Host path to a custom config yaml (default: `./configs/lerobot_control/inference_default.yaml`) |
| `MONITOR_ONLY` | Set to `true` to subscribe + log FPS without loading a model — useful to verify DDS connectivity on the GPU PC |
| `LEROBOT_EXTRAS` | Comma-separated policy extras to install in the image (e.g. `pi,diffusion`). **Rebuild the image after changing:** `docker compose build` |

Before running, review `configs/lerobot_control/inference_default.yaml`:

**Model**
```yaml
model:
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
  # Set explicitly only to override the checkpoint value.
  task_description: null
```

**Inference Tuning**
```yaml
# null = use the value the model was trained with (recommended starting point)
inference_tuning:
  # Number of predicted actions to execute before running inference again.
  # The model predicts a full chunk (e.g. 100 steps) but only executes n_action_steps of them.
  n_action_steps: null

  # ACT only — re-infers every step and blends overlapping predictions with exponential weighting.
  # Smoother motion than raising n_action_steps. Use 0.01 (paper default) forces n_action_steps=1.
  temporal_ensemble_coeff: null
```

**Safety**
```yaml
# Uncomment to override the code default of 0.1 rad/step.
# safety:
#   max_position_delta: 0.1
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

Use `docker-compose.fake-hardware.yml` to simulate this 2-PC setup locally (bridge network + CycloneDDS) before connecting to real hardware.

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
├── docker-compose.yml                    # Production inference (GPU PC)
├── docker-compose.fake-hardware.yml      # Fake hardware: simulate 2-PC DDS cooperation (monitor / inference profiles)
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

**Diffusion (TL;DR)**
- Best for tasks with multiple valid completion paths — handles multimodal action distributions naturally
- Slower inference than ACT (denoising loop); consider ACT first if latency is critical
- 100k steps / batch 64 is a solid default; larger batch reduces score-matching variance
- Tune `n_action_steps` at inference if motion feels jerky — no retraining needed

**SmolVLA (TL;DR)**
- Always fine-tune from `lerobot/smolvla_base` with `--policy.load_vlm_weights=true`
- Set a specific task description via `--task-description` — it matters
- 30k–50k steps is usually enough from a pretrained base

**Pi Series — Pi0 / Pi0.5 (TL;DR)** ([openpi](https://github.com/Physical-Intelligence/openpi))
- Both require HuggingFace access to `google/paligemma-3b-pt-224` — run `huggingface-hub login` once
- Always pass `--task-description` — Pi series is language-conditioned
- Pi0.5 (4B params) additionally needs `--policy.dtype=bfloat16 --batch_size=16 --num_workers=0` on a 24 GB GPU (`--num_workers=0` prevents CPU RAM OOM from forked workers copying the full model)
- Pi0.5 requires quantile stats in `stats.json` — mcap-convert datasets don't include them. Use `--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'` (recommended), or run `augment_dataset_quantile_stats` to compute them in-place (**backs up your dataset first** — it modifies in-place). See [Pi0.5 normalization](docs/training-tips.md#normalization-mapping) for details.

**MODEL_PATH**

Point `MODEL_PATH` to a specific checkpoint step (or `last` for the latest). The `pretrained_model` subdirectory is detected automatically.

> **Note:** Docker Compose requires bind mount paths to be absolute or start with `./`.
> Bare relative paths (e.g. `model_zoo/...`) are treated as named volumes and will error.

```bash
# Recommended: use $(pwd)/ prefix (tab-completion works on the path before wrapping)
MODEL_PATH=$(pwd)/model_zoo/pick-and-place/checkpoints/last
MODEL_PATH=$(pwd)/model_zoo/pick-and-place/checkpoints/100000

# Also valid: explicit ./relative path
MODEL_PATH=./model_zoo/pick-and-place/checkpoints/last
```

## CLI Tools

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `anvil-trainer`    | Train ML models                             |
| `mcap-convert`     | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect`     | Inspect MCAP file structure, topics, and message counts — useful before conversion to check what's inside a recording |
| `mcap-to-video`    | Extract MCAP image topics to MP4 videos — useful for visually reviewing raw recordings |
| `dataset-validate` | Validate a converted LeRobot dataset        |
| `mcap-upload`      | Upload a converted dataset to HuggingFace Hub |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
