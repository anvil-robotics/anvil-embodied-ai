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
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-v0.5.1-ff69b4?style=flat-square&logo=huggingface&logoColor=white" alt="LeRobot" /></a>
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
| **2. Model Training**  | Train ACT, Diffusion, SmolVLA, Pi0, or Pi0.5 policies via LeRobot v0.5.1                                        |
| **3. Offline Evaluation** | Validate model performance offline against ground-truth actions — dataset replay (`anvil-eval`) or full ROS2 pipeline replay (`anvil-eval-ros`) |
| **4. Run Inference**   | Deploy trained models on a GPU PC via ROS2 CycloneDDS                                                            |

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

Pick the config that matches your recording mode and arm count:

| Config | Teleop mode | Arms | Action source |
|--------|-------------|------|---------------|
| `openarm_bimanual.yaml` | Leader-follower | Bimanual | Leader joint positions |
| `openarm_bimanual_quest.yaml` | Quest VR | Bimanual | Command topics |
| `openarm_single_quest.yaml` | Quest VR | Single (right) | Command topics |
| `openarm_single_quest_afo.yaml` | Quest VR | Single (right) | Observation lookahead (`action_from_observation`) |

```bash
# Leader-follower, bimanual
uv run mcap-convert --input-dir data/raw/my-sessions --config configs/mcap_converter/openarm_bimanual.yaml

# Quest VR, bimanual
uv run mcap-convert --input-dir data/raw/my-sessions --config configs/mcap_converter/openarm_bimanual_quest.yaml

# Quest VR, single-arm — command topic was not recorded
uv run mcap-convert --input-dir data/raw/my-sessions --config configs/mcap_converter/openarm_single_quest_afo.yaml
```

**`action_from_observation`**

Use this when the command topic (`/follower_*/commands`) was not captured in the MCAP. Instead of reading from a command topic, the converter shifts the observation forward by N frames:

```
action[t] = observation.state[t + N]   (default N = 10, ≈333 ms at 30 fps)
```

This is pre-configured in `openarm_single_quest_afo.yaml`. To adjust the lookahead, set `action_from_observation_n` in the config file. When `action_from_observation: true` is set, any `action_topics` entries in the config are ignored — the pipeline is deterministic regardless of what was captured.

> **Output path:** the dataset is always saved to `<output-dir>/<input-dir-name>/`. With the default setting (`--output-dir data/datasets`), the result is `data/datasets/my-sessions/`.

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

Checkpoints are saved to `model_zoo/<dataset>/<policy>_<timestamp>/` by default. Run `anvil-trainer --help` for the full flag reference.

#### Common Parameters

| Flag | Description |
|------|-------------|
| `--dataset.root=PATH` | Path to converted LeRobot dataset |
| `--policy.type=TYPE` | Policy type (see table above) |
| `--job_name=NAME` | Run name — auto-generated as `<policy>_<timestamp>` if omitted |
| `--exclude-observation=KEY1,KEY2` | Drop observation keys from training. Use the suffix after `observation.` — supports both images and non-image keys: `images.chest`, `images.wrist_l`, `velocity`, `effort`. Can also be set via `LEROBOT_EXCLUDE_OBSERVATION`. |
| `--backbone=resnet18` | Vision backbone for ACT / Diffusion: `resnet18` (default) · `resnet34` · `resnet50` |
| `--split-ratio=8,1,1` | Dataset split (train,val,test). Val loss is logged regularly; test loss is logged at every checkpoint |
| `--task-description="..."` | Task prompt — required for SmolVLA / Pi0 / Pi0.5 |

#### Common Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--steps=100000` | 100k | Total training steps |
| `--batch_size=8` | 8 | Reduce if GPU OOM |
| `--save_freq=10000` | 10k | Checkpoint interval |
| `--use-delta-actions` | off | Convert actions to delta form (action − observation.state). Persisted to `anvil_config.json` so inference applies the inverse automatically. |
| `--delta-exclude-joints=JOINT1,JOINT2` | none | Joints to keep in absolute space when `--use-delta-actions` is on. Resolved by name from the dataset's `meta/info.json`. Useful for grippers, which often train better in absolute space (e.g. `--delta-exclude-joints=left_finger,right_finger`). |
| `--delta-stats-n-steps=N` | 8 | Look-ahead steps used when computing delta normalization statistics. Includes multi-step displacements `action[t+k] - state[t]` for k = 0…N in the stats, so the normalizer's range covers the full chunk instead of only single-step deltas. Prevents loss imbalance for ACT + delta and widens the MIN_MAX clip boundary for Diffusion + delta. Set to `1` to revert to single-frame delta stats. Episode boundaries are respected — cross-episode pairs are excluded. |
| `--policy.normalization_mapping='{...}'` | policy default | e.g. `{"ACTION":"MIN_MAX","STATE":"MEAN_STD","VISUAL":"IDENTITY"}`<br><br>Keys:<br>`ACTION` · `STATE` · `VISUAL`<br>Values:<br>`MEAN_STD`   — normalise by μ/σ<br>`MIN_MAX`    — normalise to [−1, 1] by observed min/max<br>`QUANTILE10` — normalise by p10/p90 (Pi0.5 default; requires quantile stats\*)<br>`IDENTITY`   — passthrough (always use for images)<br><br>**`ACTION` guidance by policy:**<br>• **Diffusion** — use `MIN_MAX` (default). Diffusion clips the denoised action to ±1 in normalised space at every step (`clip_sample=True, clip_sample_range=1.0`); `MEAN_STD` causes extreme actions beyond ±1 σ to be silently truncated, hurting peak tracking.<br>• **ACT / SmolVLA / Pi0** — use `MEAN_STD`.<br>• **Pi0.5** — use `MEAN_STD` unless quantile stats are available (see note below).<br><br>`QUANTILE10` requires `q01`/`q99` stats not produced by `mcap-convert` — see note below. `MIN_MAX` and `MEAN_STD` are always available. |
| `--resume=PATH` | — | Resume a previous job. `PATH` is the job root or a specific checkpoint dir (e.g. `model_zoo/my-task/checkpoints/020000`). Omit checkpoint to resume from `last`. |

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
| `--wandb.project=NAME` | W&B project name — auto-set to the dataset folder name; `--job_name` (= `<policy>_<timestamp>`) becomes the run name |

Key metrics: `train/loss` (should decrease steadily), `train/grad_norm` (spikes indicate instability — try lowering LR), `val/loss` (logged at every checkpoint if `--val-split-ratio` is set). All sample commands below include `--wandb.enable=false` — flip to `true` to start logging.

#### [ACT](docs/training-tips.md#act)

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
```

#### [Diffusion](docs/training-tips.md#diffusion-policy)

Good at tasks with multimodal action distributions (e.g. the robot can complete a task via multiple valid paths). Produces smooth motions and requires no chunk tuning — at the cost of slower inference than ACT due to the denoising loop.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.normalization_mapping='{"ACTION":"MIN_MAX","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false \  # set true to save training logs to W&B
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
  --task-description="Grab the gray doll and put it in the bucket"
```

After training SmolVLA / Pi0 / Pi0.5, the task description is automatically read from `anvil_config.json` saved in the checkpoint — no manual copy needed. To override it at inference time, set it explicitly in `configs/lerobot_control/inference_default.yaml`:

```yaml
model:
  task_description: "Grab the gray doll and put it in the bucket"
```

#### Fine-tune from a local checkpoint

To start a **new** training run using a previously trained checkpoint as the initial weights (step counter resets to 0, new output directory):

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.path=model_zoo/my-task/checkpoints/last/pretrained_model
```

When `--policy.path` is given, `--policy.type` is not needed — the policy type is read from the checkpoint. Backbone injection is also skipped since the backbone config is already embedded in the checkpoint.

> **`--policy.path` vs `--resume`:** `--policy.path` starts a fresh run from the checkpoint's weights (new output dir, step 0). `--resume` continues a stopped run in-place (same output dir, step counter carries over). Use `--policy.path` to fine-tune on a new dataset or with different hyperparameters; use `--resume` to recover from an interrupted training.

For VLA policies (SmolVLA, Pi0, Pi0.5), `--policy.pretrained_path` can also point to a local directory instead of a HuggingFace repo ID:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=/path/to/local/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="Grab the gray doll and put it in the bucket"
```

#### Resume a run

```bash
# Resume from the latest checkpoint
uv run anvil-trainer --resume=model_zoo/pick-and-place

# Resume from a specific checkpoint (e.g. step 20000)
uv run anvil-trainer --resume=model_zoo/pick-and-place/checkpoints/020000
```

Only pass `--resume` — all other settings are restored from the saved `train_config.json` in the checkpoint. Delta action settings (`--use-delta-actions`, `--delta-exclude-joints`) are inherited automatically from the checkpoint's `anvil_config.json`.

### 3. Offline Evaluation

Before deploying to a robot, you can validate model performance offline. Two complementary modes are available:

| Mode | Command | What it tests |
|------|---------|---------------|
| **Dataset replay** | `anvil-eval` | Feeds dataset observations directly into the model — fast, no ROS2 needed |
| **ROS2 MCAP replay** | `anvil-eval-ros` | Replays raw MCAP recordings through the full ROS2 inference stack in Docker — mirrors real deployment |

Both produce the same metrics (MAE, RMSE, per-joint trajectory plots) and write results to a unified layout:

```
eval_results/{dataset_name}/{job_name}/{checkpoint}/
├── raw/     ← anvil-eval output
└── ros/     ← anvil-eval-ros output
```

#### Dataset Replay (`anvil-eval`)

Feeds LeRobot dataset observations directly into the model and compares predictions against ground-truth actions. Fast — no ROS2 or Docker required.

```bash
uv run anvil-eval \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --dataset data/datasets/my-task \
  --num-eps 5 \
  --device cuda
```

**Features:**
- **Flexible Splitting:** Evaluates across `train`, `val`, and `test` splits (samples equally from each).
- **Trajectory Plots:** View predicted vs ground-truth for each joint (grippers are automatically moved to the end).
- **Summary Box Plots:** Analyze the distribution of errors across joints and dataset splits.

Results are saved to `eval_results/{dataset}/{job}/{checkpoint}/raw/`.

#### ROS2 MCAP Replay (`anvil-eval-ros`)

Replays raw MCAP recordings through the full inference Docker stack — the same inference node that runs on the real robot — and records predicted vs ground-truth actions over ROS2 topics. This mode catches integration issues that dataset replay cannot (topic remapping, timing, action chunking in the live loop).

```bash
uv run anvil-eval-ros \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --mcap-root data/raw/my-task \
  --num-eps 3
```

**How it works:**

```
Host: anvil-eval-ros
  │  generates eval_plan.json → launches docker compose
  │
  ├─ [inference]      model running on GPU, publishing to /eval/* topics
  ├─ [mcap-player]    replays one MCAP per episode; coordinates via /eval/episode_start|done
  └─ [eval-recorder]  records GT + predicted actions → computes metrics → saves results
```

- **Auto arm detection:** reads the model's `config.json` (action_dim) and the dataset's `conversion_config.yaml` to determine which arm topics to subscribe to — no manual YAML editing required.
- **Topic isolation:** inference publishes to `/eval/follower_*/commands` while GT actions replay on the original topic names, so both coexist on the same ROS2 network.
- **Graceful failures:** metrics are always saved even if plotting fails (e.g. matplotlib not available in the container).

**Common flags:**

| Flag | Description |
|------|-------------|
| `--checkpoint PATH` | Checkpoint directory (reads `split_info.json` + `anvil_config.json`) |
| `--mcap-root PATH` | Raw MCAP directory (e.g. `data/raw/my-task`) |
| `--num-eps N` | Sample up to N episodes per split (train/val/test) |
| `--episodes "0,3,5"` | Manually specify episode indices (overrides split sampling) |
| `--output-dir PATH` | Override default output directory |
| `--seed N` | Random seed for episode sampling (default: 42) |
| `--dataset-dir PATH` | Path to the converted LeRobot dataset. Used as an extra search candidate for `conversion_config.yaml` when raw MCAP and dataset are not co-located in the standard `data/raw` / `data/datasets` layout |
| `--base-inference-config PATH` | Override the default `configs/lerobot_control/inference_eval.yaml`. Useful when evaluating a model trained on a subset of cameras or a single arm |
| `--monitor` | Enable real-time inference monitor: records a per-step CSV (`obs_state`, `raw_output`, `control_cmd`) and generates a joint-level PNG report in `<output-dir>/monitor/` |

Results are saved to `eval_results/{dataset}/{job}/{checkpoint}/ros/`.

**Inference Monitor (`--monitor`)**

When `--monitor` is passed, a fourth `inference-monitor` container starts alongside the stack. It subscribes to `/monitor/*` topics published by the inference node and writes:

```
ros/
├── monitor/
│   ├── inference_data.csv      ← per-step obs_state / raw_output / control_cmd
│   └── inference_report.png   ← joint-level overlay plot
└── plots/
    └── episode_NNNN_*.png     ← GT (blue) / Pred (red) / Raw model output (orange)
```

The orange "Raw" line in episode plots shows the model's output **before** postprocessing (delta restore, safety clamping) — useful for diagnosing whether the policy or the postprocessor is responsible for a tracking error.

> **Requires Docker with NVIDIA GPU support.** The inference container is built automatically on first run. Set `LEROBOT_EXTRAS` if your model needs extra dependencies (e.g. `pi`, `smolvla`).

### 4. Run Inference

All inference scenarios go through `scripts/run_inference.sh` — the single entry point that selects the right compose file, manages the monitor output directory, and auto-plots monitor data on exit.

```
./scripts/run_inference.sh [--fake-hardware] [--monitor] [--echo-topic-only] [COMPOSE_ARGS...]
```

| Flag | Description |
|---|---|
| `--fake-hardware` | Use `docker-compose.fake-hardware.yml` (bridge network + CycloneDDS, no real robot) |
| `--monitor` | Enable monitor profile. For production: starts `inference_monitor_node`, pre-creates `MONITOR_OUTPUT_DIR` as current user, writes CSV + PNG on exit. For fake-hardware: starts FPS-only monitoring (`echo_topic_only`) |
| `--echo-topic-only` | Subscribe and log FPS without running a model — verify DDS connectivity without a checkpoint |

**Optional env overrides:**

| Variable | Description |
|---|---|
| `MODEL_PATH` | Host path to checkpoint (required for production inference) |
| `CONFIG_FILE` | Custom inference config YAML (default: `./configs/lerobot_control/inference_default.yaml`) |
| `MONITOR_OUTPUT_DIR` | Host dir for monitor CSV/PNG output (default: `./monitor_output`) |
| `LEROBOT_EXTRAS` | Comma-separated policy extras to install in the image (e.g. `pi,smolvla`). **Rebuild after changing:** `docker compose build` |

Run `./scripts/run_inference.sh --help` for the full reference.

#### Test with Fake Hardware First (Recommended)

Simulate the 2-PC setup locally (bridge network + CycloneDDS) before connecting to real hardware. `mock-robot` acts as the Robot PC; `monitor`/`inference` act as the GPU PC.

```bash
# 1. Validate DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor up --build

# 2. Validate full inference pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware up --build --profile inference
```

If `Control Loop` hits 30 Hz in the stats output, the setup is ready for real hardware.

#### Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With real-time inference monitor — records per-step CSV; plots PNG to ./monitor_output/ on exit
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor up --build

# Verify DDS connectivity without a model checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

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
# safety:
#   max_position_delta: 0.1   # Max joint position change per step (default: 0.1 rad).
#   min_position_delta: 0.05  # Min per-joint change required to publish a new command.
#                             #   Holds the last command until cumulative change exceeds
#                             #   this threshold — useful when model delta outputs are
#                             #   too small to overcome motor friction. Default: disabled.
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

Use `./scripts/run_inference.sh --fake-hardware` to simulate this 2-PC setup locally before connecting to real hardware.

## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/            # MCAP to LeRobot conversion
│   ├── anvil_trainer/             # Training utilities & transforms
│   ├── anvil_eval/                # Offline evaluation: dataset replay (anvil-eval)
│   └── anvil_eval_ros/            # ROS2 MCAP replay evaluation CLI (anvil-eval-ros)
├── ros2/
│   └── src/lerobot_control/       # ROS2 inference node (Jazzy)
├── configs/
│   ├── cyclonedds/                # CycloneDDS peer configs (GPU PC, Robot PC)
│   ├── lerobot_control/           # Inference node config (cameras, joints, arms)
│   └── mcap_converter/            # Data conversion config
├── docker/
│   └── inference/                 # Dockerfile + entrypoint
├── scripts/
│   ├── run_inference.sh               # Entry point for all inference scenarios (wraps docker compose)
│   └── plot_monitor_csv.py            # Plot obs.state / raw_output / control_cmd from monitor CSV
├── docker-compose.yml                    # Production inference (GPU PC)
├── docker-compose.fake-hardware.yml      # Fake hardware: simulate 2-PC DDS cooperation (monitor / inference profiles)
├── docker-compose.eval.yml               # ROS2 MCAP replay eval: 3-service stack (inference + mcap-player + eval-recorder)
├── material/                      # Logo and visual assets
├── .env.example                   # Environment template
└── model_zoo/                     # Trained model weights (gitignored)
```

## Training Tips

> Full guide: [docs/training-tips.md](docs/training-tips.md)

**ACT (TL;DR)**
- Match `chunk_size` and `n_action_steps` to your task speed (50 for precise, 100 for sweeping)
- Enable temporal ensemble at inference for smoother execution — no retraining needed
- Use `--exclude-observation` to drop cameras or non-image observations (e.g. `images.wrist_l`, `velocity`) that don't add signal
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
| `anvil-eval`       | Offline dataset replay evaluation — feed dataset observations into model, compare predictions against GT (MAE/RMSE/plots) |
| `anvil-eval-ros`   | ROS2 MCAP replay evaluation — replay raw recordings through full inference Docker stack, record predicted vs GT actions |
| `mcap-convert`     | Convert MCAP recordings to LeRobot datasets |
| `mcap-inspect`     | Inspect MCAP file structure, topics, and message counts — useful before conversion to check what's inside a recording |
| `mcap-to-video`    | Extract MCAP image topics to MP4 videos — useful for visually reviewing raw recordings |
| `dataset-validate` | Validate a converted LeRobot dataset        |
| `mcap-upload`      | Upload a converted dataset to HuggingFace Hub |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
