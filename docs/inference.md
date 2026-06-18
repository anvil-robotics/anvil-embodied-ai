[← Back to README](../README.md)

# Run Inference

All inference scenarios go through `scripts/run_inference.sh`.

**Start by copying `.env.example` to `.env` and editing it for your setup:**

```bash
cp .env.example .env
# then edit .env
```

### `.env` Variables

`.env` is the primary configuration file — set it once and reuse across runs. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | Yes (inference) | Host path to checkpoint dir. Must be absolute or start with `./` — bare relative paths are treated as Docker named volumes. |
| `ROS_DOMAIN_ID` | Yes | ROS2 domain ID — must match the Anvil Devbox. Leave empty for localhost-only. |
| `CYCLONEDDS_URI` | Yes | Path to CycloneDDS XML config (e.g. `configs/cyclonedds/gpu_pc.xml`). |
| `LEROBOT_EXTRAS` | VLA only | Comma-separated policy extras built into the Docker image — e.g. `smolvla`, `pi,smolvla`. **Rebuild the image after changing:** `docker compose build`. ACT and Diffusion leave this empty. |
| `HF_CACHE` | VLA only | Host path to HuggingFace model cache (default: `~/.cache/huggingface`). Required for Pi0, Pi0.5, SmolVLA — they load the PaliGemma tokenizer at runtime. |
| `CONFIG_FILE` | Yes | Path to inference config YAML (default: `configs/lerobot_control/inference_default.yaml`). |
| `ACTION_TYPE` | No | Action type passed to the **inference monitor node** (`inference_monitor_node`) only. The main inference node always reads this from `anvil_config.json` in the checkpoint via `resolve_action_type()` — this env var does **not** override it. Options: `absolute` · `delta_obs_t` · `delta_sequential`. |
| `ECHO_TOPIC_ONLY` | No | `true` = skip model loading, subscribe topics and log FPS only. For verifying DDS connectivity without a GPU or checkpoint. Equivalent to `--echo-topic-only`. |
| `MONITOR_ENABLE` | No | `true` = enable the inference monitor node (records per-step CSV + PNG report). Equivalent to `--monitor-enable`, but without the auto-plot on exit and output dir pre-creation that the flag provides. |
| `DEBUG` | No | `true` = enable extra metrics: action smoothness, queue depth stats, Action FPS. |

For full descriptions and defaults, see [`.env.example`](../.env.example).

### Script Flags

The script flags are a lightweight way to override behaviour at the command line without editing `.env`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor-enable] [--echo-topic-only] [--debug] [COMPOSE_ARGS...]
```

| Flag | What it does |
|------|-------------|
| `--fake-hardware` | Switches to `docker-compose.fake-hardware.yml` — simulates a 2-PC setup locally over a bridge network (CycloneDDS, no real robot). |
| `--monitor-enable` | Adds `--profile monitor` to the compose command. In production (non-fake-hardware) mode also exports `MONITOR_ENABLE=true`, pre-creates the output dir as the current user, and auto-plots the CSV on exit. |
| `--echo-topic-only` | Exports `ECHO_TOPIC_ONLY=true` — subscribes to topics and logs FPS without loading a model or GPU. Equivalent to setting `ECHO_TOPIC_ONLY=true` in `.env`. |
| `--debug` | Exports `DEBUG=true` — enables extra metrics: action smoothness, queue depth stats, Action FPS. Equivalent to setting `DEBUG=true` in `.env`. |

## Test with Fake Hardware First (Recommended)

```bash
# 1. Verify DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor-enable up --build

# 2. Validate full pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware --profile inference up --build
```

> **Fake-hardware note:** `--echo-topic-only` / `ECHO_TOPIC_ONLY` and `MONITOR_ENABLE` env vars are
> **not** read by `docker-compose.fake-hardware.yml`. The monitor service hardcodes
> `echo_topic_only:=true` regardless; the inference service does not expose `MONITOR_ENABLE`.
> These variables only take effect with the production `docker-compose.yml`.

If `Control Loop` hits 30 Hz, the setup is ready for real hardware.

## Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With inference monitor
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor-enable up --build

# Verify DDS connectivity without a checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

> **`MODEL_PATH` must be absolute or start with `./`.** Bare relative paths are treated as named Docker volumes.
> ```bash
> MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last   # recommended
> MODEL_PATH=./model_zoo/my-task/checkpoints/last        # also valid
> ```

## Inference Config (`configs/lerobot_control/inference_default.yaml`)

Before running, review this file:

**Model**
```yaml
model:
  task_description: null
  # VLA-only (SmolVLA / Pi0 / Pi0.5): task prompt the model was trained on.
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
```

**Per-model inference tuning** — override checkpoint defaults without retraining:
```yaml
inference_tuning:

  act:
    n_action_steps: null
    # Steps to execute per chunk before re-running inference.
    # null = use training value. Jittery? → raise. Hesitates? → lower.
    temporal_ensemble_coeff: null
    # Re-infers every step with exponentially weighted overlapping predictions.
    # Use 0.01 (paper default). Forces n_action_steps=1.

  diffusion:
    n_action_steps: null
    # Steps to execute per chunk. null = use training value.
    num_inference_steps: 10
    # Denoising iterations at inference time.
    # null = num_train_timesteps (100 steps, ~300ms on GPU).
    # 10   = ~30ms on GPU — recommended for real-time deployment.

  rtc:
    # VLA models only (SmolVLA / Pi0 / Pi0.5)
    inference_delay: 10
    # Fallback step-count before LatencyTracker auto-calibrates.
    # Rule of thumb: ceil(first_inference_ms × control_freq / 1000)
    queue_trigger_threshold: 50
    # Re-trigger inference when ActionQueue depth ≤ this.
    execution_horizon: 12
    # Steps consumed per chunk before the next inference fires.
    max_guidance_weight: 10.0
    prefix_attention_schedule: EXP
```

**Safety limits:**
```yaml
# safety:
#   max_position_delta: 0.1
#   # Hard limit on joint position change per control step (radians).
#   min_position_delta: 0.05
#   # Minimum cumulative change before publishing a new command.
#   # Holds the last command until threshold is crossed — useful for
#   # overcoming motor dead zones / friction. Default: disabled (null).
```

## Distributed Inference Architecture

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4× 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

The Anvil Devbox streams joint states and camera feeds over CycloneDDS. The GPU PC subscribes to those streams, runs the policy, and publishes action commands back. See the [full documentation](https://docs.anvil.bot/) for network setup.

---

[← Back to README](../README.md)
