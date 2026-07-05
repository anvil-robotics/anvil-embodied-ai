# Real-Robot Calibration Workflow

Use this checklist when the OpenArm arrives. The goal is to turn the first real
robot sessions into usable calibration data before training or deploying a
policy.

## Before the Robot Arrives

Run the lightweight repo preflight:

```bash
./scripts/real_robot_preflight.sh
```

For a full Python dependency check, run:

```bash
./scripts/real_robot_preflight.sh --full
```

Then edit `.env`:

- `ROS_DOMAIN_ID` must match the robot/devbox.
- `CONFIG_FILE` should match the robot setup. Start with
  `./configs/lerobot_control/inference_default.yaml` for bimanual OpenArm.
- Leave `MODEL_PATH` as a placeholder until you have a trained checkpoint.

## First Connection

Do the vendor hardware setup and motor verification before starting inference.
Keep the robot clear of obstacles and keep any emergency stop available.

From the GPU PC, verify DDS/camera/joint-state connectivity without loading a
model:

```bash
./scripts/run_inference.sh --echo-topic-only up --build
```

Expected result: the inference node subscribes to `/joint_states` and camera
topics and reports usable FPS. Fix networking or topic mismatches before
collecting task data.

## First Recording Set

Record small, boring sessions first. They are more useful for calibration than
failed task attempts.

1. `idle-YYYYMMDD`: 10-20 seconds with the robot powered, stationary, and all
   cameras unobstructed.
2. `joint-sweep-YYYYMMDD`: slow free-space movement through comfortable ranges.
3. `camera-check-YYYYMMDD`: move a textured object through each camera view.
4. `timing-sweep-YYYYMMDD`: deliberate small command changes, one arm at a
   time, with pauses between movements. This is the session used to estimate
   command-to-state response lag.
5. `task-smoke-YYYYMMDD`: 3-5 simple teleop attempts for one task.

Copy the MCAP session folders under `data/raw/`, for example:

```text
data/raw/
  openarm-calibration-20260706/
    0001/
      *.mcap
      metadata.yaml
```

## Inspect Raw MCAP

Inspect one MCAP from each session:

```bash
uv run mcap-inspect data/raw/openarm-calibration-20260706/0001/*.mcap
```

Confirm these topics exist and have nonzero counts:

- `/joint_states`
- `/cam_waist/image_raw/compressed`
- `/cam_wrist_r/image_raw/compressed`
- `/cam_chest/image_raw/compressed`
- `/cam_wrist_l/image_raw/compressed`
- `/follower_l_forward_position_controller/commands` for bimanual Quest data
- `/follower_r_forward_position_controller/commands` for bimanual Quest data

## Timing Report

Generate a timing report from every first-day calibration recording. Keep the
JSON artifact; it becomes your baseline for future model operation.

```bash
uv run --with 'mcap~=1.3' --with 'mcap-ros2-support~=0.5.7' \
  python scripts/mcap_timing_report.py \
  data/raw/openarm-calibration-20260706 \
  --output-dir eval_results/calibration/openarm-calibration-20260706
```

The report captures:

- per-topic message counts, observed Hz, period statistics, and timing gaps;
- header-stamp-to-MCAP-log-time lag for stamped messages;
- estimated command-to-joint-state lag for left and right arms when command
  topics are present;
- warnings for dropped camera frames, slow command streams, or large gaps.

Use the `timing-sweep` recording for the most reliable command-to-state lag
estimate. Slow, separated movements make lag estimates much more meaningful
than ordinary task demonstrations.

If the command topics are missing but the follower state is good, use an
`action_from_observation` config such as
`configs/mcap_converter/openarm_single_quest_afo.yaml` for single-arm data, or
add a bimanual AFO config before converting.

## Convert and Validate

For bimanual Quest teleop:

```bash
uv run mcap-convert \
  --input-dir data/raw/openarm-calibration-20260706 \
  --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --output-dir data/datasets \
  --fps 30 \
  --debug-plot-episodes 5

uv run dataset-validate --root data/datasets/openarm-calibration-20260706
```

For leader-follower teleop, use:

```bash
--config configs/mcap_converter/openarm_bimanual.yaml
```

## Calibration Checks

Before collecting a large dataset, verify:

- Joint order: converted `observation.state` and `action` are
  `[left 8 joints, right 8 joints]`.
- Sign conventions: positive command changes match positive observed joint
  motion.
- Zero pose: stationary recordings do not show unexpected offsets or drift.
- Timing: camera FPS is close to 30 Hz and the joint stream is stable.
- Camera mapping: `waist`, `chest`, `wrist_r`, and `wrist_l` point to the
  intended physical cameras.
- Action lag: command trajectories lead observed follower motion by a plausible
  amount, not by whole seconds.
- Topic gaps: the timing report does not show repeated camera or command gaps
  during otherwise normal recordings.
- Header skew: stamped messages do not show large or drifting
  header-to-log-time offsets.

Use the converter debug plots and extracted videos to check these quickly:

```bash
uv run mcap-to-video \
  -i data/raw/openarm-calibration-20260706/0001/*.mcap \
  -o videos/openarm-calibration-20260706 \
  --fps 30 \
  --resize 640x480
```

## First Training Loop

Train a small ACT baseline first. It is fast and good for validating the data
pipeline:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/openarm-calibration-20260706 \
  --policy.type=act \
  --job_name=act_first \
  --steps=20000 \
  --save_freq=5000 \
  --batch_size=8 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false
```

Run dataset replay before deploying:

```bash
uv run anvil-eval \
  --checkpoint model_zoo/openarm-calibration-20260706/act_first/checkpoints/last \
  --dataset data/datasets/openarm-calibration-20260706 \
  --num-eps 5 \
  --device cuda
```

Then replay raw MCAP through the ROS2 inference stack:

```bash
uv run anvil-eval-ros \
  --checkpoint model_zoo/openarm-calibration-20260706/act_first/checkpoints/last \
  --mcap-root data/raw/openarm-calibration-20260706 \
  --num-eps 3 \
  --monitor
```

## First Live Inference

Only run live inference after the model passes replay checks. Start with
conservative safety limits in `configs/lerobot_control/inference_default.yaml`:

```yaml
safety:
  max_position_delta: 0.05
```

Run with the monitor enabled:

```bash
MONITOR_OUTPUT_DIR=$(pwd)/monitor_output/openarm-calibration-20260706-act-first \
MODEL_PATH=$(pwd)/model_zoo/openarm-calibration-20260706/act_first/checkpoints/last \
./scripts/run_inference.sh --monitor up --build
```

Review `monitor_output/inference_report.png` before increasing speed, task
complexity, or action delta limits.

For every live model run, preserve these artifacts together:

- raw MCAP for the run;
- `eval_results/calibration/.../timing_report.json`;
- `monitor_output/.../inference_data.csv`;
- `monitor_output/.../inference_report.png`;
- the inference YAML used for the run;
- `.env` values for `ROS_DOMAIN_ID`, `CONTROL_FREQ`, `CONFIG_FILE`, and
  `MODEL_PATH`;
- the checkpoint path and git commit.
