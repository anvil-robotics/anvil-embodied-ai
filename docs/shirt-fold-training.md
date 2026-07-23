[← Back to README](../README.md)

# OpenArm 2 Shirt-fold Training

This runbook defines a controlled 5,000-step A/B test for the Anvil OpenArm 2
shirt-fold dataset. Run these two fine-tunes before spending a larger training
budget or training the embodiment adapter:

| Run | Initialization | Recipe |
|---|---|---|
| HF prior | Hugging Face OpenArm shirt-fold policy | `configs/training/shirt_fold_pi05_hf_phase_aligned.yaml` |
| Base prior | Upstream Pi0.5 base policy | `configs/training/shirt_fold_pi05_base_phase_aligned.yaml` |

The recipes intentionally differ only in the pretrained model/revision and run
identity. This isolates whether Hugging Face's folding prior helps despite its
modified OpenArm Mini embodiment.

## Dataset contract

Both runs use the generated
`datasets/shirt-fold/lerobot-hf-phase-aligned` dataset:

- 33 episodes and 34,850 frames at 30 Hz;
- three video observations:
  `left_wrist`, `right_wrist`, and `base`, each 3×270×480;
- 16D state/action vectors in radians, ordered right arm first, then left arm;
- seven arm joints plus one gripper value per side;
- `q01` and `q99` statistics required by Pi0.5's `QUANTILES`
  normalization.

The trim is reproducible from
`lerobot-hf-phase-aligned.trim-plan.json`; generated datasets and adapter
caches remain local or in artifact storage and are not committed to Git.
Do not concatenate the Hugging Face OpenArm Mini demonstrations with the Anvil
episodes. Their left-first ordering, degree units, and different link geometry
make them a reference dataset, not compatible training rows.

Trimming removes the homing/setup phase and final idle phase. Consequently,
inference must begin near a demonstrated `start_state` from
`meta/trim_manifest.json`, not from the mechanical home pose.

## Why these settings are paired

- The policy's native relative-action processor is enabled, excluding grippers.
  Pass `--action-type=absolute` to Anvil so a second delta transform is not
  applied.
- The explicit action names lock the right-first joint mapping.
- The explicit 16D features override Pi0.5 base's incompatible 32D defaults.
- The 30-action horizon matches the HF folding checkpoint and covers one second
  at 30 Hz.
- Both runs train the action expert for 5,000 steps at `1e-5`, log every 100
  steps, and checkpoint every 500 steps.
- W&B records train loss every 100 steps, validation loss every 500 steps, and
  test loss at every checkpoint.

## Local commands

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_hf_phase_aligned.yaml \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="HF folding initialization; phase-aligned Anvil OpenArm 2 demos"
```

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_base_phase_aligned.yaml \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="Pi0.5 base initialization; phase-aligned Anvil OpenArm 2 demos"
```

With seed 1000, the episode split is fixed:

- train: 0, 1, 3–5, 7–10, 13, 15–26, 28–32;
- validation: 2, 11, 14;
- test: 6, 12, 27.

## Vast launch through openarm2-cloud-runner

Upload the complete phase-aligned dataset—including `data/`, `meta/`, and
`videos/`—to a dedicated Hugging Face dataset repository and pin its commit
revision. Do not point Vast at the untrimmed upload or the HF reference dataset.

Use direct Anvil mode in two separate cloud-runner configs:

```yaml
secrets:
  wandb_api_key_env: WANDB_API_KEY

repos:
  anvil-embodied-ai:
    url: https://github.com/bensonlee5/anvil-embodied-ai.git
    default_ref: <COMMIT_CONTAINING_THE_RECIPES>

job:
  kind: train_policy
  policy_type: pi05
  task_description: Fold the T-shirt properly
  job_name: <UNIQUE_RUN_NAME>
  dataset_repo: <HF_USER>/openarm2-shirt-fold-phase-aligned-v1
  dataset_revision: <PINNED_DATASET_COMMIT>
  checkpoint_repo: <HF_USER>/<UNIQUE_CHECKPOINT_REPO>
  run_smoke_test: false
  hub_push_interval_seconds: 120
  extra_args:
    - --config_path=configs/training/<RECIPE>.yaml
    - --action-type=absolute
    - --split-ratio=8,1,1
    - --note=<RUN_DESCRIPTION>
```

Use an 80 GB or larger GPU and enough disk for ten multi-gigabyte checkpoints.
Launch each config separately:

```bash
./scripts/vast/launch --config configs/<RUN_CONFIG>.yaml
```

The checkpoint repositories and job names must be distinct. Keep the VM after
the first checkpoint is verified so a bootstrap, synchronization, or upload
problem can be inspected without losing its local logs.

## Quality gates

Treat a live process as necessary but insufficient. For both runs:

1. Confirm the resolved log prints the expected dataset, model revision, 16D
   feature schema, right-first action names, and relative-action processor.
2. Confirm loss and gradient norm are finite from the first logged step.
3. At step 500, require a numeric checkpoint, a W&B validation point, and a
   successful Hugging Face checkpoint push.
4. Compare the A/B runs at matched steps. Prefer validation/test loss over train
   loss; a lower train loss with a flat or worsening validation curve is not an
   improvement.
5. Investigate NaN/Inf immediately, repeated gradient spikes, loss increasing
   across several logging intervals, or a widening train/validation gap.
6. Before live control, run offline dataset replay and inspect per-joint error,
   especially shoulder joints 1–3, predicted motion magnitude, and temporal
   alignment.

The HF-initialized run is the first deployment candidate if its validation and
replay metrics are at least as good as the base-initialized run. Train the
embodiment adapter next only if the folding prior is useful but joint-space
replay still shows a systematic embodiment-dependent error.

## Deployment preflight

Use the shadow inference config before enabling commands. The shirt-fold live
and shadow configs deliberately apply no local absolute or delta action limiter;
the only requested enforcement is the downstream `anvil-loader` safety layer.
Stage the arms near a demonstrated trimmed start pose and verify radians,
right-first ordering, camera routing, and processor metadata before live motion.
