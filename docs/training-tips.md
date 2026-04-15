# Training Tips

## anvil-trainer Defaults

`anvil-trainer` is a thin wrapper around LeRobot's `lerobot-train` CLI. In addition to Anvil-specific flags (`--task-description`, `--camera-filter`, `--use-delta-actions`), it injects the following LeRobot defaults automatically so you don't have to repeat them in every command:

| Injected flag | Value | Reason |
|---|---|---|
| `--dataset.repo_id` | `local` | Anvil datasets are always local; HuggingFace Hub upload is not needed for training |
| `--policy.push_to_hub` | `false` | Prevents accidental upload of checkpoints to HuggingFace Hub |
| `--eval_freq` | `0` | LeRobot's default (20 000 steps) would attempt to launch a gym simulation environment, which doesn't exist for Anvil MCAP datasets |
| `--job_name` | `<policy>_<timestamp>` | Auto-generated from policy type + timestamp (e.g. `act_20260413_143052`) — used as the W&B run name |
| `--output_dir` | `model_zoo/<dataset>/<job_name>` | Nested under dataset name for organised model zoo; auto-generated if omitted |
| `--split-ratio` | `8,1,1` | Default 80/10/10 split for train/val/test sets |
| `--wandb.project` | `<dataset name>` | Auto-set to the dataset folder name so all runs for the same task group together |
| `--policy.vision_backbone` | `resnet18` | ImageNet-pretrained ResNet18 for ACT/Diffusion — override with `--backbone=resnet34` or `--backbone=resnet50` |

Any of these can be overridden by passing the flag explicitly.

---

## MODEL_PATH — Point to a Specific Checkpoint

After training, checkpoints are saved under `model_zoo/<dataset>/<job_name>/checkpoints/<step>/pretrained_model/`.
The `MODEL_PATH` in your `.env` must point all the way to the `pretrained_model` subdirectory:

```
# Correct
MODEL_PATH=/workspace/model_zoo/my-dataset/act_20260413_143052/checkpoints/100000/pretrained_model

# Wrong — config.json not found
MODEL_PATH=/workspace/model_zoo/my-dataset/act_20260413_143052
```

---

## Visualizing Training Progress (Weights & Biases)

LeRobot uses [Weights & Biases](https://wandb.ai) for training monitoring. Enable it by passing `--wandb.enable=true`:

```bash
uv run wandb login   # one-time setup

uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --wandb.enable=true
```

**W&B project and run name are auto-set.** The W&B project is automatically set to the dataset folder name (`my-dataset`), and the run name is set to `<policy>_<timestamp>` (e.g. `act_20260413_143052`). Override either with `--wandb.project=NAME` or `--job_name=NAME` if you prefer custom names.

Key metrics to watch on the W&B dashboard:

| Metric | What it tells you |
|---|---|
| `train/loss` | Overall training loss — should decrease steadily |
| `train/grad_norm` | Gradient norm — spikes indicate instability, try lowering LR |
| `eval/avg_sum_rewards` | Task success (if eval env available) |

If you don't want W&B, training still runs fine without it — logs are printed to console.

---

## save_freq

Controls how often a checkpoint is saved (in steps). Each checkpoint writes the full model to `model_zoo/<dataset>/<job_name>/checkpoints/<step>/pretrained_model/`.

```bash
--save_freq=10000   # save every 10k steps
```

**How to tune:**
- Default `10000` is fine for most runs.
- Lower (e.g. `5000`) if you want more recovery points for long runs or unstable training.
- Higher (e.g. `25000`) to save disk space — each checkpoint can be several GB.

Only the checkpoints you explicitly need should be kept. LeRobot also always writes a `last/` checkpoint at the end of training.


---

## ACT

### Data quality first

ACT is sensitive to demonstration quality. A small set of clean, consistent demos
outperforms a large set of sloppy ones. Aim for smooth, deliberate motions and
discard failed or hesitant episodes before training.

### chunk_size and n_action_steps

`chunk_size` controls how many future actions the model predicts at once.
`n_action_steps` controls how many of those predictions are executed before
re-querying the model (default: both are 100 in this repo).

- For **fast, fine-grained tasks** (small precise movements): lower values like
  `chunk_size=50`, `n_action_steps=50` give the model more chances to correct.
- For **slow, sweeping tasks**: higher values (100+) reduce jitter from
  frequent re-querying.
- A good starting rule: `n_action_steps = chunk_size` (execute all predictions).

```bash
--policy.chunk_size=50 --policy.n_action_steps=50
```

At inference, `n_action_steps` can be tuned without retraining via `inference_tuning:` in the inference YAML:

```yaml
inference_tuning:
  n_action_steps: 50   # override checkpoint default at runtime
```

### kl_weight

Controls the VAE regularization strength. The default `kl_weight=10.0` works
well in most cases. If actions are too jerky or erratic, try increasing it
(e.g. 20–50). If the model is too conservative or underfits, reduce it.

### Temporal ensemble at inference

Instead of re-querying every `n_action_steps`, temporal ensemble averages
overlapping predictions for smoother execution. Enable it in the inference
YAML — no retraining needed:

```yaml
inference_tuning:
  temporal_ensemble_coeff: 0.01   # lower = more smoothing
  # n_action_steps is forced to 1 automatically when temporal_ensemble_coeff is set
```

### Image augmentation

Enable color jitter and affine augmentation to improve generalization to
lighting and viewpoint variation. Pass these in a train config YAML or via
`--dataset.image_transforms.enable=true`. An example config is saved
alongside every checkpoint in `train_config.json`.

### Camera selection

More cameras help but slow training and inference. Use `--camera-filter`
to ablate which cameras matter most for your task:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --camera-filter=chest,waist
```

### Delta actions

For tasks where the robot needs to return to similar poses repeatedly,
`--use-delta-actions` (action = target − current state) can make learning
easier. The flag is persisted to `anvil_config.json` in the checkpoint and
auto-read at inference — no manual inference YAML change needed.

```bash
uv run anvil-trainer ... --use-delta-actions
```

### Steps and batch size

100k steps with batch size 16 is a solid default. If your dataset is small
(< 50 episodes), 50k steps is often enough and avoids overfitting. Increase
batch size if GPU memory allows — it stabilizes training.

### Validation and Test Loss

Pass `--split-ratio=train,val,test` (default `8,1,1`) to hold out episodes for validation and testing.

- **Validation Loss (`val/loss`)**: Computed every `log_freq * 5` steps. Used to monitor overfitting during training.
- **Test Loss (`eval/test_loss`)**: Computed at every checkpoint (`save_freq`). This is a more thorough evaluation on a completely held-out set.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --split-ratio=8,1,1 \
  --wandb.enable=true
```

Use the checkpoint with the lowest test loss for deployment. A rising validation loss while train loss keeps falling is an early overfitting signal.

### Vision backbone

ACT uses ImageNet-pretrained ResNet18 by default. To switch backbone:

```bash
--backbone=resnet34   # or resnet50
```

VLA policies (Pi0 / Pi0.5 / SmolVLA) ignore this flag — they use their own
pre-trained vision encoder.

---

## Diffusion Policy

### When to use Diffusion vs ACT

Diffusion Policy models the action distribution as a denoising diffusion process rather than a deterministic regression. This makes it naturally suited for tasks where multiple valid trajectories exist (e.g. the robot can approach an object from several angles). It produces smooth, natural motions without explicit chunk tuning.

Trade-off: inference is slower than ACT because each step requires running a denoising loop (default 100 DDPM steps or 10 DDIM steps). If real-time latency is tight, try ACT first.

### Training command

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

### Steps and batch size

100k steps with batch size 64 is a solid default. Diffusion models benefit more from larger batch sizes than ACT — this reduces the variance of the score-matching objective and stabilizes training. On a 24 GB GPU with 3–4 cameras at full resolution, batch size 16–32 is the practical ceiling; use `--policy.resize_shape="[256,320]"` to shrink images before the backbone if you need to recover headroom for a larger batch.

If your dataset is small (< 50 episodes), 50k steps is often enough.

### Vision backbone

Same as ACT — ImageNet-pretrained ResNet18 is the default. Use `--backbone=resnet34` or `--backbone=resnet50` to switch. The `use_group_norm` flag is automatically disabled when a pretrained backbone is used (GroupNorm conversion is incompatible with pretrained BatchNorm weights).

### n_action_steps

Diffusion Policy predicts a full action chunk (default 16 steps) and executes all of them before re-running inference. If the resulting motion feels jerky or hesitant, tune `n_action_steps` at inference without retraining:

```yaml
# configs/lerobot_control/inference_default.yaml
inference_tuning:
  n_action_steps: 8   # execute fewer steps before re-querying
```

### num_inference_steps (DDPM vs DDIM)

The denoising loop runs `num_inference_steps` iterations per inference call. The default is 100 (DDPM), which is accurate but slow. Switching to DDIM with 10 steps gives similar quality at ~10× the speed:

```bash
--policy.num_inference_steps=10
```

### Image augmentation and camera selection

Same guidance as ACT applies — use `--dataset.image_transforms.enable=true` for color jitter and affine augmentation, and `--camera-filter` to drop cameras that don't contribute signal.

### Delta actions

`--use-delta-actions` is supported and can help for tasks requiring repeated returns to similar poses. See the [ACT section](#delta-actions) for details.

---

## SmolVLA

### Always start from pretrained weights

SmolVLA has two weight sources:

| Source | Flag | What it does |
|---|---|---|
| VLM backbone | `--policy.vlm_model_name` | Loaded automatically (`SmolVLM2-500M-Video-Instruct`) |
| SmolVLA base | `--policy.pretrained_path` | Full pretrained action expert — use this |

Always fine-tune from `lerobot/smolvla_base` rather than training from scratch:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

`--policy.load_vlm_weights=true` is required when loading from a SmolVLA
checkpoint. Without it, only the VLM backbone loads and the action expert
starts from random weights.

### Task description

SmolVLA is a language-conditioned policy. A clear, specific task description
improves performance significantly. Set it via `--task-description` at training
so every sample gets the same instruction:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="pick up the red block and stack it on the blue block"
```

Mirror the same string in the inference YAML:

```yaml
model:
  task_description: "pick up the red block and stack it on the blue block"
```

### Frozen layers (defaults are good)

By default, the vision encoder is frozen (`freeze_vision_encoder=true`) and
only the action expert is trained (`train_expert_only=true`). This is the
right setting for fine-tuning on a new task with limited data. Only unfreeze
the vision encoder if you have a very large dataset and the visual domain
differs significantly from the pretrained data.

### Steps

SmolVLA converges faster than ACT from a pretrained base. 30k–50k steps is
often sufficient. The default scheduler decays over 30k steps
(`scheduler_decay_steps=30000`) which aligns well with this range.

---

## Pi0

Pi0 uses a PaliGemma-3B backbone with a flow-matching action expert. It requires
HuggingFace access to `google/paligemma-3b-pt-224`.

### Always start from pretrained weights

Fine-tune from `lerobot/pi0_base` rather than training from scratch:

| Pretrained path | Description |
|---|---|
| `lerobot/pi0_base` | General-purpose base — use this for new tasks |
| `lerobot/pi0_libero` | Pre-trained on the Libero benchmark dataset |

### Training command

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
  --task-description="pick up the red block"
```

### Key parameters

| Flag | Default | Description |
|---|---|---|
| `--policy.pretrained_path` | — | Required — start from `lerobot/pi0_base` |
| `--policy.compile_model` | `false` | Enables torch.compile for faster training |
| `--policy.gradient_checkpointing` | `false` | Reduces VRAM usage significantly — always enable |
| `--policy.dtype` | `float32` | Use `bfloat16` for efficiency |
| `--policy.train_expert_only` | `false` | `true` = freeze VLM, train only action expert + projections — lower memory, faster convergence |
| `--policy.freeze_vision_encoder` | `false` | Only freeze if GPU memory is extremely tight |

### Task description

Pi0 is language-conditioned. Always pass `--task-description` at training and
mirror it in the inference YAML — the same string must be used at both stages.

### Steps

20k–50k steps from a pretrained base is a reasonable range. Pi0 is a
flow-matching model and benefits more from demonstration consistency than
raw episode count.

---

## Pi0.5

Pi0.5 is a larger variant (~4B params vs Pi0's ~3B) with stronger language
understanding. Training flags are identical to Pi0 but GPU memory requirements
are higher.

### Always start from pretrained weights

| Pretrained path | Description |
|---|---|
| `lerobot/pi05_base` | General-purpose base — use this for new tasks |
| `lerobot/pi05_libero` | Pre-trained on the Libero benchmark dataset |

### Training command

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
  --task-description="pick up the red block"
```

### Required flags on a 24 GB GPU

| Flag | Why |
|---|---|
| `--policy.dtype=bfloat16` | Halves VRAM — required to fit 4B model on 24 GB |
| `--policy.gradient_checkpointing=true` | Further reduces VRAM during backprop |
| `--batch_size=16` | Starting point — reduce if GPU OOM |
| `--num_workers=0` | Prevents CPU RAM OOM — forked workers each copy the full model into RAM |

### Normalization mapping

Pi0.5's default normalization is `QUANTILE10`, which requires `stats.json` to contain pre-computed quantile fields (`q01` / `q99`). Datasets converted with `mcap-convert` do not include these — only `mean`, `std`, `min`, and `max` are written.

There are two ways to resolve this:

**Option A — Override normalization (recommended for Anvil datasets)**

Pass `MEAN_STD` for actions and states, which uses the existing mean/std stats:

```bash
--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

This is the approach shown in the training command above and requires no changes to your dataset.

**Option B — Augment the dataset with quantile stats**

Computes and writes quantile fields directly into the dataset's `stats.json`:

```bash
uv run python -c "
from lerobot.datasets.v30.augment_dataset_quantile_stats import main
main()
" -- --repo-id=local/your-dataset
```

> **Warning: this modifies the dataset in-place.** Back up your dataset before running:
> ```bash
> cp -r data/datasets/my-dataset data/datasets/my-dataset.bak
> ```

After augmentation, you can use the default `QUANTILE10` normalization and omit `--policy.normalization_mapping`.

Option A is simpler and sufficient for most tasks. Choose Option B only if you need to reproduce results that rely specifically on quantile normalization.

---

## Offline Evaluation (anvil-eval)

After training a model, use `anvil-eval` to quantify its performance before robot deployment.

### Key Metrics
- **MAE/MSE**: Mean Absolute/Squared Error across all joints and frames.
- **Cosine Similarity**: Measures how well the predicted action vector aligns with the ground-truth direction.
- **Smoothness**: L2 norm of consecutive action deltas. High variance here indicates "jittery" model output.

### Analyzing Plots
- **Episode Plots**: Found in `plots/episode_NNNN_<split>.png`. Each joint has its own subplot.
  - **Reordering**: Joint names ending in `finger_joint1` (grippers) are moved to the end of the grid for easier comparison.
- **Summary Box Plot**: Found in `plots/summary_per_joint_mae.png`.
  - Shows the distribution of MAE across all evaluated episodes for each joint.
  - Useful for identifying which joints the model is most/least accurate on across different dataset splits.

### Evaluation Strategies
- Use `--split all --num-eps 5` to get a representative sample from across the entire dataset.
- High error on the `train` split indicates the model is underfitting (needs more training or more capacity).
- Low error on `train` but high error on `val`/`test` indicates the model is overfitting.
