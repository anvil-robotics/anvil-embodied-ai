# Training Tips

## anvil-trainer Defaults

`anvil-trainer` is a thin wrapper around LeRobot's `lerobot-train` CLI. In addition to Anvil-specific flags (`--task-description`, `--camera-filter`, `--use-delta-actions`), it injects the following LeRobot defaults automatically so you don't have to repeat them in every command:

| Injected flag | Value | Reason |
|---|---|---|
| `--dataset.repo_id` | `local` | Anvil datasets are always local; HuggingFace Hub upload is not needed for training |
| `--policy.push_to_hub` | `false` | Prevents accidental upload of checkpoints to HuggingFace Hub |
| `--eval_freq` | `0` | LeRobot's default (20 000 steps) would attempt to launch a gym simulation environment, which doesn't exist for Anvil MCAP datasets |
| `--output_dir` | `model_zoo/<job_name>` | Resolved from `--job_name`; auto-generated from policy type + timestamp if omitted |
| `--wandb.project` | `anvil` | Default W&B project name — all runs land in the same project; `--job_name` becomes the run name |

Any of these can be overridden by passing the flag explicitly.

---

## MODEL_PATH — Point to a Specific Checkpoint

After training, checkpoints are saved under `model_zoo/<job_name>/checkpoints/<step>/pretrained_model/`.
The `MODEL_PATH` in your `.env` must point all the way to the `pretrained_model` subdirectory:

```
# Correct
MODEL_PATH=/workspace/model_zoo/grabbing-w1/checkpoints/100000/pretrained_model

# Wrong — config.json not found
MODEL_PATH=/workspace/model_zoo/grabbing-w1
```

---

## Visualizing Training Progress (Weights & Biases)

LeRobot uses [Weights & Biases](https://wandb.ai) for training monitoring. Enable it by passing `--wandb.enable=true`:

```bash
uv run wandb login   # one-time setup

uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --job_name=grabbing-w1 \
  --wandb.enable=true \
  --wandb.project=anvil
```

**`--job_name` becomes the W&B run name.** LeRobot passes `job_name` directly to `wandb.init(name=job_name)`, so each training run appears in the W&B dashboard under its job name. Set `--wandb.project` to group runs from the same robot/task together.

Key metrics to watch on the W&B dashboard:

| Metric | What it tells you |
|---|---|
| `train/loss` | Overall training loss — should decrease steadily |
| `train/grad_norm` | Gradient norm — spikes indicate instability, try lowering LR |
| `eval/avg_sum_rewards` | Task success (if eval env available) |

If you don't want W&B, training still runs fine without it — logs are printed to console.

---

## save_freq

Controls how often a checkpoint is saved (in steps). Each checkpoint writes the full model to `model_zoo/<job_name>/checkpoints/<step>/pretrained_model/`.

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
  --job_name=grabbing-w1 \
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
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --job_name=pick-and-place
```

### Steps and batch size

100k steps with batch size 64 is a solid default. Diffusion models benefit more from larger batch sizes than ACT — this reduces the variance of the score-matching objective and stabilizes training. If GPU memory is limited, batch size 32 is acceptable.

If your dataset is small (< 50 episodes), 50k steps is often enough.

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
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --job_name=grabbing-w1
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
  --job_name=grabbing-w1 \
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
  --policy.freeze_vision_encoder=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --job_name=grabbing-pi0 \
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
  --policy.freeze_vision_encoder=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=16 \
  --num_workers=0 \
  --job_name=grabbing-pi05 \
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
