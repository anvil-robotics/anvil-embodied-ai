# Training Tips

## MODEL_PATH — Point to a Specific Checkpoint

After training, checkpoints are saved under `output_dir/checkpoints/<step>/pretrained_model/`.
The `MODEL_PATH` in your `.env` must point all the way to the `pretrained_model` subdirectory:

```
# Correct
MODEL_PATH=/workspace/model_zoo/my-model/checkpoints/100000/pretrained_model

# Wrong — config.json not found
MODEL_PATH=/workspace/model_zoo/my-model
```

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

### kl_weight

Controls the VAE regularization strength. The default `kl_weight=10.0` works
well in most cases. If actions are too jerky or erratic, try increasing it
(e.g. 20–50). If the model is too conservative or underfits, reduce it.

### Temporal ensemble at inference

Instead of re-querying every `n_action_steps`, temporal ensemble averages
overlapping predictions for smoother execution. Enable it in your inference
YAML — no retraining needed:

```yaml
model:
  temporal_ensemble_coeff: 0.01   # lower = more smoothing
  n_action_steps: 1               # required when using temporal ensemble
```

### Image augmentation

Enable color jitter and affine augmentation to improve generalization to
lighting and viewpoint variation. Pass these in a train config YAML or via
`--dataset.image_transforms.enable=true`. An example config is saved
alongside every checkpoint in `train_config.json`.

### Camera selection

More cameras help but slow training and inference. Use `LEROBOT_CAMERA_FILTER`
to ablate which cameras matter most for your task:

```bash
LEROBOT_CAMERA_FILTER=chest,waist uv run lerobot-train ...
```

### Delta actions

For tasks where the robot needs to return to similar poses repeatedly,
`--use-delta-actions` (action = target − current state) can make learning
easier. Remember to enable `use_delta_actions: true` in your inference YAML
as well so actions are applied correctly at runtime.

### Steps and batch size

100k steps with batch size 16 is a solid default. If your dataset is small
(< 50 episodes), 50k steps is often enough and avoids overfitting. Increase
batch size if GPU memory allows — it stabilizes training.

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
uv run lerobot-train \
  --dataset.repo_id=local \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --output_dir=data/training-output \
  --eval_freq=0
```

`--policy.load_vlm_weights=true` is required when loading from a SmolVLA
checkpoint. Without it, only the VLM backbone loads and the action expert
starts from random weights.

### Task description

SmolVLA is a language-conditioned policy. A clear, specific task description
improves performance significantly. Set it via environment variable before
training so every sample gets the same instruction:

```bash
LEROBOT_TASK_OVERRIDE="pick up the red block and stack it on the blue block" \
uv run lerobot-train ...
```

And mirror the same string in your inference YAML:

```yaml
model:
  type: smolvla
  task_description: "pick up the red block and stack it on the blue block"
```

### Disable eval

SmolVLA evaluation requires a live robot or simulator. Set `eval_freq=0`
to skip it entirely:

```bash
--eval_freq=0
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
