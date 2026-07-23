# anvil-eval

Anvil offline model evaluation — replay dataset episodes through trained policies.

## Usage

```bash
uv run anvil-eval \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --dataset <PATH_TO_DATASET> \
  --split val \
  --num-eps 5
```

## Pi0.5 sanity test

Use chunk-correct replay for Pi0.5 checkpoints whose processors convert actions
relative to the captured observation state:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run anvil-sanity \
  --checkpoint <CHECKPOINT_STEP> \
  --dataset <LEROBOT_DATASET> \
  --inference-config <LIVE_INFERENCE_YAML> \
  --all-episodes \
  --debug-image-dir debug_images
```

The command audits feature/camera/order contracts, compares the policy with a
hold-position baseline, simulates the synchronous prefetch queue, and writes
arrays, plots, `sanity_report.json`, and `GO_NO_GO.md` under the checkpoint's
default evaluation directory. It never publishes robot commands.
