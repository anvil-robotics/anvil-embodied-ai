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
