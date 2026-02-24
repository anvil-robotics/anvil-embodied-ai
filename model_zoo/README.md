# Model Zoo

This directory stores trained model weights for inference. Contents are gitignored.

## Directory Structure

```
model_zoo/
├── <your-model-name>/
│   └── checkpoints/
│       └── last/
│           └── pretrained_model/
│               ├── config.json
│               └── model.safetensors
└── README.md
```

## Adding Models

### Manual Download

1. Visit the model page on HuggingFace Hub
2. Download the checkpoint directory containing:
   - `config.json` — model architecture parameters
   - `model.safetensors` — model weights
3. Place it under `model_zoo/<model-name>/`
4. Update `.env` with the container path:
   ```
   MODEL_PATH=/workspace/model_zoo/<model-name>/checkpoints/last/pretrained_model
   ```

## Model Format

Models should be in LeRobot-compatible format (output of `lerobot-train`):

- **config.json**: Model architecture, input/output dimensions, chunk size
- **model.safetensors**: Weights in safetensors format

## Training Your Own Models

See [Training Guide](../docs/training-guide.md) for instructions on training custom models.
