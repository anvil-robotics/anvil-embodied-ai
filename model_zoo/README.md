# Model Zoo

This directory stores trained model weights for inference.

## Directory Structure

```
model_zoo/
├── test/                    # Dummy model for CI testing
│   └── dummy_model/
├── <your-model-name>/       # Your trained models
│   ├── config.json
│   ├── model.safetensors
│   └── inference.yaml
└── README.md
```

## Downloading Models

### From HuggingFace Hub

Use the download script to fetch models from HuggingFace:

```bash
./scripts/download-models.sh <huggingface-repo-id> <model-name>
```

Example:
```bash
./scripts/download-models.sh anvil-robotics/openarm-act-v1 openarm-act-v1
```

### Manual Download

1. Visit the model page on HuggingFace Hub
2. Download the following files:
   - `config.json` - Model configuration
   - `model.safetensors` - Model weights
3. Create inference configuration `inference.yaml` (see examples in `configs/lerobot_control/`)

## Model Format

Models should be in LeRobot-compatible format:

- **config.json**: Contains model architecture parameters, input/output dimensions
- **model.safetensors**: Model weights in safetensors format
- **inference.yaml**: (Optional) Inference parameters specific to your robot setup

## Training Your Own Models

See [Training Guide](../docs/training-guide.md) for instructions on training custom models.
