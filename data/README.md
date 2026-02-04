# Data Directory

This directory stores robot demonstration data for training imitation learning models.

**Note:** Data files are gitignored. Each user should either:
1. Record their own demonstrations
2. Download datasets from HuggingFace Hub

## Directory Structure

```
data/
├── raw/                     # Raw MCAP recordings from teleoperation
│   └── <session-name>/
│       ├── recording_001.mcap
│       ├── recording_002.mcap
│       └── ...
├── datasets/                # Processed LeRobot datasets
│   └── <dataset-name>/
│       ├── meta/
│       │   ├── info.json
│       │   ├── episodes.jsonl
│       │   └── tasks.jsonl
│       ├── data/
│       │   └── chunk-000/
│       │       └── episode_000000.parquet
│       └── videos/
│           └── chunk-000/
│               └── observation.images.cam_wrist/
└── README.md
```

## Recording Data

See [Data Collection Guide](../docs/data-collection.md) for instructions on recording teleoperation sessions.

## Converting Data

Convert raw MCAP recordings to LeRobot format:

```bash
uv run mcap-convert \
  --input data/raw/my-session \
  --output data/datasets/my-dataset \
  --config configs/mcap_converter/openarm_bimanual.yaml
```

## Downloading Datasets

### From HuggingFace Hub

```bash
# Using huggingface-cli
huggingface-cli download anvil-robotics/openarm-demos --local-dir data/datasets/openarm-demos

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download("anvil-robotics/openarm-demos", local_dir="data/datasets/openarm-demos")
```

## Validating Datasets

Verify dataset integrity before training:

```bash
uv run mcap-validate --root data/datasets/my-dataset
```
