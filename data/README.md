# Data Directory

Data files are gitignored. Record your own demonstrations or download from HuggingFace Hub.

## Directory Structure

```
data/
├── raw/                     # Raw MCAP recordings from teleoperation
│   └── <session-name>/
│       ├── recording_001.mcap
│       ├── recording_002.mcap
│       └── ...
└── datasets/                # Converted LeRobot datasets
    └── <dataset-name>/
        ├── meta/
        │   ├── info.json
        │   ├── episodes.jsonl
        │   └── tasks.jsonl
        ├── data/
        │   └── chunk-000/
        │       └── episode_000000.parquet
        └── videos/
            └── chunk-000/
                └── observation.images.<camera>/
```

## Workflow

1. Record teleoperation sessions → `data/raw/`
2. Convert with `mcap-convert` → `data/datasets/`
3. Validate with `dataset-validate`
4. Train with `lerobot-train`

See [Data Collection Guide](../docs/data-collection.md) for recording instructions.
