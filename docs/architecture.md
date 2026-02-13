# Architecture Overview

## System Overview

```
+------------------+     +-------------------+     +-------------------+
|                  |     |                   |     |                   |
|  Teleoperation   |---->|  mcap_converter   |---->|  lerobot_training |
|  (ROS2 Bag)      |     |                   |     |                   |
|                  |     |                   |     |                   |
+------------------+     +-------------------+     +-------------------+
        |                        |                        |
        v                        v                        v
   MCAP Files              LeRobot Dataset          Model Weights
   (raw recordings)        (Parquet + MP4)          (SafeTensors)
```

## Data Flow

### Stage 1: Data Collection

- **Input**: Robot sensors (cameras, joint encoders) via teleoperation
- **Output**: MCAP files containing timestamped ROS2 messages
- **Tool**: `ros2 bag record`

### Stage 2: Data Conversion

- **Input**: Raw MCAP recordings + robot configuration YAML
- **Output**: LeRobot v3.0 format dataset (Parquet + MP4)
- **Tool**: `mcap-convert`

```
MCAP File → McapReader → BufferedStreamExtractor → LeRobotWriter → LeRobot Dataset
```

### Stage 3: Model Training

- **Input**: LeRobot dataset + training configuration
- **Output**: Trained model weights (SafeTensors)
- **Tool**: `lerobot-train`

## Components

### mcap_converter

Converts ROS2 MCAP recordings to LeRobot dataset format.

**Location**: `packages/mcap_converter/`

| Module | Purpose |
|--------|---------|
| `McapReader` | Read and parse MCAP files |
| `BufferedStreamExtractor` | Memory-efficient streaming extraction |
| `LeRobotWriter` | Write data in LeRobot v3.0 format |
| `ConfigLoader` | Load robot configuration from YAML |

| CLI | Purpose |
|-----|---------|
| `mcap-convert` | Convert MCAP to LeRobot dataset |
| `mcap-inspect` | Inspect MCAP file structure and topics |
| `mcap-to-video` | Extract MCAP image topics to MP4 |
| `dataset-validate` | Validate converted dataset integrity |
| `mcap-upload` | Upload dataset to HuggingFace Hub |

### lerobot_training

Custom training utilities extending the LeRobot framework with pluggable transforms.

**Location**: `packages/lerobot_training/`

| Component | Purpose |
|-----------|---------|
| `TrainingConfig` | Configuration for training options |
| `CameraFilterTransform` | Filter dataset to specific cameras |
| `TaskOverrideTransform` | Override task string for VLA models |
| `DeltaActionTransform` | Convert to relative actions |
| `TransformRunner` | Orchestrate transform application |

## Configuration

```
configs/
├── mcap_converter/          # Robot-specific MCAP conversion
│   └── openarm_bimanual.yaml
└── lerobot_training/        # Training customization
    └── act_default.yaml
```

See each package README for configuration details:
- [mcap_converter configuration](../packages/mcap_converter/README.md#configuration)
- [lerobot_training configuration](../packages/lerobot_training/README.md#configuration)

## Dependencies

| Package | Purpose |
|---------|---------|
| `lerobot` | Core imitation learning framework |
| `mcap` | MCAP file format library |
| `mcap-ros2-support` | ROS2 message support for MCAP |
| `torch` | Deep learning framework |
| `wandb` | Experiment tracking |
| `huggingface-hub` | Model/dataset hosting |
