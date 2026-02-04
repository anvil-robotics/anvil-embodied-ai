# Architecture Overview

This document describes the system architecture of the Anvil-Embodied-AI platform.

## System Overview

```
+------------------+     +-------------------+     +-------------------+     +-------------------+
|                  |     |                   |     |                   |     |                   |
|  Teleoperation   |---->|  mcap_converter   |---->|  lerobot_training |---->|  lerobot_control  |
|  (ROS2 Bag)      |     |                   |     |                   |     |  (ROS2 Node)      |
|                  |     |                   |     |                   |     |                   |
+------------------+     +-------------------+     +-------------------+     +-------------------+
        |                        |                        |                        |
        v                        v                        v                        v
   MCAP Files              LeRobot Dataset          Model Weights            Robot Actions
   (raw recordings)        (Parquet + MP4)          (SafeTensors)           (ROS2 Topics)
```

## Data Flow

The platform follows a four-stage data pipeline:

### Stage 1: Data Collection

```
+-------------+     +----------------+     +----------------+
|   Robot     |---->|  ROS2 Topics   |---->|   MCAP File    |
|  Hardware   |     |  (Sensors)     |     |  (Recording)   |
+-------------+     +----------------+     +----------------+
      ^
      |
  Teleoperation
```

- **Input**: Robot sensors (cameras, joint encoders)
- **Output**: MCAP files containing timestamped ROS2 messages
- **Tool**: `ros2 bag record`

### Stage 2: Data Conversion

```
+----------------+     +-------------------+     +-------------------+
|   MCAP Files   |---->|   mcap_converter  |---->|  LeRobot Dataset  |
|                |     |                   |     |                   |
+----------------+     +-------------------+     +-------------------+
                              |
                              v
                       Robot Config
                        (YAML)
```

- **Input**: Raw MCAP recordings + robot configuration
- **Output**: LeRobot v3.0 format dataset
- **Tool**: `mcap-convert`

### Stage 3: Model Training

```
+-------------------+     +-------------------+     +------------------+
|  LeRobot Dataset  |---->|  lerobot_training |---->|  Model Weights   |
|                   |     |                   |     |  (SafeTensors)   |
+-------------------+     +-------------------+     +------------------+
                                 |
                                 v
                          Training Config
                           (YAML/CLI)
```

- **Input**: LeRobot dataset + training configuration
- **Output**: Trained model weights
- **Tool**: `lerobot-train`

### Stage 4: Inference Deployment

```
+------------------+     +-------------------+     +----------------+
|  Model Weights   |---->|  lerobot_control  |---->|  Robot Actions |
|                  |     |  (ROS2 Node)      |     |  (/actions)    |
+------------------+     +-------------------+     +----------------+
                                 ^
                                 |
                         +---------------+
                         | Observations  |
                         | (Cameras,     |
                         |  Joint State) |
                         +---------------+
```

- **Input**: Model weights + real-time sensor data
- **Output**: Robot actions (joint commands)
- **Tool**: `ros2 run lerobot_control inference_node`

## Component Descriptions

### mcap_converter

The `mcap_converter` package handles conversion from ROS2 MCAP recordings to LeRobot dataset format.

**Location**: `packages/mcap_converter/`

**Key Modules**:

| Module | Purpose |
|--------|---------|
| `McapReader` | Read and parse MCAP files |
| `DataExtractor` | Extract images and joint states from ROS2 messages |
| `TimeAligner` | Synchronize data across different sensors |
| `LeRobotWriter` | Write data in LeRobot v3.0 format |
| `ConfigLoader` | Load robot configuration from YAML |

**CLI Tools**:

| Command | Purpose |
|---------|---------|
| `mcap-convert` | Convert MCAP to LeRobot dataset |
| `mcap-inspect` | Inspect MCAP file contents |
| `mcap-validate` | Validate dataset integrity |
| `mcap-upload` | Upload dataset to HuggingFace Hub |

**Data Flow**:

```
MCAP File
    |
    v
McapReader.read_messages()
    |
    v
DataExtractor.extract()
    |
    +---> Images (PNG/JPEG)
    |
    +---> Joint States (float arrays)
    |
    v
TimeAligner.align()
    |
    v
LeRobotWriter.write()
    |
    v
LeRobot Dataset (Parquet + MP4)
```

### lerobot_training

The `lerobot_training` package provides custom training utilities that extend the LeRobot framework.

**Location**: `packages/lerobot_training/`

**Key Components**:

| Component | Purpose |
|-----------|---------|
| `TrainingConfig` | Configuration dataclass for training options |
| `Transform` | Abstract base class for dataset transforms |
| `CameraFilterTransform` | Filter dataset to specific cameras |
| `TaskOverrideTransform` | Override task string for VLA models |
| `DeltaActionTransform` | Convert to relative actions |
| `TransformRunner` | Orchestrate transform application |

**Transform Architecture**:

```
TrainingConfig
      |
      v
TransformRunner
      |
      +---> apply_metadata_patches()  # Before lerobot import
      |
      +---> apply_dataset_patches()   # Patch LeRobotDataset.__getitem__
      |
      v
lerobot_train()                       # Standard LeRobot training
```

**Extension Points**:

Adding a new transform:

1. Create a subclass of `Transform`
2. Implement `name`, `is_enabled()`, and `apply()`
3. Optionally implement `patch_metadata()`
4. Register in `TransformRunner.TRANSFORMS`

### lerobot_control

The `lerobot_control` package is a ROS2 package for real-time model inference.

**Location**: `ros2/src/lerobot_control/`

**Key Components**:

| Component | Purpose |
|-----------|---------|
| `ModelLoader` | Load and manage LeRobot policy models |
| `ObservationManager` | Aggregate observations from ROS2 topics |
| `ImageConverter` | Convert ROS2 Image messages to tensors |
| `ActionLimiter` | Enforce safety limits on predicted actions |
| `MetricsTracker` | Track inference performance metrics |
| `SharedImageBuffer` | Shared memory for multi-process architecture |

**Inference Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `mp` (Multi-Process) | Separate processes for ROS2 and inference | Production (better isolation) |
| `single` (Single-Process) | Single process with threading | Debugging (simpler) |

**ROS2 Node Architecture**:

```
                    +---------------------------+
                    |     InferenceNode         |
                    |---------------------------|
                    |                           |
  /camera/*  ------>|  ObservationManager       |
                    |     |                     |
  /joint_states --->|     v                     |
                    |  ImageConverter           |
                    |     |                     |
                    |     v                     |
                    |  ModelLoader.infer()      |
                    |     |                     |
                    |     v                     |
                    |  ActionLimiter.limit()    |
                    |     |                     |
                    +-----|---------------------+
                          |
                          v
                      /actions
```

## Configuration System

### Configuration Hierarchy

```
configs/
├── mcap_converter/       # Robot-specific MCAP conversion
│   └── openarm_bimanual.yaml
├── lerobot_training/     # Training customization
│   └── act_config.yaml
└── lerobot_control/      # Inference configuration
    └── inference.yaml
```

### Configuration Schema

**mcap_converter Configuration**:

```yaml
robot_name: string
fps: int
cameras:
  <camera_name>:
    topic: string
    resolution: [width, height]
joint_state:
  topic: string
  joints: [string]
action_joints: [string]
```

**lerobot_training Configuration**:

```yaml
cameras: [string] | null
task_override: string | null
use_delta_actions: bool
```

**lerobot_control Configuration**:

```yaml
model:
  chunk_size: int
  n_action_steps: int
  device: string
observation:
  cameras: dict
  joint_state: dict
action:
  topic: string
  joints: [string]
safety:
  position_limits: dict
  velocity_limit: float
  acceleration_limit: float
```

## Extension Points

### Adding New Robot Support

1. **MCAP Converter Config**: Create `configs/mcap_converter/<robot>.yaml`
2. **Inference Config**: Create `configs/lerobot_control/<robot>.yaml`
3. **Test with Mock Controller**: Validate topic mapping

See: [Extending to New Robots](extending-to-new-robots.md)

### Adding New Training Transforms

1. Subclass `Transform` in `lerobot_training/train.py`
2. Implement required methods
3. Add configuration fields to `TrainingConfig`
4. Register in `TransformRunner.TRANSFORMS`

### Adding New Inference Strategies

1. Create strategy class in `lerobot_control/strategies/`
2. Implement inference loop
3. Register strategy in inference node

## Dependencies

### Python Packages

| Package | Purpose |
|---------|---------|
| `lerobot` | Core imitation learning framework |
| `mcap` | MCAP file format library |
| `mcap-ros2-support` | ROS2 message support for MCAP |
| `torch` | Deep learning framework |
| `wandb` | Experiment tracking |
| `huggingface-hub` | Model/dataset hosting |

### ROS2 Packages

| Package | Purpose |
|---------|---------|
| `rclpy` | ROS2 Python client library |
| `sensor_msgs` | Standard sensor message types |
| `std_msgs` | Standard message types |
| `control_msgs` | Control message types |

## Performance Considerations

### Data Pipeline

- **MCAP reading**: Chunked reading for memory efficiency
- **Video encoding**: Hardware-accelerated when available
- **Dataset storage**: Parquet for efficient columnar storage

### Training

- **Data loading**: Multi-worker data loading
- **GPU utilization**: Mixed precision training supported
- **Memory management**: Gradient checkpointing for large models

### Inference

- **Multi-process**: Isolate ROS2 and ML inference
- **Shared memory**: Zero-copy image transfer
- **Action chunking**: Amortize inference cost over multiple timesteps

## Security Considerations

- **Model isolation**: Inference runs in Docker container
- **Action limiting**: Hardware-enforced safety limits
- **Network isolation**: ROS2 domain IDs for segmentation
- **Credential management**: HuggingFace tokens via environment variables
