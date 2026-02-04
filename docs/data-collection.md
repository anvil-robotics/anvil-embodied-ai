# Data Collection Guide

This guide covers how to collect high-quality robot demonstration data for training imitation learning models.

## Overview

Data collection is the foundation of imitation learning. You will:

1. Operate the robot via teleoperation to perform desired tasks
2. Record sensor data (camera images, joint states) to MCAP files
3. Organize recordings for conversion to training datasets

## Recording Format: MCAP

Anvil-Embodied-AI uses [MCAP](https://mcap.dev/) as the primary recording format. MCAP is an open-source container format designed for multimodal robotics data.

### Why MCAP?

- **Efficient storage**: Compressed, chunked storage for large recordings
- **Fast random access**: Seek to any timestamp without reading entire file
- **ROS2 native**: First-class support for ROS2 message types
- **Multi-sensor**: Stores images, joint states, and other data in sync
- **Self-describing**: Schema information embedded in file

### MCAP File Structure

A typical recording contains:

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera images (RGB) |
| `/joint_states` | `sensor_msgs/JointState` | Joint positions, velocities |
| `/gripper/state` | `sensor_msgs/JointState` | Gripper position |

## Recording Demonstrations

### Prerequisites

Ensure your robot is set up with:

- Camera(s) publishing to ROS2 topics
- Joint state publisher running
- Teleoperation interface ready

### Using ros2 bag

Record all topics to an MCAP file:

```bash
# Record all topics
ros2 bag record -a -s mcap -o data/raw/my-session/recording_001

# Record specific topics
ros2 bag record -s mcap \
  /camera/image_raw \
  /joint_states \
  /gripper/state \
  -o data/raw/my-session/recording_001
```

### Recording Session Structure

Organize recordings by session and task:

```
data/raw/
└── pick-and-place-session-01/
    ├── recording_001.mcap    # First demonstration
    ├── recording_002.mcap    # Second demonstration
    ├── recording_003.mcap    # Third demonstration
    └── ...
```

Each MCAP file represents one episode (demonstration) of the task.

### Recording Workflow

1. **Prepare the environment**: Set up objects in initial positions
2. **Start recording**: Begin ros2 bag record
3. **Perform task**: Teleoperate the robot through the task
4. **Stop recording**: Ctrl+C to end recording
5. **Reset environment**: Return objects to initial positions
6. **Repeat**: Record multiple demonstrations

## Inspecting Recordings

Use the `mcap-inspect` tool to examine your recordings:

```bash
# List topics and message counts
uv run mcap-inspect data/raw/my-session/recording_001.mcap

# Example output:
# Topics:
#   /camera/image_raw    (sensor_msgs/Image)      - 1500 messages
#   /joint_states        (sensor_msgs/JointState) - 3000 messages
#   /gripper/state       (sensor_msgs/JointState) - 3000 messages
# Duration: 50.0 seconds
```

### Using mcap CLI

The official mcap CLI tool provides additional inspection capabilities:

```bash
# Install mcap CLI
pip install mcap-cli

# Summary of file
mcap info data/raw/my-session/recording_001.mcap

# List topics
mcap topics data/raw/my-session/recording_001.mcap
```

## Best Practices for Data Quality

### Task Design

- **Consistent starting positions**: Begin each demonstration from similar initial conditions
- **Clear task boundaries**: Define clear start and end states
- **Varied demonstrations**: Include natural variation in trajectories
- **Reasonable speed**: Move at consistent, moderate speeds

### Recording Quality

1. **Sufficient data**: Aim for 50-100 demonstrations per task
2. **Complete episodes**: Ensure each recording captures the full task
3. **Clean data**: Avoid pausing, teleop errors, or interruptions mid-recording
4. **Consistent lighting**: Maintain stable lighting conditions

### Camera Guidelines

- **Resolution**: 640x480 recommended (balance quality vs. processing speed)
- **Frame rate**: 30 Hz typical for cameras
- **Focus**: Ensure cameras are properly focused on workspace
- **Occlusion**: Minimize occlusion of objects and gripper

### Joint State Guidelines

- **Frequency**: 30-100 Hz recommended
- **All joints**: Include all controlled joints
- **Calibration**: Verify joint encoders are calibrated

## Validating Recordings

Before conversion, validate your recordings:

```bash
# Check recording integrity
uv run mcap-validate data/raw/my-session/

# Validates:
# - File format integrity
# - Required topics present
# - Message frequency consistency
# - Timestamp synchronization
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing topics | mcap-inspect shows fewer topics than expected | Check ROS2 node is publishing |
| Low frame rate | Fewer messages than expected for duration | Check sensor configuration |
| Time jumps | Large gaps in timestamps | Check system clock, avoid pauses |
| Corrupted file | mcap-inspect fails | Re-record, check disk space |

## Data Organization

### Naming Conventions

Use descriptive names for sessions:

```
data/raw/
├── pick-cube-v1/           # Task name and version
│   ├── recording_001.mcap
│   └── recording_002.mcap
├── place-cup-v1/
│   └── ...
└── bimanual-handover-v1/
    └── ...
```

### Metadata

Consider creating a `metadata.json` in each session folder:

```json
{
  "task": "pick-and-place",
  "robot": "openarm_bimanual",
  "date": "2024-01-15",
  "operator": "john_doe",
  "notes": "Picking red cubes from table to bin"
}
```

## Next Steps

Once you have collected recordings:

1. **Convert to LeRobot format**: See [Training Guide - Data Conversion](training-guide.md#converting-mcap-to-lerobot-format)
2. **Train a model**: See [Training Guide](training-guide.md)
3. **Deploy for inference**: See [Deployment Guide](deployment-guide.md)
