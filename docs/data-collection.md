# Data Collection Guide

How to collect robot demonstration data for training imitation learning models.

## Recording Format: MCAP

[MCAP](https://mcap.dev/) is an open-source container format for multimodal robotics data with efficient storage, fast random access, and native ROS2 support.

A typical recording contains:

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera images (RGB) |
| `/joint_states` | `sensor_msgs/JointState` | Joint positions, velocities |

## Recording Demonstrations

### Using ros2 bag

```bash
# Record all topics
ros2 bag record -a -s mcap -o data/raw/my-session/recording_001

# Record specific topics
ros2 bag record -s mcap \
  /camera/image_raw \
  /joint_states \
  -o data/raw/my-session/recording_001
```

### Session Structure

Organize recordings by session. Each MCAP file is one episode (demonstration):

```
data/raw/
└── pick-and-place-v1/
    ├── recording_001.mcap
    ├── recording_002.mcap
    └── ...
```

### Recording Workflow

1. Set up objects in initial positions
2. Start `ros2 bag record`
3. Teleoperate the robot through the task
4. Ctrl+C to stop recording
5. Reset environment and repeat

## Inspecting Recordings

```bash
uv run mcap-inspect data/raw/my-session/recording_001.mcap
```

## Best Practices

### Data Quality

- **50-100 demonstrations** per task recommended
- Complete episodes only — avoid pauses or interruptions
- Include natural variation in trajectories
- Consistent, moderate speeds

### Camera

- 640x480 resolution recommended
- 30 Hz frame rate
- Stable lighting, minimize occlusion

### Joint State

- 30-100 Hz recommended
- Include all controlled joints
- Verify encoder calibration

## Next Steps

1. [Convert to LeRobot format](training-guide.md)
2. [Train a model](training-guide.md#step-3-train)
