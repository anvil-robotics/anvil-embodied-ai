# Deployment Guide

This guide covers deploying trained LeRobot models for real-time inference on Anvil robot platforms.

## Overview

Deployment involves:

1. Building the Docker inference container
2. Configuring the inference node
3. Running inference alongside your robot's ROS2 stack
4. Tuning safety parameters

## Docker Setup

### Prerequisites

- Docker Engine installed and running
- Access to trained model weights (local or HuggingFace Hub)
- ROS2 network configured for your robot

### Directory Structure

```
docker/inference/
├── Dockerfile          # Container build instructions
├── entrypoint.sh       # Container startup script
└── docker-compose.yml  # Orchestration configuration
```

## Building the Docker Image

### Build Command

```bash
# From repository root
docker build -t anvil-inference:latest -f docker/inference/Dockerfile .
```

### Build Options

```bash
# Build with custom tag
docker build -t anvil-inference:v1.0.0 -f docker/inference/Dockerfile .

# Build with no cache (fresh build)
docker build --no-cache -t anvil-inference:latest -f docker/inference/Dockerfile .
```

### Dockerfile Overview

The inference container:

1. Uses ROS2 Jazzy as the base image
2. Installs Python dependencies via uv
3. Copies the `lerobot_control` ROS2 package
4. Builds the ROS2 workspace with colcon
5. Sets up entrypoint to source ROS2 environment

```dockerfile
FROM ros:jazzy-ros-base

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv git curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy and build workspace
WORKDIR /workspace
COPY ros2/src/lerobot_control /workspace/src/lerobot_control
COPY configs/lerobot_control /workspace/configs

RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
```

## Running the Inference Container

### Using Docker Compose (Recommended)

Create or modify `docker-compose.yml`:

```yaml
version: '3.8'

services:
  inference:
    image: anvil-inference:latest
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=0
      - MODEL_PATH=/models/my-model
    volumes:
      - ./model_zoo:/models:ro
      - ./configs/lerobot_control:/workspace/configs:ro
    devices:
      - /dev/dri:/dev/dri  # GPU access (if needed)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Start the container:

```bash
docker compose -f docker/inference/docker-compose.yml up
```

### Using Docker Run

```bash
docker run --rm -it \
  --network host \
  -e ROS_DOMAIN_ID=0 \
  -e MODEL_PATH=/models/my-model \
  -v $(pwd)/model_zoo:/models:ro \
  -v $(pwd)/configs/lerobot_control:/workspace/configs:ro \
  anvil-inference:latest
```

### GPU Support

For GPU-accelerated inference:

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  -e MODEL_PATH=/models/my-model \
  -v $(pwd)/model_zoo:/models:ro \
  anvil-inference:latest
```

## Configuration Options

### Inference Node Parameters

The inference node accepts ROS2 parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_path` | string | Path to model directory | Required |
| `config_file` | string | Path to inference config YAML | None |
| `mode` | string | `mp` (multi-process) or `single` | `mp` |
| `inference_rate` | float | Target inference frequency (Hz) | 30.0 |
| `action_smoothing` | bool | Enable action smoothing | true |

### Running with Parameters

```bash
ros2 run lerobot_control inference_node \
  --ros-args \
  -p model_path:=/models/my-model \
  -p config_file:=/workspace/configs/inference.yaml \
  -p mode:=mp \
  -p inference_rate:=30.0
```

### Inference Configuration File

Create an inference configuration file:

```yaml
# configs/lerobot_control/inference.yaml

# Model settings
model:
  chunk_size: 100
  n_action_steps: 50
  device: cuda  # or cpu

# Input configuration
observation:
  cameras:
    cam_wrist:
      topic: /camera/wrist/image_raw
      resize: [224, 224]
  joint_state:
    topic: /joint_states
    joints:
      - joint_1
      - joint_2
      - joint_3
      - joint_4
      - joint_5
      - joint_6
      - gripper

# Output configuration
action:
  topic: /actions
  joints:
    - joint_1
    - joint_2
    - joint_3
    - joint_4
    - joint_5
    - joint_6
    - gripper

# Safety limits
safety:
  position_limits:
    joint_1: [-3.14, 3.14]
    joint_2: [-2.0, 2.0]
    # ... other joints
  velocity_limit: 1.0  # rad/s
  acceleration_limit: 2.0  # rad/s^2
```

## Safety Parameters

### Action Limiting

The `ActionLimiter` class enforces safety constraints on predicted actions:

```python
# In lerobot_control
from lerobot_control import ActionLimiter

limiter = ActionLimiter(
    position_limits={
        'joint_1': (-3.14, 3.14),
        'joint_2': (-2.0, 2.0),
    },
    velocity_limit=1.0,
    acceleration_limit=2.0,
)

safe_action = limiter.limit(predicted_action, current_state, dt)
```

### Configuration in YAML

```yaml
# Safety limits configuration
safety:
  # Joint position limits (radians)
  position_limits:
    joint_1: [-3.14, 3.14]
    joint_2: [-1.57, 1.57]
    joint_3: [-3.14, 3.14]
    joint_4: [-1.57, 1.57]
    joint_5: [-3.14, 3.14]
    joint_6: [-1.57, 1.57]
    gripper: [0.0, 0.08]

  # Maximum velocity (rad/s)
  velocity_limit: 1.0

  # Maximum acceleration (rad/s^2)
  acceleration_limit: 2.0

  # Emergency stop threshold
  force_limit: 50.0  # Newtons
```

### Safety Best Practices

1. **Start slow**: Begin with conservative velocity limits
2. **Test thoroughly**: Validate limits with mock controller first
3. **Monitor metrics**: Watch for action clipping frequency
4. **Gradual increase**: Increase limits gradually as confidence grows

## ROS2 Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/*/image_raw` | `sensor_msgs/Image` | Camera images |
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/actions` | `std_msgs/Float64MultiArray` | Predicted actions |
| `/inference/status` | `std_msgs/String` | Inference status |
| `/inference/metrics` | `std_msgs/String` | Performance metrics (JSON) |

## Troubleshooting

### Common Issues

#### Container fails to start

**Symptom**: Container exits immediately

**Solutions**:
1. Check logs: `docker logs <container_id>`
2. Verify model path exists and is mounted correctly
3. Ensure ROS2 environment variables are set

```bash
# Debug interactively
docker run --rm -it \
  --entrypoint /bin/bash \
  anvil-inference:latest
```

#### No actions published

**Symptom**: `/actions` topic has no messages

**Solutions**:
1. Verify input topics are publishing:
   ```bash
   ros2 topic echo /camera/wrist/image_raw --once
   ros2 topic echo /joint_states --once
   ```
2. Check topic names match configuration
3. Verify model loaded successfully

#### High latency

**Symptom**: Inference slower than target rate

**Solutions**:
1. Enable GPU acceleration
2. Reduce image resolution in config
3. Use multi-process mode (`mode:=mp`)
4. Check CPU/GPU utilization

```bash
# Monitor performance
ros2 topic hz /actions
```

#### Out of memory

**Symptom**: Container killed with OOM error

**Solutions**:
1. Reduce batch size
2. Use smaller model
3. Increase container memory limit
4. Use CPU inference if GPU memory insufficient

#### ROS2 communication issues

**Symptom**: Topics not visible across network

**Solutions**:
1. Verify `ROS_DOMAIN_ID` matches
2. Use `--network host` in Docker
3. Check firewall settings
4. Verify DDS discovery working

```bash
# Test ROS2 communication
ros2 topic list
ros2 node list
```

### Debug Mode

Run in single-process mode for easier debugging:

```bash
ros2 run lerobot_control inference_node \
  --ros-args \
  -p model_path:=/models/my-model \
  -p mode:=single
```

### Logging

Increase logging verbosity:

```bash
ros2 run lerobot_control inference_node \
  --ros-args \
  --log-level debug
```

## Testing with Mock Controller

Before deploying on real hardware, test with the mock controller:

```bash
# Terminal 1: Run mock controller
ros2 run lerobot_control mock_controller_node \
  --ros-args \
  -p timeout:=60.0 \
  -p required_actions:=100

# Terminal 2: Run inference
ros2 run lerobot_control inference_node \
  --ros-args \
  -p model_path:=/models/my-model
```

The mock controller:
- Publishes simulated camera images and joint states
- Subscribes to actions and validates them
- Exits with success after receiving required valid actions

## Production Deployment

### Health Monitoring

Monitor inference health in production:

```bash
# Check node is running
ros2 node list | grep inference

# Monitor action rate
ros2 topic hz /actions

# Check latency metrics
ros2 topic echo /inference/metrics
```

### Logging and Metrics

Configure logging for production:

```yaml
# In inference.yaml
logging:
  level: info
  file: /var/log/lerobot_inference.log
  max_size: 100MB
  backup_count: 5

metrics:
  enable: true
  prometheus_port: 9090
```

### Container Management

Use systemd or similar for container lifecycle:

```bash
# Example systemd service
[Unit]
Description=LeRobot Inference Container
After=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker compose -f /path/to/docker-compose.yml up
ExecStop=/usr/bin/docker compose -f /path/to/docker-compose.yml down

[Install]
WantedBy=multi-user.target
```

## Next Steps

- [Architecture Overview](architecture.md) - Understand system design
- [Extending to New Robots](extending-to-new-robots.md) - Adapt for your robot
- [Data Collection Guide](data-collection.md) - Collect more training data
