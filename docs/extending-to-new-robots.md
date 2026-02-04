# Extending to New Robots

This guide explains how to adapt the Anvil-Embodied-AI platform to support new robot configurations.

## Overview

Supporting a new robot requires:

1. Creating a robot configuration file for MCAP conversion
2. Defining topic mappings for cameras and joints
3. Creating an inference configuration
4. Testing with the mock controller

## Creating Robot Configuration Files

### MCAP Converter Configuration

Create a new configuration file in `configs/mcap_converter/`:

```yaml
# configs/mcap_converter/my_robot.yaml

# Robot identifier
robot_name: my_robot

# Target frame rate for dataset
fps: 30

# Camera configuration
cameras:
  # Key is the camera name used in dataset (observation.images.<name>)
  cam_wrist:
    # ROS2 topic for this camera
    topic: /my_robot/camera/wrist/image_raw
    # Output resolution [width, height]
    resolution: [640, 480]
    # Encoding (optional, auto-detected)
    encoding: rgb8

  cam_overhead:
    topic: /my_robot/camera/overhead/image_raw
    resolution: [640, 480]

# Joint state configuration
joint_state:
  # ROS2 topic for joint states
  topic: /my_robot/joint_states

  # List of joint names to extract (in order)
  # These must match the names in your JointState messages
  joints:
    - shoulder_pan
    - shoulder_lift
    - elbow
    - wrist_1
    - wrist_2
    - wrist_3
    - gripper_position

# Action joints (subset of joints that are controlled)
# Order matters - this defines action vector ordering
action_joints:
  - shoulder_pan
  - shoulder_lift
  - elbow
  - wrist_1
  - wrist_2
  - wrist_3
  - gripper_position

# Optional: Gripper configuration
gripper:
  # If gripper is separate from main joint state
  topic: /my_robot/gripper/state
  joint_name: gripper_position
  # Normalize to [0, 1] range
  normalize: true
  min_value: 0.0
  max_value: 0.08
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `robot_name` | Yes | Unique identifier for the robot |
| `fps` | Yes | Target frame rate (typically 30) |
| `cameras` | Yes | Map of camera names to configurations |
| `joint_state` | Yes | Joint state topic and joint names |
| `action_joints` | Yes | Joints that are controlled (ordered) |
| `gripper` | No | Separate gripper configuration |

## Mapping Joints and Cameras

### Joint Naming Conventions

Joint names must match exactly what appears in your ROS2 `JointState` messages:

```python
# Example JointState message
joint_state.name = ['shoulder_pan', 'shoulder_lift', 'elbow', ...]
joint_state.position = [0.1, 0.2, 0.3, ...]
```

To discover joint names, echo your joint state topic:

```bash
ros2 topic echo /my_robot/joint_states --once
```

### Camera Naming Conventions

Camera names become part of the observation key in the dataset:

- Config name: `cam_wrist`
- Dataset key: `observation.images.cam_wrist`

Choose descriptive names that indicate camera position:

| Name | Description |
|------|-------------|
| `cam_wrist` | Mounted on robot wrist/end-effector |
| `cam_overhead` | Looking down at workspace |
| `cam_left` | Left-side view |
| `cam_right` | Right-side view |
| `cam_chest` | Front-facing on robot body |

### Finding Camera Topics

List available image topics:

```bash
ros2 topic list | grep image
```

Verify topic message type:

```bash
ros2 topic info /my_robot/camera/wrist/image_raw
# Should show: Type: sensor_msgs/msg/Image
```

## Topic Naming Conventions

### Standard Topic Structure

Follow ROS2 naming conventions:

```
/<robot_name>/<subsystem>/<data_type>
```

Examples:
- `/my_robot/joint_states`
- `/my_robot/camera/wrist/image_raw`
- `/my_robot/gripper/state`

### Namespace Remapping

If your robot uses different namespaces, you can remap topics:

```yaml
# In robot config
cameras:
  cam_wrist:
    # Original topic name used in your ROS2 system
    topic: /real_robot_ns/sensors/rgb_camera
```

## Inference Configuration

Create an inference configuration for your robot:

```yaml
# configs/lerobot_control/my_robot_inference.yaml

# Model configuration
model:
  chunk_size: 100
  n_action_steps: 50
  device: cuda

# Observation inputs
observation:
  cameras:
    cam_wrist:
      topic: /my_robot/camera/wrist/image_raw
      resize: [224, 224]  # Resize for model input
      encoding: rgb8
    cam_overhead:
      topic: /my_robot/camera/overhead/image_raw
      resize: [224, 224]

  joint_state:
    topic: /my_robot/joint_states
    # Must match training configuration
    joints:
      - shoulder_pan
      - shoulder_lift
      - elbow
      - wrist_1
      - wrist_2
      - wrist_3
      - gripper_position

# Action outputs
action:
  topic: /my_robot/commands
  # Must match action_joints from training
  joints:
    - shoulder_pan
    - shoulder_lift
    - elbow
    - wrist_1
    - wrist_2
    - wrist_3
    - gripper_position

# Safety limits (IMPORTANT: adjust for your robot!)
safety:
  position_limits:
    shoulder_pan: [-3.14159, 3.14159]
    shoulder_lift: [-1.5708, 1.5708]
    elbow: [-3.14159, 3.14159]
    wrist_1: [-1.5708, 1.5708]
    wrist_2: [-3.14159, 3.14159]
    wrist_3: [-1.5708, 1.5708]
    gripper_position: [0.0, 0.08]

  velocity_limit: 1.0      # rad/s
  acceleration_limit: 2.0  # rad/s^2
```

### Safety Limits

Determine appropriate safety limits for your robot:

1. **Position limits**: Check robot URDF or datasheet
2. **Velocity limits**: Start conservative (0.5 rad/s)
3. **Acceleration limits**: Start conservative (1.0 rad/s^2)

## Bimanual Robot Configuration

For dual-arm robots, include joints from both arms:

```yaml
# configs/mcap_converter/bimanual_robot.yaml

robot_name: bimanual_robot
fps: 30

cameras:
  cam_left_wrist:
    topic: /left_arm/camera/wrist/image_raw
    resolution: [640, 480]
  cam_right_wrist:
    topic: /right_arm/camera/wrist/image_raw
    resolution: [640, 480]
  cam_overhead:
    topic: /camera/overhead/image_raw
    resolution: [640, 480]

joint_state:
  topic: /joint_states
  joints:
    # Left arm
    - left_shoulder_pan
    - left_shoulder_lift
    - left_elbow
    - left_wrist_1
    - left_wrist_2
    - left_wrist_3
    - left_gripper
    # Right arm
    - right_shoulder_pan
    - right_shoulder_lift
    - right_elbow
    - right_wrist_1
    - right_wrist_2
    - right_wrist_3
    - right_gripper

action_joints:
    # Same order as joints
    - left_shoulder_pan
    - left_shoulder_lift
    - left_elbow
    - left_wrist_1
    - left_wrist_2
    - left_wrist_3
    - left_gripper
    - right_shoulder_pan
    - right_shoulder_lift
    - right_elbow
    - right_wrist_1
    - right_wrist_2
    - right_wrist_3
    - right_gripper
```

## Testing with Mock Controller

Before deploying on real hardware, test your configuration with the mock controller.

### Customize Mock Controller

Modify the mock controller to match your robot:

```python
# ros2/src/lerobot_control/test/mock_controller/mock_controller_node.py

# Update joint names
self.joint_names = [
    'shoulder_pan', 'shoulder_lift', 'elbow',
    'wrist_1', 'wrist_2', 'wrist_3', 'gripper_position'
]

# Update number of joints
self.num_joints = 7

# Update camera topic
self.image_pub = self.create_publisher(
    Image, '/my_robot/camera/wrist/image_raw', 10
)
```

### Run Integration Test

```bash
# Terminal 1: Run mock controller
ros2 run lerobot_control mock_controller_node \
  --ros-args \
  -p timeout:=60.0 \
  -p required_actions:=50

# Terminal 2: Run inference with your config
ros2 run lerobot_control inference_node \
  --ros-args \
  -p model_path:=/path/to/model \
  -p config_file:=/path/to/configs/lerobot_control/my_robot_inference.yaml
```

### Validate Topics

Check that topics are correctly mapped:

```bash
# List all topics
ros2 topic list

# Check topic types
ros2 topic info /my_robot/joint_states

# Echo messages
ros2 topic echo /my_robot/joint_states --once
```

## Common Configuration Issues

### Joint Order Mismatch

**Symptom**: Robot moves erratically or wrong joints move

**Cause**: Joint order in config doesn't match JointState message

**Solution**: Verify joint order by echoing the topic:

```bash
ros2 topic echo /joint_states --field name
```

### Missing Cameras

**Symptom**: Model fails to load or produces errors about missing observations

**Cause**: Camera names in inference config don't match training

**Solution**: Ensure camera names match exactly between:
- MCAP converter config (training data)
- Inference config (deployment)

### Coordinate Frame Issues

**Symptom**: Actions seem correct but robot moves wrong direction

**Cause**: Different coordinate conventions between simulation and real robot

**Solution**: Verify joint sign conventions and add transforms if needed

### Topic Timing Issues

**Symptom**: Inference is slow or observations seem stale

**Cause**: Topics publishing at different rates

**Solution**:
- Ensure cameras and joint states publish at similar rates
- Check timestamp synchronization

## Validation Checklist

Before deploying on real hardware:

- [ ] MCAP converter config creates valid dataset
- [ ] Dataset can be visualized (images look correct)
- [ ] Joint values are in expected ranges
- [ ] Model trains successfully on converted data
- [ ] Mock controller test passes
- [ ] Safety limits are appropriate for robot
- [ ] Topic names match real robot

## Example: Adding Support for UR5 Robot

```yaml
# configs/mcap_converter/ur5.yaml

robot_name: ur5
fps: 30

cameras:
  cam_wrist:
    topic: /ur5/camera/wrist/image_raw
    resolution: [640, 480]

joint_state:
  topic: /ur5/joint_states
  joints:
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint

action_joints:
  - shoulder_pan_joint
  - shoulder_lift_joint
  - elbow_joint
  - wrist_1_joint
  - wrist_2_joint
  - wrist_3_joint

gripper:
  topic: /ur5/gripper/joint_states
  joint_name: finger_joint
  normalize: true
  min_value: 0.0
  max_value: 0.8
```

## Next Steps

After configuring your robot:

1. **Record demonstrations**: See [Data Collection Guide](data-collection.md)
2. **Convert and train**: See [Training Guide](training-guide.md)
3. **Deploy**: See [Deployment Guide](deployment-guide.md)
