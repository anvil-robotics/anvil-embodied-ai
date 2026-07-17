# MCAP Converter

Convert ROS2 MCAP recordings into LeRobot v3.0 datasets.
Supports both **joint-space** and **EE Cartesian-space** action representations.

## Installation

```bash
# From repository root (installs all workspace packages)
uv sync --all-packages
```

## CLI Tools

### mcap-convert

Convert MCAP files to LeRobot dataset format.

```bash
# EE Cartesian bimanual (new)
mcap-convert -i data/raw/my-session -o data/datasets \
  --config configs/mcap_converter/v1.1/openarm_ee_bimanual.yaml

# Joint space bimanual (existing)
mcap-convert -i data/raw/my-session -o data/datasets \
  --config configs/mcap_converter/v1.1/openarm_joint_bimanual.yaml
```

Key options:

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input-dir` | Directory containing MCAP files | required |
| `-o, --output-dir` | Output base directory | `data/datasets` |
| `--config` | Conversion config YAML | |
| `--fps` | Output framerate (auto-detected if omitted) | auto |
| `--vcodec` | Video codec (`h264`, `hevc`, `libsvtav1`) | `h264` |
| `--resume` | Skip already-converted episodes | |
| `--max-episodes N` | Convert only the first N episodes | |
| `--act-from-obs` | Force `action[t] = obs[t]` even when action topics are configured | |
| `--push-to-hub` | Upload to HuggingFace Hub after conversion | |

Output is always saved to `<output-dir>/<input-dir-name>/`.

### Other tools

```bash
mcap-inspect /path/to/file.mcap          # Analyze MCAP topics and message types
mcap-to-video -i recording.mcap -o ./videos
dataset-valid --root data/datasets/my-session
mcap-upload /path/to/dataset --repo-id anvil-robot/my_dataset
```

## Configuration

All configs share the same **unified format**. The only required change between modes is `data_space`.

### EE Cartesian mode

Reads `/ee_pose_left` / `/ee_pose_right` (`anvil_msgs/msg/CommandedEEPose`):

```yaml
data_space: "ee"

observation_topics:
  left:  "/ee_pose_left"
  right: "/ee_pose_right"

# action_topics must be empty in EE mode —
# action[t] = obs[t] (future window applied by delta_timestamps at train time)
action_topics: {}

camera_topics:
  - "/cam_waist/image_raw/compressed"
  - "/cam_wrist_r/image_raw/compressed"
  - "/cam_chest/image_raw/compressed"
  - "/cam_wrist_l/image_raw/compressed"

camera_topic_mapping:
  "/cam_waist/image_raw/compressed":  "waist"
  "/cam_wrist_r/image_raw/compressed": "wrist_r"
  "/cam_chest/image_raw/compressed":  "chest"
  "/cam_wrist_l/image_raw/compressed": "wrist_l"

image_resolution: [640, 480]
```

**Output schema:**
```
observation.state  float32 (8 * n_arms,)   per arm: [x, y, z, qx, qy, qz, qw, gripper]
action             float32 (10 * n_arms,)  per arm: [x, y, z, r0, r1, r2, r3, r4, r5, gripper]
```
The action uses **6D rotation (Zhou et al. 2019)**: first two columns of the 3×3 rotation matrix,
flattened. This avoids discontinuities in Euler/quaternion representations and improves regression.

Single-arm: list only one entry under `observation_topics` (arm scope is implicit from the dict keys).

### Joint mode (Quest teleop)

Reads `/joint_states` and per-arm command topics:

```yaml
data_space: "joint"

observation_topics:
  left:  "/joint_states"
  right: "/joint_states"

action_topics:
  left:
    topic: "/follower_l_forward_position_controller/commands"
    joint_order: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"]
  right:
    topic: "/follower_r_forward_position_controller/commands"
    joint_order: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"]

# joint_names: how to split /joint_states by arm
joint_names:
  separator: "_"
  source: { follower: observation }
  arms:   { l: left, r: right }

observation_feature_mapping:
  state: "position"
  others: []   # velocity/effort omitted in new configs

action_feature_mapping:
  state: "position"
  others: []

camera_topics: [ ... ]
camera_topic_mapping: { ... }
image_resolution: [640, 480]
```

**Joint mode act-from-obs:** set `action_topics: {}` (or use `--act-from-obs` on the CLI) to write
`action[t] = observation.state[t]`. The future prediction window is then deferred to LeRobot's
`delta_timestamps` at train time.

### Available configs

Split by schema version under `configs/mcap_converter/v1.0/` (legacy, pre-unification —
not directly usable, see that folder's `README.md`) and `configs/mcap_converter/v1.1/`
(current — use these). Full version/migration design:
`claude_docs/mcap-converter-encoding-refactor-plan.md`.

| Config | `data_space` | Arms | Output dims |
|--------|-------------|------|-------------|
| `v1.1/openarm_ee_bimanual.yaml` | `ee` | left + right | state `(16,)`, action `(20,)` |
| `v1.1/openarm_ee_bimanual_16x9.yaml` | `ee` | left + right | same (16:9 cameras) |
| `v1.1/openarm_ee_left.yaml` | `ee` | left only | state `(8,)`, action `(10,)` |
| `v1.1/openarm_joint_bimanual.yaml` | `joint` | left + right | state `(16,)`, action `(16,)` |
| `v1.0/openarm_bimanual.yaml` | `joint` (v1.0, legacy fmt) | left + right | needs `dataset-config-migrate` first |
| `v1.0/openarm_bimanual_quest.yaml` | `joint` (v1.0, legacy fmt) | left + right | superseded by `v1.1/openarm_joint_bimanual.yaml` |
| `v1.0/openarm_single_quest.yaml` | `joint` (v1.0, legacy fmt) | right only | needs `dataset-config-migrate` first |
| `v1.0/openarm_single_quest_afo.yaml` | `joint` (v1.0, legacy fmt) | right only | needs `dataset-config-migrate` first |

## Python API

```python
from mcap_converter import McapReader, LeRobotWriter, ConfigLoader

config = ConfigLoader.from_yaml("configs/mcap_converter/v1.1/openarm_ee_bimanual.yaml")

writer = LeRobotWriter(
    output_dir="data/datasets/my-session",
    repo_id="anvil_robot/my_dataset",
    config=config,
    fps=30,
)
# EE mode: joint_names not needed — dims from config.observation_topics
dataset = writer.create_dataset(joint_names={}, camera_names=["chest", "waist", "wrist_l", "wrist_r"])
writer.add_episode(dataset, frames)
writer.finalize(dataset)
```

## Module Structure

```
mcap_converter/
├── core/
│   ├── reader.py      # MCAP file reading
│   ├── extractor.py   # Streaming extraction (joint + EE Cartesian paths)
│   ├── aligner.py     # Time synchronization
│   └── writer.py      # LeRobot dataset writing (joint + EE feature schemas)
├── cli/
│   ├── convert.py     # mcap-convert (--act-from-obs, EE/joint gate)
│   ├── inspect.py     # mcap-inspect
│   ├── validate.py    # dataset-valid
│   ├── upload.py      # mcap-upload
│   └── video.py       # mcap-to-video
├── config/
│   ├── schema.py      # Unified DataConfig (data_space, observation_topics, action_topics)
│   ├── loader.py      # YAML config loading (new unified format)
│   └── validators.py  # Config validation (called at runtime before conversion)
├── utils/
│   ├── image_utils.py # Image processing (JPEG decode with PIL fallback)
│   └── logging.py     # Logging utilities
└── exceptions.py      # Custom exceptions

packages/anvil_shared/src/anvil_shared/
└── rotation.py        # Rotation helpers: quat_to_matrix, matrix_to_rot6d,
                       #   rot6d_to_matrix, matrix_to_quat (xyzw convention)
```

## Rotation representation

EE action uses **6D rotation (rot6d)**:

```python
from anvil_shared.rotation import quat_to_matrix, matrix_to_rot6d, rot6d_to_matrix, matrix_to_quat

# Encode (converter writes this into 'action')
R = quat_to_matrix([qx, qy, qz, qw])   # (3,3)
r6d = matrix_to_rot6d(R)               # (6,) = first two columns of R, flattened

# Decode (trainer / inference reads this back)
R = rot6d_to_matrix(r6d)               # (3,3) via Gram-Schmidt
quat = matrix_to_quat(R)               # (4,) [x,y,z,w]
```

Quaternion convention is `[x, y, z, w]` throughout, consistent with ROS/TF2.
