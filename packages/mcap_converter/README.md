# MCAP Converter

Convert ROS2 MCAP recordings into LeRobot v3.0 datasets.

## Installation

```bash
# From repository root (installs all workspace packages)
uv sync --all-packages
```

## CLI Tools

### mcap-convert

Convert MCAP files to LeRobot dataset format.

```bash
mcap-convert -i /path/to/mcap/dir -o /tmp/dataset --config configs/mcap_converter/openarm_bimanual.yaml
```

Key options:

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input-dir` | Directory containing MCAP files | required |
| `-o, --output-dir` | Output directory | `data/processed/dataset` |
| `--config` | Robot config YAML file | |
| `--frequency` | Output dataset sample rate (Hz). Clamps to source rate if higher. | from config (`60`) |
| `--vcodec` | Video codec (`h264`, `hevc`, `libsvtav1`) | `h264` |
| `--task` | Task name for the dataset | `manipulation` |
| `--push-to-hub` | Upload to HuggingFace Hub after conversion | |

### mcap-inspect

Analyze MCAP file structure and message types.

```bash
mcap-inspect /path/to/file.mcap
mcap-inspect /path/to/file.mcap --topic /joint_states --format json
```

### mcap-to-video

Extract image topics from MCAP files directly to MP4 videos.

```bash
mcap-to-video -i recording.mcap -o ./videos
mcap-to-video -i recording.mcap --scan-only
```

### dataset-validate

Validate a converted LeRobot dataset by loading and reading frames.

```bash
dataset-validate --root /path/to/dataset
```

### mcap-upload

Upload a LeRobot dataset to Hugging Face Hub.

```bash
mcap-upload /path/to/dataset --repo-id anvil-robot/my_dataset
```

## Python API

```python
from mcap_converter import McapReader, LeRobotWriter, ConfigLoader

# Load configuration
config = ConfigLoader.from_yaml("configs/mcap_converter/openarm_bimanual.yaml")

# Read MCAP file
reader = McapReader("recording.mcap")

# Write dataset
writer = LeRobotWriter(
    output_dir="output_dataset",
    repo_id="anvil_robot/my_dataset",
    fps=30,
)
dataset = writer.create_dataset(joint_names, camera_names)
writer.add_episode(dataset, frames, episode_index=0)
writer.finalize(dataset)
```

## Configuration

Example for bimanual OpenArm robot (`configs/mcap_converter/openarm_bimanual.yaml`):

```yaml
robot_state_topic: "/joint_states"

joint_names:
  separator: "_"
  source:
    leader: action
    follower: observation
  arms:
    r: right
    l: left

camera_topic_mapping:
  "/usb_cam_waist/image_raw/compressed": "waist"
  "/usb_cam_wrist_r/image_raw/compressed": "wrist_r"

image_resolution: [640, 480]

observation_feature_mapping:
  state: "position"
  others:
    - "velocity"
    - "effort"

action_feature_mapping:
  state: "position"
  others: []
```

## Module Structure

```
mcap_converter/
├── core/
│   ├── reader.py      # MCAP file reading
│   ├── extractor.py   # Data extraction & streaming
│   ├── aligner.py     # Time synchronization
│   └── writer.py      # LeRobot dataset writing
├── cli/
│   ├── convert.py     # mcap-convert
│   ├── inspect.py     # mcap-inspect
│   ├── validate.py    # dataset-validate
│   ├── upload.py      # mcap-upload
│   └── video.py       # mcap-to-video
├── config/
│   ├── schema.py      # Configuration dataclasses
│   ├── loader.py      # YAML config loading
│   └── validators.py  # Config validation
├── utils/
│   ├── image_utils.py # Image processing
│   └── logging.py     # Logging utilities
└── exceptions.py      # Custom exceptions
```
