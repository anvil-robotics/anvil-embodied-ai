# MCAP Converter

A modular conversion pipeline for transforming ROS2 MCAP recordings into LeRobot v3.0 format datasets.

## Installation

```bash
# Install from local source
cd packages/mcap_converter
pip install -e .

# Or with uv
uv pip install -e .
```

## CLI Tools

### mcap-convert

Convert MCAP files to LeRobot dataset format.

```bash
mcap-convert -i /path/to/mcap/files -o /path/to/output/dataset --config configs/mcap_converter/openarm_bimanual.yaml
```

Options:
- `-i, --input-dir`: Input directory containing MCAP files (required)
- `-o, --output-dir`: Output directory for dataset (default: data/processed/dataset)
- `--config`: Path to YAML config file
- `--robot-type`: Robot type (anvil_openarm, anvil_yam)
- `--fps`: Video framerate (default: 30)
- `--task`: Task name for the dataset
- `--push-to-hub`: Upload to Hugging Face Hub after conversion
- `--buffer-seconds`: Buffer window for time alignment (default: 5.0)

### mcap-inspect

Analyze MCAP file structure and message types.

```bash
mcap-inspect /path/to/file.mcap
mcap-inspect /path/to/file.mcap --topic /joint_states
mcap-inspect /path/to/file.mcap --format json --output analysis.json
```

### mcap-validate

Test that a converted LeRobot dataset loads correctly.

```bash
mcap-validate --repo_id anvil_robot/my_dataset --root /path/to/dataset
```

### mcap-upload

Upload a LeRobot dataset to Hugging Face Hub.

```bash
mcap-upload /path/to/dataset --repo-id anvil-robot/my_dataset
```

## Python API

```python
from mcap_converter import (
    McapReader,
    DataExtractor,
    TimeAligner,
    LeRobotWriter,
    ConfigLoader,
)

# Load configuration
config = ConfigLoader.from_yaml("configs/mcap_converter/openarm_bimanual.yaml")

# Read MCAP file
reader = McapReader("recording.mcap")
topics = reader.list_topics()

# Extract data
extractor = DataExtractor(config)
data = extractor.extract_episode("recording.mcap")

# Align sensors
aligner = TimeAligner(config)
frames = aligner.align_sensors(data, camera_names=["head", "wrist"])

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

Example configuration for bimanual OpenArm robot:

```yaml
# ROS2 Topic containing all joint states
robot_state_topic: "/joint_states"

# Joint name parsing
joint_names:
  separator: "_"
  source:
    leader: action
    follower: observation
  arms:
    r: right
    l: left

# Camera configuration
camera_topics:
  - "/camera0/image_raw"
  - "/camera1/image_raw"

camera_topic_mapping:
  "/camera0/image_raw": "waist"
  "/camera1/image_raw": "wrist_r"

image_resolution: [640, 480]

# Feature extraction
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
│   ├── extractor.py   # Data extraction from MCAP
│   ├── aligner.py     # Time synchronization
│   └── writer.py      # LeRobot dataset writing
├── cli/
│   ├── convert.py     # mcap-convert CLI
│   ├── inspect.py     # mcap-inspect CLI
│   ├── validate.py    # mcap-validate CLI
│   └── upload.py      # mcap-upload CLI
├── config/
│   ├── schema.py      # Configuration dataclasses
│   ├── loader.py      # YAML config loading
│   └── validators.py  # Config validation
├── utils/
│   ├── image_utils.py # Image processing
│   └── logging.py     # Logging utilities
└── exceptions.py      # Custom exceptions
```
