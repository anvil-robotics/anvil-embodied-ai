[← Back to README](../README.md)

# Data Conversion

Convert MCAP recordings into LeRobot v3.0 datasets.

---

## mcap-convert

Pick the config that matches your recording setup:

| Config | Teleop mode | Arms | Action source |
|--------|-------------|------|---------------|
| `openarm_bimanual.yaml` | Leader-follower | Bimanual | Leader joint positions |
| `openarm_bimanual_quest.yaml` | Quest VR | Bimanual | Command topics |
| `openarm_single_quest.yaml` | Quest VR | Single (right) | Command topics |
| `openarm_single_quest_afo.yaml` | Quest VR | Single (right) | Observation lookahead |

**action_from_observation** — used by `openarm_single_quest_afo.yaml` when `/follower_*/commands` was not recorded. Instead of reading from command topics, the converter derives actions from the follower's own joint positions shifted N frames forward in time. Enable in your conversion config YAML:

```yaml
action_from_observation: true
action_from_observation_n: 10 # action[t] = observation.state[t + n] (default n=10, ≈333ms at 30fps)
```

```bash
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --output-dir data/datasets \
  --fps 30
```

**`--output-dir`** sets the base output directory. Output is always saved to `<output-dir>/<input-dir-name>/`.

**`--output-path`** bypasses auto-naming entirely — the dataset lands exactly where you point it.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir PATH` | _(required)_ | Directory containing MCAP session folders |
| `--config PATH` | _(required)_ | Conversion config YAML (see table above) |
| `--output-dir PATH` | `data/datasets` | Base output directory — dataset lands at `<output-dir>/<input-dir-name>/` |
| `--output-path PATH` | — | Full output path override — bypasses auto-naming |
| `--resume` | — | Skip already-converted episodes — safe to re-run after interruption |
| `--max-episodes N` | all | Convert only the first N episodes |
| `--fps N` | auto | Override output FPS (must not exceed source FPS) |
| `--vcodec` | `h264` | `h264` · `hevc` · `libsvtav1` |
| `--robot-type` | `anvil_openarm` | `anvil_openarm` · `anvil_yam` |
| `--act-from-obs-n-step N` | config value | Override `action_from_observation_n` at runtime: `action[t] = observation[t+N]` |

---

## dataset-validate

Validate a converted dataset — runs 5 structural checks.

```bash
uv run dataset-validate --root data/datasets/my-sessions
```

Expected: 5 checks all showing `[OK]`.

---

## merge-datasets

Merge two or more LeRobot datasets into one. All datasets must share the same feature schema; use `--remove-features` to strip mismatched features before merging.

```bash
uv run merge-datasets data/datasets/ds-a data/datasets/ds-b \
  --output data/datasets/ds-merged

# Strip extra features from datasets recorded with velocity+effort
uv run merge-datasets data/datasets/ds-a data/datasets/ds-b \
  --output data/datasets/ds-merged \
  --remove-features observation.velocity,observation.effort
```

| Flag | Description |
|------|-------------|
| `PATH [PATH ...]` | Two or more dataset paths to merge (positional) |
| `--output PATH` | _(required)_ Output path for the merged dataset |
| `--remove-features F1,F2` | Comma-separated features to strip from any dataset that has them |

> When `--remove-features` is used, a trimmed copy (`<path>-trimmed`) is written alongside the original and reused on subsequent runs.

---

## mcap-inspect

Inspect an MCAP file's topics, message types, and message counts.

```bash
uv run mcap-inspect /path/to/recording.mcap
uv run mcap-inspect /path/to/recording.mcap --topic /joint_states --format json
uv run mcap-inspect /path/to/recording.mcap --format json --output report.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `mcap_path` | _(required)_ | Path to MCAP file (positional) |
| `--topic TOPIC` | all topics | Only analyze the specified topic |
| `--max-samples N` | `5` | Max message samples to analyze per topic |
| `--format` | `text` | Output format: `text` · `json` |
| `--output PATH` | stdout | Write output to file instead of stdout |

---

## mcap-to-video

Extract image topics from an MCAP file to MP4 videos (one file per camera). Memory-efficient — processes one frame at a time.

```bash
uv run mcap-to-video -i recording.mcap -o ./videos
uv run mcap-to-video -i recording.mcap --scan-only                            # list topics only
uv run mcap-to-video -i recording.mcap -o ./videos --fps 30 --resize 640x480 # resize + fps
uv run mcap-to-video -i recording.mcap -o ./videos \
  --topics /cam_waist/image_raw/compressed                                    # specific topic
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | MCAP file or directory of MCAP files |
| `-o / --output-dir PATH` | `./videos` | Output directory for MP4 files |
| `--topics TOPIC [...]` | auto-detect | Specific topics to convert |
| `--fps N` | `30` | Output video FPS |
| `--codec` | `libx264` | `libx264` · `libx265` · `libaom-av1` |
| `--crf N` | `23` | Constant rate factor — lower = better quality |
| `--resize WxH` | — | Resize frames, e.g. `640x480` |
| `--scan-only` | — | List image topics without converting |

---

## hf-upload

Upload a converted LeRobot dataset to HuggingFace Hub.

```bash
# Login first (one-time)
huggingface-cli login

uv run hf-upload /path/to/dataset                                  # repo-id auto from dir name
uv run hf-upload /path/to/dataset --repo-id your-org/my_dataset
uv run hf-upload /path/to/dataset --repo-id your-org/my_dataset --private
```

| Flag | Default | Description |
|------|---------|-------------|
| `dataset_path` | _(required)_ | Path to local dataset directory (positional) |
| `--repo-id ORG/NAME` | auto from dir name | HuggingFace repository ID |
| `--private` | — | Make the repository private |
| `--force` | — | Skip confirmation prompt if repo already exists |
| `--hf-user USER` | auto-detect | HuggingFace username |

---

[← Back to README](../README.md)
