[← Back to README](../README.md)

# Data Conversion

Convert MCAP recordings into LeRobot v3.0 datasets.

---

## mcap-convert

Configs live under `configs/mcap_converter/`, split by schema version — see
`configs/mcap_converter/v1.0/README.md` / `v1.1/README.md` for the full explanation, and
`claude_docs/mcap-converter-encoding-refactor-plan.md` for the versioning/migration design.
Pick a **v1.1** config for any new conversion.

### v1.1 — current schema (use these)

Output lands in `<output-dir>/<joint|ee>-space/<input-dir-name>/` (or `ee-delta-space/`
for `action_encoding: delta` configs).

| Config | Teleop mode | Arms | `observation.state` | `action` |
|--------|-------------|------|---------------------|---------|
| `v1.1/openarm_joint_bimanual.yaml` | Quest VR | Bimanual | `(16,)` joint positions | Command topics |
| `v1.1/openarm_ee_bimanual.yaml` | Quest VR | Bimanual | `(16,)` xyz+quat+gripper ×2 | `(20,)` xyz+rot6d+gripper ×2 |
| `v1.1/openarm_ee_bimanual_16x9.yaml` | Quest VR | Bimanual | same as above | same as above (16:9 cameras) |
| `v1.1/openarm_ee_left.yaml` | Quest VR | Left only | `(8,)` xyz+quat+gripper | `(10,)` xyz+rot6d+gripper |

**EE Cartesian format:**

```
observation.state per arm (8 dims): [x, y, z, qx, qy, qz, qw, gripper]
action         per arm (10 dims): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]
```

The action uses [6D rotation representation](https://arxiv.org/abs/1812.07035) for regression stability. `action_encoding: absolute` (the default) is act-from-obs — `action[t] = ee_pose[t]`; the future prediction window is applied by LeRobot's `delta_timestamps` at train time. `action_encoding: delta` instead bakes a per-frame Delta(n-(n-1)) target at convert time — see `claude_docs/mcap-converter-encoding-refactor-plan.md`. `observation_encoding` (`quaternion`/`rot6d`/`axis_angle`) independently controls `observation.state`'s rotation representation.

### v1.0 — legacy, pre-unification schema (not directly usable — migrate first)

| Config | Teleop mode | Arms | Notes |
|--------|-------------|------|-------|
| `v1.0/openarm_bimanual.yaml` | Leader-follower | Bimanual | action from leader-prefixed `joint_states` |
| `v1.0/openarm_bimanual_quest.yaml` | Quest VR | Bimanual | topic-keyed `action_topics` — superseded by `v1.1/openarm_joint_bimanual.yaml` |
| `v1.0/openarm_single_quest.yaml` | Quest VR | Single (right) | topic-keyed `action_topics` |
| `v1.0/openarm_single_quest_afo.yaml` | Quest VR | Single (right) | topic-keyed `action_topics` + `action_from_observation` |

These predate `data_space`/`observation_topics`/EE support entirely and are rejected outright by the current loader's strict mode. Run `dataset-config-migrate` on one first (see `configs/mcap_converter/v1.0/README.md`), or use the already-migrated `v1.1/openarm_joint_bimanual.yaml` directly for the Quest-bimanual case.

**EE Cartesian format:**

```
observation.state per arm (8 dims): [x, y, z, qx, qy, qz, qw, gripper]
action         per arm (10 dims): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]
```

The action uses [6D rotation representation](https://arxiv.org/abs/1812.07035) for regression stability. EE mode is always act-from-obs — `action[t] = ee_pose[t]` in the converter; the future prediction window is applied by LeRobot's `delta_timestamps` at train time.

---

**action_from_observation** — a `v1.0`-only mechanism (`v1.0/openarm_single_quest_afo.yaml`), used when `/follower_*/commands` was not recorded: instead of reading from command topics, the converter derived actions from the follower's own joint positions shifted N frames forward in time. **No longer accepted by the current loader** (`action_from_observation`/`action_from_observation_n` have no field-level equivalent in the current schema — `dataset-config-migrate` drops them, see `configs/mcap_converter/v1.0/README.md`). For a current joint config, use empty `action_topics: {}` instead — EE mode's own act-from-obs convention (`action[t] = observation.state[t]`, future window applied by LeRobot's `delta_timestamps` at train time) already works the same way for joint mode.

```bash
# Joint space — output: data/datasets/joint-space/my-sessions/
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/v1.1/openarm_joint_bimanual.yaml \
  --output-dir data/datasets \
  --fps 30

# EE Cartesian — output: data/datasets/ee-space/my-sessions/
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/v1.1/openarm_ee_bimanual.yaml \
  --output-dir data/datasets
```

**`--output-dir`** sets the base output directory. Output is saved to `<output-dir>/<space>-space/<input-dir-name>/` where `<space>` is `ee` or `joint` based on the config.

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

## dataset-valid

Validate a converted dataset — runs 5 structural checks.

```bash
uv run dataset-valid --root data/datasets/my-sessions
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
