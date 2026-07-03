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
| `--quality-report PATH` | — | Consume a `mcap-valid --format json` report to skip flagged episodes (see below) |
| `--skip-flagged [critical\|warning]` | — | Requires `--quality-report`. Bare flag skips `critical`-only episodes; `--skip-flagged warning` also skips `warning` episodes |
| `--skip-episode-idx SPEC` | — | Manually skip episodes by 1-based index (see below) |

**Skipping flagged or known-bad episodes** — two independent mechanisms, usable together:

```bash
# Scan first, then convert skipping anything flagged critical
uv run mcap-valid -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --format json --output /tmp/quality.json
uv run mcap-convert -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --quality-report /tmp/quality.json --skip-flagged

# Also skip warning-level episodes (only convert fully-clean episodes)
uv run mcap-convert -i data/raw/my-session --config ... --quality-report /tmp/quality.json --skip-flagged warning

# Manually skip specific episodes by 1-based index — no quality report needed
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "3,7"       # episodes 3 and 7
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1:4"       # episodes 1,2,3 (end EXCLUSIVE, like Python's range())
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1,5:8,12"  # mixed: 1, 5, 6, 7, 12
```

`--skip-episode-idx` ranges follow Python slice convention — the end index is **not included** (`1:4` → episodes 1, 2, 3; matches `range(1, 4)`, not "1 through 4 inclusive"). An omitted start defaults to `1`; an omitted end reaches the last episode (`3:` → episode 3 through the end).

Without `--quality-report`, running `mcap-valid` first isn't required — `--skip-episode-idx` alone is enough if you already know which episodes are bad.

---

## mcap-valid

Scan **raw** MCAP recordings for quality issues before conversion — dropped frames, silent topics, cross-episode fps degradation. Run this against `data/raw/...`, not a converted dataset: converted datasets have gap-filled timestamps that hide the original drops.

```bash
uv run mcap-valid -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml
uv run mcap-valid -i data/raw/my-session --config ... --verbose            # show healthy topics too
uv run mcap-valid -i data/raw/my-session --config ... --fail-on-critical   # CI gate, exit 1 on any critical episode
```

A JSON report and a comprehensive Markdown report are **always** written to `./mcap_valid_reports/<input-dir-name>.{json,md}`, in addition to the terminal table — no flags required. The JSON report is what `mcap-convert --quality-report` consumes.

Severity model:

| Severity | Meaning |
|----------|---------|
| 🔴 `critical` | A camera or `joint_states` stream has zero messages, or an internal/leading/trailing gap — real data loss, no benign explanation |
| 🟡 `warning` | An action topic has zero messages or an idle gap (e.g. one arm not yet picked up — this is normal teleop behavior, not necessarily a defect), or a stream's average fps dropped noticeably relative to the rest of the batch |
| 🟢 `ok` | No issues detected |

An episode's overall status is its single worst topic's severity. `--fail-on-critical` only fails on `critical` — `warning` episodes convert normally unless you also pass `mcap-convert --skip-flagged warning`.

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | MCAP file, or a directory scanned recursively for `*.mcap` |
| `--config PATH` | default config | Conversion config YAML — determines which topics are monitored |
| `--format` | `table` | `table` · `json` |
| `--output PATH` | — | Additionally write the report here (independent of the always-on `mcap_valid_reports/` output) |
| `--fail-on-critical` | — | Exit 1 if any episode has a critical issue — for CI gating |
| `--verbose` | — | Show per-topic detail even for episodes with no issues |
| `--stream-gap-factor N` | `5.0` | Stream gap threshold, as a multiple of that topic's own median interval |
| `--stream-min-gap N` | `0.5` | Absolute floor (seconds) below which a stream gap is never flagged |
| `--action-warn-gap N` | `1.0` | Minimum idle duration (seconds) before an action-topic gap is reported |
| `--fps-tolerance N` | `0.15` | Fraction below the batch's median fps before flagging degradation |

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
