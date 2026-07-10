[← Back to README](../README.md)

# Data Conversion

Convert a **recorded session** into a LeRobot v3.0 dataset.

A session is a directory of raw MCAP files (e.g. `data/raw/my-session/`) — one `.mcap` file per episode, all from the same recording run. Every tool below targets that same session directory, from the initial quality scan through to the finished dataset.

**The usual flow:** `mcap-valid` (scan the target session for problems) → `mcap-convert` (convert that session into a dataset) → `dataset-valid` (sanity-check the result). The other tools below (`mcap-to-video`, `merge-datasets`, `hf-upload`) are used as needed, not part of every conversion.

```
┌─────────────────┐
│ target session  │  data/raw/<session>/ — one *.mcap file per episode
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   mcap-valid    │  scan every episode in the session for quality issues
│                 │  - severity per episode: critical / warning / pass
│                 │  - writes <session>/mcap_valid_reports/report.{json,md}
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  mcap-convert   │  convert the whole session → one LeRobot v3.0 dataset
│                 │  - required: refuses to run without the report above
│                 │  - skips critical episodes by default (--include-flagged to override)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  dataset-valid  │  sanity-check the converted dataset
└─────────────────┘

Used as needed, not part of every conversion:
  mcap-to-video   — MCAP camera topics → MP4
  merge-datasets  — combine multiple LeRobot datasets
  hf-upload       — push a dataset to HuggingFace Hub
```

---

## mcap-valid

Scan a **raw session** — the same `data/raw/<session>/` directory you're about to convert — for quality issues before conversion: dropped frames, silent topics, cross-episode fps degradation. Point `-i` at the raw session, not a converted dataset: converted datasets have gap-filled timestamps that hide the original drops.

No config needed — topic roles (joint-state stream, camera stream, action command) are inferred entirely from each topic's own ROS2 message type. Every topic present in the file appears in the output; message types outside the 3 known roles show up as `unclassified` (informational only, never affects severity).

```bash
uv run mcap-valid -i data/raw/my-session
uv run mcap-valid -i data/raw/my-session --verbose            # show healthy topics too
uv run mcap-valid -i data/raw/my-session --fail-on-critical   # CI gate, exit 1 on any critical episode
uv run mcap-valid -i data/raw/my-session --topic /joint_states  # deep field-structure dump for one topic
```

A JSON report and a comprehensive Markdown report are **always** written to `<session>/mcap_valid_reports/report.{json,md}` — inside the session directory itself, not relative to wherever you happened to run the command from — in addition to the terminal table, no flags required. **`mcap-convert` refuses to run without this report** (see below) — running `mcap-valid` first is a required step, not optional.

Every run also prints a baseline table of every topic found in the file (`Topic | Type | Messages | Role`), regardless of severity — this replaces the old standalone `mcap-inspect` tool's topic listing. Pass `--topic TOPIC` for a deeper per-message field-structure dump of one topic (also folded in from the old `mcap-inspect`).

Severity model:

| Severity | Meaning |
|----------|---------|
| 🔴 `critical` | A camera or `joint_states` stream has zero messages, or an internal/leading/trailing gap — real data loss, no benign explanation. Also raised if a topic present in the majority of sibling episodes in the same batch is completely absent from this one (catches a camera/driver that never started) |
| 🟡 `warning` | An action topic has zero messages or an idle gap (e.g. one arm not yet picked up — this is normal teleop behavior, not necessarily a defect), or a stream's average fps dropped noticeably relative to the rest of the batch |
| 🟢 `pass` | No issues detected. Unclassified topics are always `pass` — they never affect the episode's overall severity |

An episode's overall status is its single worst topic's severity. `--fail-on-critical` only fails on `critical` — `warning` episodes convert normally unless you also pass `mcap-convert --include-flagged pass`.

**Known tradeoff — `action_from_observation` (AFO) datasets:** without a config, `mcap-valid` can't know a dataset is configured to derive actions from observations instead of a dedicated command topic. If the action-command topic was never recorded at all, it just doesn't appear in the report (not a warning). If the topic exists but has zero messages, it shows as `warning` here (vs. `pass` under the old config-aware behavior). This never blocks conversion — it only matters if you explicitly pass `mcap-convert --include-flagged pass` on AFO data.

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | A single MCAP file, or a session directory scanned recursively for `*.mcap` episode files |
| `--format` | `table` | `table` · `json` |
| `--output PATH` | — | Additionally write the report here (independent of the always-on `mcap_valid_reports/` output) |
| `--fail-on-critical` | — | Exit 1 if any episode has a critical issue — for CI gating |
| `--verbose` | — | Show per-topic detail even for episodes with no issues |
| `--topic TOPIC` | — | Deep field-structure dump for one topic (folded in from the old `mcap-inspect`) |
| `--max-samples N` | `5` | Max message samples to analyze per topic, for `--topic` |
| `--stream-gap-factor N` | `5.0` | Stream gap threshold, as a multiple of that topic's own median interval |
| `--stream-min-gap N` | `0.5` | Absolute floor (seconds) below which a stream gap is never flagged |
| `--action-warn-gap N` | `1.0` | Minimum idle duration (seconds) before an action-topic gap is reported |
| `--fps-tolerance N` | `0.15` | Fraction below the batch's median fps before flagging degradation |

---

## mcap-convert

Converts every episode in a target session (`--input-dir`) into one LeRobot v3.0 dataset — this is the core, required step of the whole pipeline; everything else here supports or follows it.

**Requires a `mcap-valid` quality report for that same session to exist first** — either auto-discovered at the default `<input-dir>/mcap_valid_reports/report.json` (written automatically by `mcap-valid`, see above, inside the session directory itself), or pointed at explicitly with `--quality-report PATH`. If neither is found, `mcap-convert` exits with an error telling you to run `mcap-valid` first — it does not fall back to converting without one. This gate only checks that a report *file* exists — but `mcap-convert` also acts on its *contents* automatically: `--include-flagged` defaults to `warning`, so `pass` and `warning` episodes convert normally while `critical` episodes (e.g. a camera with zero messages) are skipped without you having to pass anything. `--include-flagged pass` is the stricter override — it also skips warning-level episodes, converting only fully-clean ones. `--include-flagged critical` is the looser override — it converts every episode regardless of severity.

Pick the config that matches your recording setup. `--config` is technically optional — omitting it falls back to a bare default `DataConfig()` that doesn't match any real robot's topic layout — so in practice always pass one of these:

| Config | Teleop mode | Arms | Action source | Camera aspect |
|--------|-------------|------|---------------|---------------|
| `openarm_bimanual.yaml` | Leader-follower | Bimanual | Leader joint positions | 4:3 (640×480) |
| `openarm_bimanual_quest.yaml` | Quest VR | Bimanual | Command topics | 4:3 (640×480) |
| `openarm_bimanual_quest_16x9.yaml` | Quest VR | Bimanual | Command topics | 16:9 (480×270) |
| `openarm_single_quest.yaml` | Quest VR | Single (right) | Command topics | 4:3 (640×480) |
| `openarm_single_quest_afo.yaml` | Quest VR | Single (right) | Observation lookahead | 4:3 (640×480) |

`_16x9` variants set `image_resolution: [480, 270]`, an exact ÷4 downscale of 1920×1080 source cameras with zero letterbox padding. Use the matching `_16x9` config instead of the 4:3 default if your cameras natively output 1920×1080 — see [docs/training.md](training.md#diffusion) for details.

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
| `-i / --input-dir PATH` | _(required)_ | The target session directory — walked recursively for `*.mcap` episode files |
| `--config PATH` | bare `DataConfig()` | Conversion config YAML (see table above) — always pass one in practice |
| `-o / --output-dir PATH` | `data/datasets` | Base output directory — dataset lands at `<output-dir>/<input-dir-name>/` |
| `--output-path PATH` | — | Full output path override — bypasses auto-naming |
| `--resume` | — | Skip already-converted episodes — safe to re-run after interruption |
| `--max-episodes N` | all | Convert only the first N episodes |
| `--fps N` | auto | Override output FPS (must not exceed source FPS) |
| `--tolerance-s N` | `0.001` | Timestamp sync tolerance in seconds |
| `--task NAME` | `manipulation` | Task name recorded into the dataset |
| `--buffer-seconds N` | `5.0` | Buffer window for time alignment, in seconds |
| `--debug-plot-episodes N` | `5` | Number of episodes to include in debug plots |
| `--vcodec` | `h264` | `h264` · `hevc` · `libsvtav1` |
| `--robot-type` | `anvil_openarm` | `anvil_openarm` · `anvil_yam` |
| `--act-from-obs-n-step N` | config value | Override `action_from_observation_n` at runtime: `action[t] = observation[t+N]` |
| `--quality-report PATH` | auto-discovered | Path to a mcap-valid JSON report — mcap-convert requires one to exist; if omitted, the default `<input-dir>/mcap_valid_reports/report.json` is used |
| `--include-flagged [pass\|warning\|critical]` | `warning` | Highest severity tier to include when converting (inclusive threshold). `warning` (default) converts `pass` and `warning` episodes, skipping only `critical` ones automatically. `pass` is stricter — it also skips `warning` episodes. `critical` is the "convert everything" escape hatch — nothing is skipped. Works against whichever report the mandatory gate resolved (explicit or auto-discovered) |
| `--skip-episode-idx SPEC` | — | Manually skip episodes by 1-based index (see below) |
| `--push-to-hub` | — | Upload to HuggingFace Hub after conversion |
| `--hf-user NAME` | auto-detect | HuggingFace username, used when `--push-to-hub` is set |
| `--hf-repo NAME` | output dir name | HuggingFace dataset repo name, used when `--push-to-hub` is set |

**Skipping flagged or known-bad episodes** — two independent mechanisms, usable together:

```bash
# Scan first — critical episodes from this report are now skipped by default,
# with no extra flag needed:
uv run mcap-valid -i data/raw/my-session --format json --output /tmp/quality.json
uv run mcap-convert -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --quality-report /tmp/quality.json

# Also skip warning-level episodes (only convert fully-clean episodes)
uv run mcap-convert -i data/raw/my-session --config ... --quality-report /tmp/quality.json --include-flagged pass

# Opt out of quality-based skipping entirely — convert every episode, including critical ones
uv run mcap-convert -i data/raw/my-session --config ... --quality-report /tmp/quality.json --include-flagged critical

# Manually skip specific episodes by 1-based index — no quality report needed
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "3,7"       # episodes 3 and 7
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1:4"       # episodes 1,2,3 (end EXCLUSIVE, like Python's range())
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1,5:8,12"  # mixed: 1, 5, 6, 7, 12
```

`--skip-episode-idx` ranges follow Python slice convention — the end index is **not included** (`1:4` → episodes 1, 2, 3; matches `range(1, 4)`, not "1 through 4 inclusive"). An omitted start defaults to `1`; an omitted end reaches the last episode (`3:` → episode 3 through the end).

`--skip-episode-idx` doesn't need the quality report's *contents* to be relevant — but `mcap-convert` still needs *a* report to exist at all (per the mandatory gate above) before it will run, even if you're only using `--skip-episode-idx`.

---

## dataset-valid

Validate a converted dataset — loads it and runs 5 checks (load, inspect features, read first frame, batch-read, print stats).

```bash
uv run dataset-valid --root data/datasets/my-sessions
```

Expected: 5 checks all showing `[OK]`.

| Flag | Default | Description |
|------|---------|-------------|
| `--root PATH` | `output_dataset` | Dataset root directory to validate |
| `--repo-id ID` | `anvil_robot/manipulation_v1` | Dataset repository ID passed to `LeRobotDataset` |

To browse the converted dataset with a Rerun viewer, see [Dataset Visualization](dataset-viz.md).

---

## Additional tools

Used as needed — not part of every conversion.

### mcap-to-video

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

### merge-datasets

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

### hf-upload

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
