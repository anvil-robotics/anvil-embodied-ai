# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Rename `mcap-upload` CLI entry point to `hf-upload` — clarifies the command uploads a converted LeRobot dataset, not raw MCAP files
- Add `--debug` flag to `run_inference.sh` (exports `DEBUG=true`; enables action smoothness, queue depth stats, Action FPS in inference node)
- Add `inference_flags_smoke_test.py` covering all `run_inference.sh` flags via docker shim assertions and live Docker startup tests
- Reorganize README and add per-stage docs: `docs/training.md`, `docs/evaluation.md`, `docs/inference.md`, `docs/data-conversion.md`; fill all previously undocumented CLI flags
- Add `--output-path` flag to `mcap-convert` and improve data conversion description in README (#29)
- Add `merge-datasets` CLI to `mcap_converter` (#24)
- Add artifact provenance tracking (#28)

### Changed
- Rename `--monitor` flag to `--monitor-enable` in `run_inference.sh` for clarity
- Rename `--exclude-observation` / `--exclude-cams` to `--exclude-observs` in `anvil-trainer`; adopt dot-suffix notation (`images.chest`, `state.velocity`) that mirrors feature key namespaces directly

### Fixed
- Fix `anvil-trainer` defaulting to wandb artifact uploads — `--wandb.disable_artifact=true` is now injected by default for both new and resumed runs; override with `--wandb.disable_artifact=false` to re-enable
- Fix `mcap-to-video` failing silently on legacy MCAP files — schema names without the `/msg/` infix (`sensor_msgs/Image`, `sensor_msgs/CompressedImage`) are now recognised in both topic detection and frame decoding
- Fix `--camera-filter` semantics: now discards listed cameras (was incorrectly keeping them)
- Fix `inference-eval-smoke-test.yaml`: restructure camera config from legacy top-level `camera_mapping:` to `cameras.mapping:` nested format (`inference_node.py` reads `cameras.mapping` since the config refactor)
- Fix `EVAL_DATASET_FPS` type mismatch: compose default was integer `30`; `eval_recorder_node` declares `dataset_fps` as ROS2 `DOUBLE` and rejected it — changed to `30.0`
- Fix `run_inference.sh` example command: `--profile` is a docker compose global flag and must precede the subcommand
- Add context-managed alias for `relative_actions_processor` in Pi0.5 training (#30)
- Show per-episode scrolling summary in `mcap-convert` progress (#26)
- Fix inference node FPS regression (#25)
- Add PIL fallback and skip corrupt-JPEG episodes in `mcap_converter` (#22)
- Initialize `n_action_steps_override` before `_log_startup()` (#23)

---

## [2026-05]

### Added
- Use ROS header timestamp for time syncing (#19)

### Changed
- Upgrade to lerobot v0.5.1 (#17)

---

## [2026-04]

### Added
- Inference node refactor with lerobot v0.5.0 VLA policy support (#15)
- Smart `output_dir` auto-generation and per-checkpoint `anvil_config.json` (#14)

---

## [2026-03]

### Added
- Checkpoint-aware inference config and `anvil-trainer` CLI
- Initial public release of the repository

---

[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
