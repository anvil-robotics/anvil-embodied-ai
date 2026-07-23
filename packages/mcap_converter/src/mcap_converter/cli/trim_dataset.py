"""Analyze and materialize motion-trimmed LeRobot datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from av import logging as av_logging
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import DEFAULT_FEATURES

from mcap_converter.core.motion_trim import MotionTrimConfig, detect_motion_window, joint_groups

av_logging.set_level(av_logging.ERROR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-trim",
        description=(
            "Find sustained task motion, write an auditable trim plan, and optionally "
            "materialize a frame- and video-aligned LeRobot dataset. SOURCE may be a local "
            "dataset path or a Hugging Face dataset repo ID."
        ),
    )
    parser.add_argument("source", help="Local LeRobot dataset path or Hugging Face repo ID")
    parser.add_argument(
        "--output", type=Path, help="New dataset directory (required unless --dry-run)"
    )
    parser.add_argument("--download-root", type=Path, help="Where to download a Hub dataset")
    parser.add_argument("--revision", help="Optional Hugging Face revision")
    parser.add_argument(
        "--start-mode",
        choices=("motion", "displacement", "gripper"),
        default="motion",
        help="Alignment event: home departure, beyond-home displacement, or first gripper event",
    )
    parser.add_argument("--baseline-frames", type=int, default=15)
    parser.add_argument("--sustain-frames", type=int, default=5)
    parser.add_argument("--arm-threshold", type=float, default=0.02, metavar="RAD")
    parser.add_argument("--displacement-threshold", type=float, default=0.10, metavar="RAD")
    parser.add_argument("--gripper-threshold", type=float, default=0.01)
    parser.add_argument(
        "--start-offset-frames",
        type=int,
        help=(
            "Signed offset from the detected event. Defaults: motion=-10, displacement=0, "
            "gripper=-15."
        ),
    )
    parser.add_argument("--end-arm-threshold", type=float, default=0.02, metavar="RAD")
    parser.add_argument("--end-gripper-threshold", type=float, default=0.005)
    parser.add_argument("--end-postroll-frames", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument(
        "--overrides",
        type=Path,
        help='JSON mapping episode index to {"start": frame, "end": frame}',
    )
    parser.add_argument("--manifest", type=Path, help="Trim-plan JSON path")
    parser.add_argument(
        "--reference-plan",
        type=Path,
        help=(
            "With --start-mode gripper, retain the reference plan's median number of "
            "pre-gripper frames for embodiment-independent task-phase alignment"
        ),
    )
    parser.add_argument(
        "--apply-plan",
        type=Path,
        help="Materialize exact windows from an existing plan instead of detecting them again",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only analyze and write the plan")
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser


def _load_source(args: argparse.Namespace) -> LeRobotDataset:
    source_path = Path(args.source).expanduser()
    if source_path.exists():
        return LeRobotDataset(
            source_path.name,
            root=source_path.resolve(),
            video_backend="pyav",
            return_uint8=True,
        )
    if "/" not in args.source:
        raise FileNotFoundError(
            f"{args.source!r} is neither an existing path nor a Hugging Face repo ID"
        )
    return LeRobotDataset(
        args.source,
        root=args.download_root.resolve() if args.download_root else None,
        revision=args.revision,
        video_backend="pyav",
        return_uint8=True,
    )


def _config_from_args(
    args: argparse.Namespace, reference_preroll_frames: int | None = None
) -> MotionTrimConfig:
    defaults = {"motion": -10, "displacement": 0, "gripper": -15}
    start_offset = (
        args.start_offset_frames
        if args.start_offset_frames is not None
        else (
            -reference_preroll_frames
            if reference_preroll_frames is not None
            else defaults[args.start_mode]
        )
    )
    return MotionTrimConfig(
        start_mode=args.start_mode,
        baseline_frames=args.baseline_frames,
        sustain_frames=args.sustain_frames,
        arm_threshold=args.arm_threshold,
        displacement_threshold=args.displacement_threshold,
        gripper_threshold=args.gripper_threshold,
        start_offset_frames=start_offset,
        end_arm_threshold=args.end_arm_threshold,
        end_gripper_threshold=args.end_gripper_threshold,
        end_postroll_frames=args.end_postroll_frames,
        min_frames=args.min_frames,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_preroll(path: Path) -> tuple[int, dict[str, Any]]:
    plan = json.loads(path.read_text())
    events = [
        int(episode["start_event"])
        for episode in plan.get("episodes", [])
        if episode.get("start_event_found")
    ]
    if not events:
        raise ValueError(f"Reference plan contains no detected start events: {path}")
    median_event = float(np.median(events))
    frames = int(np.floor(median_event + 0.5))
    return frames, {
        "method": "matched_pre_gripper_duration",
        "plan": str(path.resolve()),
        "sha256": _sha256(path),
        "repo_id": plan.get("source", {}).get("repo_id"),
        "median_pre_gripper_frames": median_event,
        "applied_preroll_frames": frames,
    }


def _source_fingerprint(root: Path) -> dict[str, dict[str, Any]]:
    paths = [root / "meta" / "info.json", *sorted((root / "data").rglob("*.parquet"))]
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
        if path.exists()
    }


def _load_overrides(path: Path | None) -> dict[int, dict[str, int]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    raw = raw.get("episodes", raw)
    return {
        int(index): {key: int(value) for key, value in values.items()}
        for index, values in raw.items()
    }


def _task_map(dataset: LeRobotDataset) -> dict[int, str]:
    if dataset.meta.tasks is None:
        return {}
    return {int(row["task_index"]): str(task) for task, row in dataset.meta.tasks.iterrows()}


def build_trim_plan(
    dataset: LeRobotDataset,
    config: MotionTrimConfig,
    overrides: dict[int, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Analyze every episode without decoding video."""
    overrides = overrides or {}
    action_feature = dataset.meta.features.get("action")
    if action_feature is None or not action_feature.get("names"):
        raise ValueError("Dataset action feature must have named joints")
    action_names = list(action_feature["names"])
    state_key = "observation.state"
    task_by_index = _task_map(dataset)
    episode_plans: list[dict[str, Any]] = []
    start_states: list[np.ndarray] = []

    for episode_index in range(dataset.meta.total_episodes):
        metadata = dataset.meta.episodes[episode_index]
        source_from = int(metadata["dataset_from_index"])
        source_to = int(metadata["dataset_to_index"])
        rows = dataset.hf_dataset[source_from:source_to]
        actions = np.asarray(rows["action"], dtype=np.float32)
        states = np.asarray(rows[state_key], dtype=np.float32)
        window = detect_motion_window(actions, action_names, config)
        start, end = window.start, window.end
        override = overrides.get(episode_index, {})
        start = int(override.get("start", start))
        end = int(override.get("end", end))
        if not 0 <= start < end <= len(actions):
            raise ValueError(
                f"Invalid episode {episode_index} window [{start}, {end}) for {len(actions)} frames"
            )

        task_indices = rows.get("task_index", [0] * len(actions))
        task_index = int(task_indices[start])
        start_state = states[start]
        start_states.append(start_state)
        episode_plans.append(
            {
                "episode_index": episode_index,
                "source_from_index": source_from,
                "source_to_index": source_to,
                "source_frames": len(actions),
                "start_event": window.start_event,
                "start_event_found": window.start_event_found,
                "final_settle_event": window.final_settle_event,
                "final_settle_found": window.final_settle_found,
                "start": start,
                "end": end,
                "kept_frames": end - start,
                "removed_start_frames": start,
                "removed_end_frames": len(actions) - end,
                "task": task_by_index.get(task_index, str(task_index)),
                "start_state": start_state.astype(float).tolist(),
                "start_action": actions[start].astype(float).tolist(),
                "manual_override": bool(override),
            }
        )

    starts = np.stack(start_states)
    arm_indices, _ = joint_groups(action_names)
    arm_starts = starts[:, arm_indices]
    median_arm_start = np.median(arm_starts, axis=0)
    medoid_position = int(np.argmin(np.linalg.norm(arm_starts - median_arm_start, axis=1)))
    input_frames = int(sum(plan["source_frames"] for plan in episode_plans))
    output_frames = int(sum(plan["kept_frames"] for plan in episode_plans))
    requires_staging = config.start_mode != "motion" or config.start_offset_frames > 0

    return {
        "schema_version": 1,
        "source": {
            "repo_id": dataset.repo_id,
            "root": str(dataset.root.resolve()),
            "fps": dataset.fps,
            "episodes": dataset.meta.total_episodes,
            "frames": dataset.meta.total_frames,
            "fingerprint": _source_fingerprint(dataset.root),
        },
        "config": asdict(config),
        "summary": {
            "input_frames": input_frames,
            "output_frames": output_frames,
            "removed_frames": input_frames - output_frames,
            "removed_seconds": (input_frames - output_frames) / dataset.fps,
            "representative_start_episode": episode_plans[medoid_position]["episode_index"],
            "arm_start_std_rad": np.std(arm_starts, axis=0).astype(float).tolist(),
            "requires_staged_start": requires_staging,
            "start_guidance": (
                "Stage the robot near a demonstrated start_state before inference."
                if requires_staging
                else "Start at home; retained pre-motion context covers departure from home."
            ),
        },
        "episodes": episode_plans,
    }


def _write_manifest(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")


def _validate_plan_source(dataset: LeRobotDataset, plan: dict[str, Any]) -> None:
    expected = plan.get("source", {}).get("fingerprint")
    if expected and expected != _source_fingerprint(dataset.root):
        raise ValueError("The trim plan fingerprint does not match the loaded source dataset")
    if len(plan.get("episodes", [])) != dataset.meta.total_episodes:
        raise ValueError("The trim plan episode count does not match the source dataset")


def _decode_video_batch(
    dataset: LeRobotDataset,
    episode_index: int,
    timestamps: list[float],
) -> dict[str, Any]:
    metadata = dataset.meta.episodes[episode_index]

    def decode(key: str) -> tuple[str, Any]:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        shifted = [float(metadata[f"videos/{key}/from_timestamp"]) + value for value in timestamps]
        frames = decode_video_frames(
            video_path,
            shifted,
            dataset.tolerance_s,
            backend="pyav",
            return_uint8=True,
            is_depth=key in dataset.meta.depth_keys,
        )
        return key, frames

    with ThreadPoolExecutor(max_workers=max(1, len(dataset.meta.video_keys))) as executor:
        return dict(executor.map(decode, dataset.meta.video_keys))


def _coerce_feature(value: Any, feature: dict[str, Any]) -> Any:
    if feature["dtype"] == "string":
        return str(value)
    return np.asarray(value, dtype=np.dtype(feature["dtype"]))


def materialize_plan(
    source: LeRobotDataset,
    plan: dict[str, Any],
    output: Path,
    decode_batch_size: int = 32,
    image_writer_threads: int = 4,
) -> LeRobotDataset:
    """Apply the same frame windows to every dataset modality."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if decode_batch_size < 1:
        raise ValueError("decode_batch_size must be at least 1")
    _validate_plan_source(source, plan)

    user_features = {
        key: deepcopy(feature)
        for key, feature in source.meta.features.items()
        if key not in DEFAULT_FEATURES
    }
    rgb_encoder = None
    if source.meta.video_keys:
        first_video_key = source.meta.video_keys[0]
        rgb_encoder = RGBEncoderConfig.from_video_info(
            source.meta.features[first_video_key].get("info")
        )
    output_dataset = LeRobotDataset.create(
        repo_id=output.name,
        fps=source.fps,
        root=output,
        robot_type=source.meta.info.robot_type,
        features=user_features,
        use_videos=bool(source.meta.video_keys),
        video_backend="pyav",
        image_writer_threads=image_writer_threads,
        rgb_encoder=rgb_encoder,
        data_files_size_in_mb=source.meta.info.data_files_size_in_mb,
        video_files_size_in_mb=source.meta.info.video_files_size_in_mb,
    )
    task_by_index = _task_map(source)
    non_video_keys = [key for key in user_features if key not in source.meta.video_keys]

    for plan_index, episode_plan in enumerate(plan["episodes"]):
        episode_index = int(episode_plan["episode_index"])
        source_start = int(episode_plan["source_from_index"]) + int(episode_plan["start"])
        source_end = int(episode_plan["source_from_index"]) + int(episode_plan["end"])
        print(
            f"Episode {episode_index:03d}: source [{episode_plan['start']}, "
            f"{episode_plan['end']}) -> {source_end - source_start} frames"
        )
        for batch_start in range(source_start, source_end, decode_batch_size):
            batch_end = min(source_end, batch_start + decode_batch_size)
            rows = source.hf_dataset[batch_start:batch_end]
            timestamps = [float(value) for value in rows["timestamp"]]
            videos = _decode_video_batch(source, episode_index, timestamps)
            for offset in range(batch_end - batch_start):
                task_index = int(rows["task_index"][offset])
                frame = {
                    key: _coerce_feature(rows[key][offset], user_features[key])
                    for key in non_video_keys
                }
                frame.update({key: values[offset] for key, values in videos.items()})
                frame["task"] = task_by_index.get(task_index, str(task_index))
                output_dataset.add_frame(frame)
        output_dataset.save_episode(parallel_encoding=True)
        print(f"  saved {plan_index + 1}/{len(plan['episodes'])}")

    output_dataset.finalize()
    _write_manifest(plan, output / "meta" / "trim_manifest.json")
    return output_dataset


def _default_manifest_path(args: argparse.Namespace, dataset: LeRobotDataset) -> Path:
    if args.manifest:
        return args.manifest
    if args.output:
        return args.output.parent / f"{args.output.name}.trim-plan.json"
    return dataset.root.parent / f"{dataset.root.name}-{args.start_mode}.trim-plan.json"


def main() -> None:
    args = _build_parser().parse_args()
    if not args.dry_run and args.output is None:
        print("Error: --output is required unless --dry-run is used", file=sys.stderr)
        raise SystemExit(2)
    if args.apply_plan and args.dry_run:
        print("Error: --apply-plan cannot be combined with --dry-run", file=sys.stderr)
        raise SystemExit(2)
    if args.reference_plan and args.start_mode != "gripper":
        print("Error: --reference-plan requires --start-mode gripper", file=sys.stderr)
        raise SystemExit(2)
    if args.reference_plan and args.start_offset_frames is not None:
        print(
            "Error: --reference-plan and --start-offset-frames are mutually exclusive",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        source = _load_source(args)
        print(
            f"Loaded {source.repo_id}: {source.meta.total_episodes} episodes, "
            f"{source.meta.total_frames} frames at {source.fps} Hz"
        )
        if args.apply_plan:
            plan = json.loads(args.apply_plan.read_text())
            _validate_plan_source(source, plan)
        else:
            reference_frames = None
            reference_details = None
            if args.reference_plan:
                reference_frames, reference_details = _reference_preroll(args.reference_plan)
            config = _config_from_args(args, reference_frames)
            plan = build_trim_plan(source, config, _load_overrides(args.overrides))
            if reference_details is not None:
                plan["alignment_reference"] = reference_details

        manifest_path = _default_manifest_path(args, source)
        _write_manifest(plan, manifest_path)
        summary = plan["summary"]
        print(
            f"Plan: {summary['input_frames']} -> {summary['output_frames']} frames "
            f"({summary['removed_seconds']:.2f}s removed); manifest: {manifest_path}"
        )
        print(summary["start_guidance"])
        if not args.dry_run:
            assert args.output is not None
            output = args.output.resolve()
            materialize_plan(
                source,
                plan,
                output,
                decode_batch_size=args.decode_batch_size,
                image_writer_threads=args.image_writer_threads,
            )
            print(f"Created trimmed dataset: {output}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
