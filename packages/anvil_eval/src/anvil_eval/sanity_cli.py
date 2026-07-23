"""CLI for an offline, chunk-correct Pi0.5 policy sanity test."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .contract import audit_policy_contract, inspect_debug_images, resolve_pretrained_model
from .dataset import EvaluationDataset, get_episode_indices
from .evaluator import load_model
from .sanity import (
    NativeEpisodeResult,
    aggregate_sanity_metrics,
    compute_episode_sanity,
    evaluate_native_relative_episode,
    find_native_relative_step,
    simulate_sync_prefetch,
)
from .snapshot import evaluate_saved_snapshot

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline contract, policy-quality, and queue sanity test for Pi0.5"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--inference-config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--debug-image-dir")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Run only the saved-camera preprocessing/output/timing check",
    )
    parser.add_argument("--episodes", help="Comma-separated episode indices")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--num-eps", type=int, default=3)
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Evaluate every episode in the selected split instead of sampling --num-eps",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--task-description")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    if args.snapshot_only and not args.debug_image_dir:
        raise SystemExit("--snapshot-only requires --debug-image-dir")
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    inference_config_path = Path(args.inference_config)
    model_dir = resolve_pretrained_model(checkpoint_path)

    contract = audit_policy_contract(checkpoint_path, dataset_path, inference_config_path)
    if contract["errors"]:
        for error in contract["errors"]:
            log.error("Contract: %s", error)
        raise SystemExit("Contract audit failed; refusing to evaluate an incompatible policy")
    for warning in contract["warnings"]:
        log.warning("Contract: %s", warning)

    anvil_config_path = model_dir / "anvil_config.json"
    anvil_config = json.loads(anvil_config_path.read_text()) if anvil_config_path.exists() else {}
    task_description = (
        args.task_description
        or anvil_config.get("task_description")
        or _load_yaml(inference_config_path).get("model", {}).get("task_description")
    )

    eval_dataset = EvaluationDataset(dataset_path, video_backend=args.video_backend)
    split_info = eval_dataset.resolve_splits(anvil_config, checkpoint_path=checkpoint_path)
    episodes = _select_episodes(args, split_info, checkpoint_path, dataset_path)
    if not episodes:
        raise SystemExit("No episodes selected")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EvalConfig(checkpoint_path, dataset_path).resolve_output_dir().parent / "sanity"
    )
    arrays_dir = output_dir / "arrays"
    plots_dir = output_dir / "plots"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "contract_report.json", contract)

    camera_report = None
    if args.debug_image_dir:
        camera_report = inspect_debug_images(
            Path(args.debug_image_dir), contract["checks"]["camera_roles"]
        )
        _write_json(output_dir / "debug_camera_report.json", camera_report)

    log.info("Loading checkpoint %s", checkpoint_path)
    model, preprocessor, postprocessor, model_type = load_model(str(checkpoint_path), args.device)
    if model_type != "pi05":
        raise SystemExit(f"This sanity runner currently requires pi05, got {model_type}")
    if preprocessor is None or postprocessor is None:
        raise SystemExit("Checkpoint processor pipelines did not load")
    if find_native_relative_step(preprocessor) is None:
        raise SystemExit("Checkpoint does not contain an enabled native relative-action processor")

    max_delta = float(contract["checks"]["max_position_delta"])
    debug_snapshot_report = None
    if camera_report is not None:
        source_episode, source_split = episodes[0]
        source_frame = eval_dataset.get_episode_frames(source_episode)[0]
        source_state = eval_dataset.dataset.hf_dataset[source_frame]["observation.state"]
        debug_snapshot_report = evaluate_saved_snapshot(
            model=model,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            camera_report=camera_report,
            state=source_state,
            state_source=f"episode {source_episode} ({source_split}) frame 0",
            device=args.device,
            task_description=task_description,
            joint_names=eval_dataset.joint_names,
            max_position_delta=max_delta,
        )
        _write_json(output_dir / "debug_snapshot_report.json", debug_snapshot_report)
        if args.snapshot_only:
            log.info("Saved-camera snapshot test complete: %s", output_dir.resolve())
            return
    results: list[NativeEpisodeResult] = []
    episode_metrics: list[dict[str, Any]] = []
    for position, (episode_idx, split_label) in enumerate(episodes, start=1):
        log.info(
            "Evaluating episode %d (%s), %d/%d",
            episode_idx,
            split_label,
            position,
            len(episodes),
        )
        result = evaluate_native_relative_episode(
            model=model,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset.dataset,
            frame_indices=eval_dataset.get_episode_frames(episode_idx),
            episode_idx=episode_idx,
            split_label=split_label,
            device=args.device,
            task_description=task_description,
            joint_names=eval_dataset.joint_names,
        )
        metrics, limited = compute_episode_sanity(result, max_position_delta=max_delta)
        metrics["max_position_delta"] = max_delta
        results.append(result)
        episode_metrics.append(metrics)
        _save_episode_arrays(arrays_dir, result, limited)
        _plot_episode(plots_dir, result, limited, metrics)
        log.info(
            "Episode %d: model MAE %.4f, hold MAE %.4f, clamp %.1f%%",
            episode_idx,
            metrics["model_mae"],
            metrics["hold_mae"],
            100.0 * metrics["clamp_element_rate"],
        )

    summary = aggregate_sanity_metrics(results, episode_metrics)
    latencies = [latency for result in results for latency in result.inference_latencies]
    if debug_snapshot_report is not None:
        initial_latency = debug_snapshot_report["timing_ms"]["model"] / 1000.0
        steady_latencies = latencies
    else:
        initial_latency = latencies[0]
        steady_latencies = latencies[1:] or latencies
    scheduler = contract["checks"]["scheduler"]
    queue_report = simulate_sync_prefetch(
        steady_latencies,
        initial_latency=initial_latency,
        total_steps=sum(len(result.predicted) for result in results),
        control_hz=args.control_hz,
        chunk_size=int(scheduler["n_action_steps"] or scheduler["chunk_size"]),
        refill_threshold=int(scheduler["prefetch_threshold"]),
        replace_pending=bool(scheduler["replace_pending_actions"]),
    )
    decision = _deployment_decision(contract, summary, queue_report)

    report = {
        "checkpoint": str(model_dir),
        "dataset": str(dataset_path.resolve()),
        "inference_config": str(inference_config_path.resolve()),
        "task_description": task_description,
        "joint_names": eval_dataset.joint_names,
        "episodes": episode_metrics,
        "summary": summary,
        "queue_simulation": queue_report,
        "debug_snapshot": debug_snapshot_report,
        "decision": decision,
    }
    _write_json(output_dir / "sanity_report.json", report)
    _write_decision(output_dir / "GO_NO_GO.md", report)
    log.info("Sanity test complete: %s", decision["status"])
    log.info("Results: %s", output_dir.resolve())


def _select_episodes(
    args: argparse.Namespace,
    split_info: dict[str, list[int]],
    checkpoint_path: Path,
    dataset_path: Path,
) -> list[tuple[int, str]]:
    if args.episodes:
        selected: list[tuple[int, str]] = []
        for episode_idx in [int(value.strip()) for value in args.episodes.split(",")]:
            label = next(
                (name for name, values in split_info.items() if episode_idx in values),
                "manual",
            )
            selected.append((episode_idx, label))
        return selected

    if args.all_episodes:
        split_names = (
            [name for name in ("train", "val", "test") if name in split_info]
            if args.split == "all"
            else [args.split]
        )
        return sorted(
            (episode_idx, split_name)
            for split_name in split_names
            for episode_idx in split_info.get(split_name, [])
        )

    config = EvalConfig(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        num_episodes=args.num_eps,
        split=args.split,
        seed=args.seed,
    )
    return get_episode_indices(split_info, config)


def _save_episode_arrays(
    output_dir: Path,
    result: NativeEpisodeResult,
    limited: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / f"episode_{result.episode_idx:04d}_{result.split_label}.npz",
        predicted_absolute=result.predicted,
        ground_truth_absolute=result.ground_truth,
        hold_position=result.reference_states,
        observed_state=result.observation_states,
        limited_absolute=limited,
        predicted_relative=result.relative_output,
        ground_truth_relative=result.relative_ground_truth,
        normalized_model_output=result.normalized_output,
        chunk_starts=np.asarray(result.chunk_starts),
        inference_latencies=np.asarray(result.inference_latencies),
        preprocess_latencies=np.asarray(result.preprocess_latencies),
        postprocess_latencies=np.asarray(result.postprocess_latencies),
    )


def _plot_episode(
    output_dir: Path,
    result: NativeEpisodeResult,
    limited: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    count = len(result.joint_names)
    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows * 2, columns, figsize=(16, rows * 5), squeeze=False)
    frames = np.arange(len(result.predicted))
    for index, name in enumerate(result.joint_names):
        row, column = divmod(index, columns)
        absolute_axis = axes[row, column]
        absolute_axis.plot(
            frames, result.ground_truth[:, index], label="demonstration", linewidth=0.8
        )
        absolute_axis.plot(frames, result.predicted[:, index], label="policy", linewidth=0.8)
        absolute_axis.plot(frames, result.reference_states[:, index], label="hold", linewidth=0.6)
        absolute_axis.plot(frames, limited[:, index], label="limited", linewidth=0.6, alpha=0.8)
        absolute_axis.set_title(
            f"{name}\nMAE {metrics['per_joint'][name]['model_mae']:.3f} / "
            f"hold {metrics['per_joint'][name]['hold_mae']:.3f}",
            fontsize=8,
        )
        if index == 0:
            absolute_axis.legend(fontsize=6)

        relative_axis = axes[rows + row, column]
        relative_axis.plot(
            frames,
            result.relative_ground_truth[:, index],
            label="demonstration relative",
            linewidth=0.8,
        )
        relative_axis.plot(
            frames,
            result.relative_output[:, index],
            label="policy relative",
            linewidth=0.8,
        )
        relative_axis.set_title(f"{name} relative", fontsize=8)
        if index == 0:
            relative_axis.legend(fontsize=6)

    figure.suptitle(
        f"Episode {result.episode_idx} [{result.split_label}] — "
        f"policy MAE {metrics['model_mae']:.4f}, hold {metrics['hold_mae']:.4f}"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"episode_{result.episode_idx:04d}_{result.split_label}.png",
        dpi=120,
    )
    plt.close(figure)


def _deployment_decision(
    contract: dict[str, Any],
    summary: dict[str, Any],
    queue_report: dict[str, Any],
) -> dict[str, Any]:
    overall = summary["overall"]
    gates = {
        "contract_has_no_errors": not contract["errors"],
        "policy_beats_hold_overall": overall["model_beats_hold"],
        "shoulders_beat_hold": overall["shoulders_beat_hold"],
        "model_p95_within_refill_budget": (
            queue_report["model_latency_p95_ms"] <= queue_report["refill_budget_ms"]
        ),
        "no_post_warmup_queue_starvation": queue_report["post_warmup_starved_steps"] == 0,
        "clamp_element_rate_below_5_percent": overall["clamp_element_rate"] < 0.05,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "status": "GO_FOR_DEVBOX_SHADOW" if not failed else "NO_GO",
        "gates": gates,
        "failed_gates": failed,
        "retraining_recommended": bool(
            not overall["model_beats_hold"] or not overall["shoulders_beat_hold"]
        ),
    }


def _write_decision(path: Path, report: dict[str, Any]) -> None:
    decision = report["decision"]
    overall = report["summary"]["overall"]
    queue = report["queue_simulation"]
    lines = [
        f"# {decision['status']}",
        "",
        "## Offline evidence",
        "",
        f"- Policy MAE: {overall['model_mae']:.5f} rad",
        f"- Hold-position MAE: {overall['hold_mae']:.5f} rad",
        f"- Shoulder policy/hold MAE: {overall['shoulder_model_mae']:.5f} / {overall['shoulder_hold_mae']:.5f} rad",
        f"- Model latency p95: {queue['model_latency_p95_ms']:.1f} ms",
        f"- Refill budget: {queue['refill_budget_ms']:.1f} ms",
        f"- Simulated action rate: {queue['action_publication_hz']:.2f} Hz",
        f"- Post-warm-up starvation: {queue['post_warmup_starved_steps']} steps",
        f"- Safety clamp element rate: {100.0 * overall['clamp_element_rate']:.1f}%",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in decision["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Deferred devbox checks",
            "",
            "1. Jog every physical joint and verify its observation index, arm, and sign.",
            "2. Run disconnected-topic shadow inference with live cameras and joint states.",
            "3. Compare raw policy targets, limited commands, and measured movement before folding.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
