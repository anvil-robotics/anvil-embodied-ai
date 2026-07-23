"""Chunk-correct offline sanity checks for stateful relative-action policies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class NativeEpisodeResult:
    """Outputs and timings from one chunk-correct episode replay."""

    episode_idx: int
    split_label: str
    joint_names: list[str]
    predicted: np.ndarray
    ground_truth: np.ndarray
    relative_output: np.ndarray
    relative_ground_truth: np.ndarray
    normalized_output: np.ndarray
    observation_states: np.ndarray
    reference_states: np.ndarray
    chunk_starts: list[int]
    inference_latencies: list[float]
    preprocess_latencies: list[float]
    postprocess_latencies: list[float]


def find_native_relative_step(preprocessor: Any) -> Any | None:
    """Return the enabled LeRobot relative-action processor, when present."""
    for step in getattr(preprocessor, "steps", ()) if preprocessor else ():
        registry_name = getattr(step.__class__, "_registry_name", "")
        if (
            registry_name == "relative_actions_processor"
            or step.__class__.__name__ == "RelativeActionsProcessorStep"
        ) and bool(getattr(step, "enabled", False)):
            return step
    return None


def evaluate_native_relative_episode(
    *,
    model: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    frame_indices: list[int],
    episode_idx: int,
    split_label: str,
    device: str,
    task_description: str | None,
    joint_names: list[str],
) -> NativeEpisodeResult:
    """Replay one episode without invalidating the relative-action state cache.

    The preprocessor is called exactly once per predicted chunk. The whole chunk is
    postprocessed immediately, before another state can replace the cached reference.
    """
    relative_step = find_native_relative_step(preprocessor)
    if relative_step is None:
        raise RuntimeError("Checkpoint does not have an enabled relative_actions_processor")
    if postprocessor is None:
        raise RuntimeError("Checkpoint postprocessor is required for absolute action restoration")
    if not hasattr(model, "predict_action_chunk"):
        raise RuntimeError("Policy does not implement predict_action_chunk")

    if hasattr(model, "reset"):
        model.reset()

    n_action_steps = int(
        getattr(getattr(model, "config", None), "n_action_steps", 0)
        or getattr(getattr(model, "config", None), "chunk_size", 0)
        or 1
    )
    predicted_chunks: list[np.ndarray] = []
    ground_truth_chunks: list[np.ndarray] = []
    relative_chunks: list[np.ndarray] = []
    relative_gt_chunks: list[np.ndarray] = []
    normalized_chunks: list[np.ndarray] = []
    observation_chunks: list[np.ndarray] = []
    reference_chunks: list[np.ndarray] = []
    chunk_starts: list[int] = []
    inference_latencies: list[float] = []
    preprocess_latencies: list[float] = []
    postprocess_latencies: list[float] = []

    offsets = range(0, len(frame_indices), n_action_steps)
    for chunk_start in tqdm(offsets, desc=f"Episode {episode_idx}", leave=False):
        item = dataset[frame_indices[chunk_start]]
        observation = {key: value for key, value in item.items() if key.startswith("observation.")}
        ref_state = _to_numpy_vector(item["observation.state"])

        batch = dict(observation)
        if task_description:
            batch["task"] = [task_description]

        _synchronize(device)
        stage_start = time.perf_counter()
        processed = preprocessor(batch)
        processed = _move_to_device(processed, device)
        _synchronize(device)
        preprocess_latencies.append(time.perf_counter() - stage_start)

        _synchronize(device)
        stage_start = time.perf_counter()
        with torch.inference_mode():
            raw_chunk = model.predict_action_chunk(processed)
        _synchronize(device)
        inference_latencies.append(time.perf_counter() - stage_start)

        normalized_chunk = _to_numpy_matrix(raw_chunk)
        post_input = (
            raw_chunk.squeeze(0)
            if isinstance(raw_chunk, torch.Tensor) and raw_chunk.dim() == 3
            else raw_chunk
        )
        _synchronize(device)
        stage_start = time.perf_counter()
        absolute_chunk = postprocessor.process_action(post_input)
        _synchronize(device)
        postprocess_latencies.append(time.perf_counter() - stage_start)
        absolute_chunk = _to_numpy_matrix(absolute_chunk)

        remaining = len(frame_indices) - chunk_start
        executable = min(n_action_steps, remaining, len(absolute_chunk))
        if executable <= 0:
            raise RuntimeError("predict_action_chunk returned no executable actions")

        absolute_chunk = absolute_chunk[:executable]
        normalized_chunk = normalized_chunk[:executable]
        target_indices = frame_indices[chunk_start : chunk_start + executable]
        gt_chunk, state_chunk = _read_action_state_rows(dataset, target_indices)
        ref_chunk = np.repeat(ref_state[np.newaxis, :], executable, axis=0)
        relative_chunk = _to_relative(absolute_chunk, ref_state, relative_step)
        relative_gt = _to_relative(gt_chunk, ref_state, relative_step)

        chunk_starts.append(sum(len(chunk) for chunk in predicted_chunks))
        predicted_chunks.append(absolute_chunk)
        ground_truth_chunks.append(gt_chunk)
        relative_chunks.append(relative_chunk)
        relative_gt_chunks.append(relative_gt)
        normalized_chunks.append(normalized_chunk)
        observation_chunks.append(state_chunk)
        reference_chunks.append(ref_chunk)

    return NativeEpisodeResult(
        episode_idx=episode_idx,
        split_label=split_label,
        joint_names=joint_names,
        predicted=np.concatenate(predicted_chunks),
        ground_truth=np.concatenate(ground_truth_chunks),
        relative_output=np.concatenate(relative_chunks),
        relative_ground_truth=np.concatenate(relative_gt_chunks),
        normalized_output=np.concatenate(normalized_chunks),
        observation_states=np.concatenate(observation_chunks),
        reference_states=np.concatenate(reference_chunks),
        chunk_starts=chunk_starts,
        inference_latencies=inference_latencies,
        preprocess_latencies=preprocess_latencies,
        postprocess_latencies=postprocess_latencies,
    )


def compute_episode_sanity(
    result: NativeEpisodeResult,
    *,
    max_position_delta: float,
    direction_epsilon: float = 1e-4,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compare the policy with demonstrations, hold-position, and the live limiter."""
    predicted = result.predicted
    target = result.ground_truth
    hold = result.reference_states
    observation = result.observation_states
    limited = observation + np.clip(
        predicted - observation,
        -max_position_delta,
        max_position_delta,
    )
    clamped = np.abs(predicted - observation) > max_position_delta

    model_error = np.abs(predicted - target)
    hold_error = np.abs(hold - target)
    limited_error = np.abs(limited - target)
    predicted_displacement = predicted - hold
    target_displacement = target - hold

    per_joint: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(result.joint_names):
        active = np.abs(target_displacement[:, index]) > direction_epsilon
        direction = (
            float(
                np.mean(
                    np.sign(predicted_displacement[active, index])
                    == np.sign(target_displacement[active, index])
                )
            )
            if np.any(active)
            else None
        )
        per_joint[name] = {
            "model_mae": float(np.mean(model_error[:, index])),
            "hold_mae": float(np.mean(hold_error[:, index])),
            "limited_mae": float(np.mean(limited_error[:, index])),
            "displacement_correlation": _safe_correlation(
                predicted_displacement[:, index], target_displacement[:, index]
            ),
            "direction_agreement": direction,
            "clamp_element_rate": float(np.mean(clamped[:, index])),
        }

    endpoints = _chunk_endpoints(result.chunk_starts, len(predicted))
    shoulder_indices = [
        index
        for index, name in enumerate(result.joint_names)
        if any(f"joint_{joint}." in name for joint in (1, 2, 3))
    ]
    metrics = {
        "episode_idx": result.episode_idx,
        "split_label": result.split_label,
        "num_frames": int(len(predicted)),
        "num_predictions": len(result.inference_latencies),
        "model_mae": float(np.mean(model_error)),
        "hold_mae": float(np.mean(hold_error)),
        "limited_mae": float(np.mean(limited_error)),
        "model_beats_hold": bool(np.mean(model_error) < np.mean(hold_error)),
        "shoulder_model_mae": float(np.mean(model_error[:, shoulder_indices])),
        "shoulder_hold_mae": float(np.mean(hold_error[:, shoulder_indices])),
        "shoulders_beat_hold": bool(
            np.mean(model_error[:, shoulder_indices]) < np.mean(hold_error[:, shoulder_indices])
        ),
        "endpoint_mae": float(np.mean(model_error[endpoints])),
        "direction_agreement": _direction_agreement(
            predicted_displacement, target_displacement, direction_epsilon
        ),
        "shoulder_direction_agreement": _direction_agreement(
            predicted_displacement[:, shoulder_indices],
            target_displacement[:, shoulder_indices],
            direction_epsilon,
        ),
        "clamp_element_rate": float(np.mean(clamped)),
        "clamp_frame_rate": float(np.mean(np.any(clamped, axis=1))),
        "timing_ms": {
            "preprocess_mean": _milliseconds(np.mean(result.preprocess_latencies)),
            "model_mean": _milliseconds(np.mean(result.inference_latencies)),
            "model_p50": _milliseconds(np.percentile(result.inference_latencies, 50)),
            "model_p95": _milliseconds(np.percentile(result.inference_latencies, 95)),
            "postprocess_mean": _milliseconds(np.mean(result.postprocess_latencies)),
        },
        "per_joint": per_joint,
    }
    return metrics, limited


def aggregate_sanity_metrics(
    results: list[NativeEpisodeResult],
    episode_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build weighted overall and split summaries from episode outputs."""
    groups: dict[str, list[int]] = {"overall": list(range(len(results)))}
    for index, result in enumerate(results):
        groups.setdefault(result.split_label, []).append(index)

    summary: dict[str, Any] = {}
    for label, indices in groups.items():
        predicted = np.concatenate([results[index].predicted for index in indices])
        target = np.concatenate([results[index].ground_truth for index in indices])
        hold = np.concatenate([results[index].reference_states for index in indices])
        observation = np.concatenate([results[index].observation_states for index in indices])
        max_delta = float(episode_metrics[indices[0]]["max_position_delta"])
        clamped = np.abs(predicted - observation) > max_delta
        model_error = np.abs(predicted - target)
        hold_error = np.abs(hold - target)
        shoulder_indices = [
            joint_index
            for joint_index, name in enumerate(results[indices[0]].joint_names)
            if any(f"joint_{joint}." in name for joint in (1, 2, 3))
        ]
        latencies = [latency for index in indices for latency in results[index].inference_latencies]
        summary[label] = {
            "episodes": len(indices),
            "frames": int(len(predicted)),
            "predictions": len(latencies),
            "model_mae": float(np.mean(model_error)),
            "hold_mae": float(np.mean(hold_error)),
            "model_beats_hold": bool(np.mean(model_error) < np.mean(hold_error)),
            "shoulder_model_mae": float(np.mean(model_error[:, shoulder_indices])),
            "shoulder_hold_mae": float(np.mean(hold_error[:, shoulder_indices])),
            "shoulders_beat_hold": bool(
                np.mean(model_error[:, shoulder_indices]) < np.mean(hold_error[:, shoulder_indices])
            ),
            "clamp_element_rate": float(np.mean(clamped)),
            "clamp_frame_rate": float(np.mean(np.any(clamped, axis=1))),
            "model_latency_ms": {
                "mean": _milliseconds(np.mean(latencies)),
                "p50": _milliseconds(np.percentile(latencies, 50)),
                "p95": _milliseconds(np.percentile(latencies, 95)),
                "p99": _milliseconds(np.percentile(latencies, 99)),
            },
        }
    return summary


def simulate_sync_prefetch(
    latencies: list[float],
    *,
    total_steps: int,
    control_hz: float,
    chunk_size: int,
    refill_threshold: int,
    replace_pending: bool,
    initial_latency: float | None = None,
) -> dict[str, Any]:
    """Discrete-event simulation of inference_node synchronous chunk prefetch."""
    if not latencies:
        raise ValueError("At least one latency sample is required")
    if control_hz <= 0 or chunk_size <= 0 or total_steps <= 0:
        raise ValueError("control_hz, chunk_size, and total_steps must be positive")

    queue_depth = 0
    in_flight_until: float | None = None
    latency_index = 0
    first_fill = False
    starved = 0
    post_warmup_starved = 0
    published = 0
    predictions_started = 0
    replaced_actions = 0
    min_depth_after_warmup: int | None = None
    tick_seconds = 1.0 / control_hz

    for step in range(total_steps):
        now = step * tick_seconds
        if in_flight_until is not None and in_flight_until <= now:
            if replace_pending:
                replaced_actions += queue_depth
                queue_depth = 0
            queue_depth += chunk_size
            in_flight_until = None
            first_fill = True

        if queue_depth > 0:
            queue_depth -= 1
            published += 1
            if first_fill:
                min_depth_after_warmup = (
                    queue_depth
                    if min_depth_after_warmup is None
                    else min(min_depth_after_warmup, queue_depth)
                )
        else:
            starved += 1
            if first_fill:
                post_warmup_starved += 1

        if in_flight_until is None and queue_depth <= refill_threshold:
            if predictions_started == 0 and initial_latency is not None:
                latency = initial_latency
            else:
                latency = latencies[latency_index % len(latencies)]
                latency_index += 1
            predictions_started += 1
            in_flight_until = now + latency

    duration = total_steps / control_hz
    return {
        "control_hz": control_hz,
        "duration_seconds": duration,
        "chunk_size": chunk_size,
        "refill_threshold": refill_threshold,
        "replace_pending_actions": replace_pending,
        "initial_warmup_latency_ms": (
            _milliseconds(initial_latency) if initial_latency is not None else None
        ),
        "prediction_rate_hz": predictions_started / duration,
        "raw_model_capacity_hz": 1.0 / float(np.mean(latencies)),
        "action_publication_hz": published / duration,
        "starved_steps_including_warmup": starved,
        "post_warmup_starved_steps": post_warmup_starved,
        "minimum_queue_depth_after_warmup": min_depth_after_warmup,
        "replaced_actions": replaced_actions,
        "refill_budget_ms": 1000.0 * refill_threshold / control_hz,
        "model_latency_p95_ms": _milliseconds(np.percentile(latencies, 95)),
    }


def _read_action_state_rows(dataset: Any, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    table = getattr(dataset, "hf_dataset", None)
    rows = [table[index] if table is not None else dataset[index] for index in indices]
    actions = np.stack([_to_numpy_vector(row["action"]) for row in rows])
    states = np.stack([_to_numpy_vector(row["observation.state"]) for row in rows])
    return actions, states


def _to_relative(actions: np.ndarray, ref_state: np.ndarray, relative_step: Any) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    action_dim = result.shape[-1]
    mask = np.asarray(relative_step._build_mask(action_dim), dtype=bool)
    dims = min(action_dim, len(ref_state), len(mask))
    result[..., :dims][..., mask[:dims]] -= ref_state[:dims][mask[:dims]]
    return result


def _move_to_device(data: Any, device: str) -> Any:
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _to_numpy_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _to_numpy_matrix(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim == 1:
        result = result[np.newaxis, :]
    if result.ndim != 2:
        raise ValueError(f"Expected an action matrix, got shape {result.shape}")
    return result


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _direction_agreement(predicted: np.ndarray, target: np.ndarray, epsilon: float) -> float | None:
    active = np.abs(target) > epsilon
    if not np.any(active):
        return None
    return float(np.mean(np.sign(predicted[active]) == np.sign(target[active])))


def _chunk_endpoints(starts: list[int], length: int) -> np.ndarray:
    ends = [next_start - 1 for next_start in starts[1:]]
    ends.append(length - 1)
    return np.asarray(ends, dtype=np.int64)


def _milliseconds(seconds: Any) -> float:
    return float(seconds) * 1000.0
