"""Offline cache, supervised training, and evaluation for embodiment adapters."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from anvil_eval.dataset import EvaluationDataset
from anvil_eval.evaluator import load_model
from anvil_shared.embodiment import EmbodimentError
from torch import Tensor

from .artifact import AdapterArtifact, load_adapter_artifact, sha256_file
from .kinematics import torch_forward_kinematics
from .policy import EmbodimentAdaptedPolicy
from .residual import ACTIVE_JOINT_INDICES, AdapterLossWeights, compute_adapter_loss


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _pretrained_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if (candidate / "config.json").is_file():
        return candidate
    nested = candidate / "pretrained_model"
    if (nested / "config.json").is_file():
        return nested
    raise FileNotFoundError(f"no pretrained_model config under {candidate}")


def _register_processor_compat_aliases() -> None:
    """Allow pinned LeRobot checkpoints to load across processor renames."""
    from lerobot.processor.pipeline import ProcessorStepRegistry
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    names = ProcessorStepRegistry.list()
    if "delta_actions_processor" in names:
        return
    canonical_name = getattr(
        RelativeActionsProcessorStep,
        "_registry_name",
        "relative_actions_processor",
    )
    ProcessorStepRegistry.register("delta_actions_processor")(RelativeActionsProcessorStep)
    # Registration annotates the shared class. Keep new artifacts on the
    # installed LeRobot release's canonical registry name.
    RelativeActionsProcessorStep._registry_name = canonical_name


def _load_policy(path: str | Path, device: str) -> tuple[Any, Any, Any]:
    _register_processor_compat_aliases()
    model, preprocessor, postprocessor, _ = load_model(str(_pretrained_dir(path)), device)
    if preprocessor is None or postprocessor is None:
        raise EmbodimentError(f"policy at {path} is missing processor pipelines")
    return model, preprocessor, postprocessor


@torch.no_grad()
def _predict_direct_chunk(
    *,
    model: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    processed = _move_to_device(preprocessor(observation), device)
    raw = model.predict_action_chunk(processed)
    post_input = raw.squeeze(0) if isinstance(raw, Tensor) and raw.ndim == 3 else raw
    absolute = _numpy(postprocessor.process_action(post_input))
    if absolute.ndim == 3 and absolute.shape[0] == 1:
        absolute = absolute[0]
    if absolute.ndim != 2 or absolute.shape[1] != 16:
        raise EmbodimentError(f"policy returned an invalid chunk shape: {absolute.shape}")
    return absolute


def _load_splits(path: str | Path, episode_count: int) -> dict[int, str]:
    raw = json.loads(Path(path).read_text())
    result: dict[int, str] = {}
    for split, key in (
        ("train", "train_episodes"),
        ("val", "val_episodes"),
        ("test", "test_episodes"),
    ):
        for episode in raw.get(key, []):
            episode = int(episode)
            if episode < 0 or episode >= episode_count or episode in result:
                raise EmbodimentError(f"invalid or duplicate split episode: {episode}")
            result[episode] = split
    if len(result) != episode_count:
        missing = sorted(set(range(episode_count)) - set(result))
        raise EmbodimentError(f"split_info does not cover every episode; missing={missing}")
    return result


def _target_chunk(dataset: Any, frame_indices: list[int], offset: int, size: int) -> np.ndarray:
    last = len(frame_indices) - 1
    rows = []
    for horizon in range(size):
        index = frame_indices[min(offset + horizon, last)]
        rows.append(np.asarray(dataset.hf_dataset[index]["action"], dtype=np.float32))
    return np.stack(rows)


def cache_policy_predictions(
    *,
    manifest: str | Path,
    base_policy: str | Path,
    dataset_path: str | Path,
    split_info: str | Path,
    output: str | Path,
    task: str,
    device: str = "cuda",
    stride: int = 10,
    seed: int = 42,
    video_backend: str = "pyav",
    baseline_policy: str | Path | None = None,
) -> dict[str, Any]:
    """Cache frozen folding predictions after deterministic kinematic bridging."""
    if stride < 1:
        raise ValueError("stride must be positive")
    artifact = load_adapter_artifact(
        manifest,
        base_policy_dir=_pretrained_dir(base_policy),
        device=device,
        require_weights=False,
    )
    model, preprocessor, postprocessor = _load_policy(base_policy, device)
    adapted = EmbodimentAdaptedPolicy(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        artifact=artifact,
        device=device,
    )
    baseline = _load_policy(baseline_policy, device) if baseline_policy else None
    dataset_wrapper = EvaluationDataset(Path(dataset_path), video_backend=video_backend)
    dataset = dataset_wrapper.dataset
    episode_splits = _load_splits(split_info, dataset_wrapper.total_episodes)
    device_value = torch.device(device)

    current_rows: list[np.ndarray] = []
    bridge_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    episode_rows: list[int] = []
    frame_rows: list[int] = []
    split_rows: list[str] = []
    rejected: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for episode in range(dataset_wrapper.total_episodes):
        frames = dataset_wrapper.get_episode_frames(episode)
        adapted.reset()
        if baseline is not None and callable(getattr(baseline[0], "reset", None)):
            baseline[0].reset()
        for offset in range(0, len(frames), stride):
            frame = frames[offset]
            item = dataset[frame]
            observation = {
                key: value for key, value in item.items() if key.startswith("observation.")
            }
            observation["task"] = [task]
            sample_seed = seed + episode * 1_000_003 + int(_numpy(item["frame_index"]))
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            try:
                bridge_chunk = _numpy(adapted.predict_action_chunk(observation))[0]
                target_chunk = _target_chunk(
                    dataset, frames, offset, artifact.spec.residual.chunk_size
                )
                if baseline is not None:
                    torch.manual_seed(sample_seed)
                    direct = _predict_direct_chunk(
                        model=baseline[0],
                        preprocessor=baseline[1],
                        postprocessor=baseline[2],
                        observation=observation,
                        device=device_value,
                    )
                    if len(direct) < artifact.spec.residual.chunk_size:
                        direct = np.concatenate(
                            [
                                direct,
                                np.repeat(
                                    direct[-1:],
                                    artifact.spec.residual.chunk_size - len(direct),
                                    axis=0,
                                ),
                            ]
                        )
                    baseline_rows.append(direct[: artifact.spec.residual.chunk_size])
            except Exception as exc:
                rejected.append({"episode": episode, "frame": frame, "error": str(exc)})
                continue
            current_rows.append(_numpy(item["observation.state"]).astype(np.float32))
            bridge_rows.append(bridge_chunk.astype(np.float32))
            target_rows.append(target_chunk.astype(np.float32))
            episode_rows.append(episode)
            frame_rows.append(int(_numpy(item["frame_index"])))
            split_rows.append(episode_splits[episode])

    if not current_rows:
        raise EmbodimentError("every cache sample was rejected")
    arrays: dict[str, np.ndarray] = {
        "current_state": np.stack(current_rows),
        "bridge_chunk": np.stack(bridge_rows),
        "target_chunk": np.stack(target_rows),
        "episode_index": np.asarray(episode_rows, dtype=np.int64),
        "frame_index": np.asarray(frame_rows, dtype=np.int64),
        "split": np.asarray(split_rows),
    }
    if baseline_rows:
        arrays["baseline_chunk"] = np.stack(baseline_rows).astype(np.float32)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_rows:
        np.savez_compressed(
            output_path,
            current_state=arrays["current_state"],
            bridge_chunk=arrays["bridge_chunk"],
            target_chunk=arrays["target_chunk"],
            episode_index=arrays["episode_index"],
            frame_index=arrays["frame_index"],
            split=arrays["split"],
            baseline_chunk=arrays["baseline_chunk"],
        )
    else:
        np.savez_compressed(
            output_path,
            current_state=arrays["current_state"],
            bridge_chunk=arrays["bridge_chunk"],
            target_chunk=arrays["target_chunk"],
            episode_index=arrays["episode_index"],
            frame_index=arrays["frame_index"],
            split=arrays["split"],
        )
    report = {
        "schema_version": 1,
        "samples": len(current_rows),
        "rejected_samples": len(rejected),
        "rejected": rejected,
        "stride": stride,
        "seed": seed,
        "seconds": time.perf_counter() - start_time,
        "base_policy": str(_pretrained_dir(base_policy)),
        "baseline_policy": str(_pretrained_dir(baseline_policy)) if baseline_policy else None,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def _cache_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {"current_state", "bridge_chunk", "target_chunk", "split"}
    if not required.issubset(arrays):
        raise EmbodimentError(f"adapter cache is missing {sorted(required - arrays.keys())}")
    if arrays["bridge_chunk"].shape != arrays["target_chunk"].shape:
        raise EmbodimentError("bridge and target cache shapes differ")
    return arrays


def train_residual_adapter(
    *,
    manifest: str | Path,
    cache: str | Path,
    output: str | Path,
    device: str = "cuda",
    steps: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    eval_every: int = 100,
    seed: int = 42,
    loss_weights: AdapterLossWeights = AdapterLossWeights(),
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
) -> dict[str, Any]:
    """Train only the bounded target-space residual; the VLA remains frozen."""
    if min(steps, batch_size, eval_every) < 1:
        raise ValueError("steps, batch_size, and eval_every must be positive")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache_path = Path(cache)
    cache_hash = sha256_file(cache_path)
    arrays = _cache_arrays(cache_path)
    train_indices = np.flatnonzero(arrays["split"] == "train")
    val_indices = np.flatnonzero(arrays["split"] == "val")
    if not len(train_indices):
        raise EmbodimentError("cache contains no training samples")
    if not len(val_indices):
        val_indices = train_indices

    artifact = load_adapter_artifact(manifest, device=device, require_weights=False)
    residual = artifact.residual
    residual.train()
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    device_value = torch.device(device)
    current = torch.as_tensor(arrays["current_state"], dtype=torch.float32)
    bridge = torch.as_tensor(arrays["bridge_chunk"], dtype=torch.float32)
    target = torch.as_tensor(arrays["target_chunk"], dtype=torch.float32)
    target_ranges = torch.as_tensor(
        artifact.bridge.target_joint_ranges, dtype=torch.float32, device=device_value
    )
    correction_bounds = artifact.residual.correction_bounds.to(device_value)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    wandb_run: Any | None = None
    if wandb_project is not None and wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"{artifact.spec.adapter_id}-seed-{seed}",
            job_type="embodiment-adapter",
            mode=wandb_mode,
            tags=["embodiment-adapter", "shirt-fold", "offline-only"],
            config={
                "adapter_id": artifact.spec.adapter_id,
                "deployment_status": artifact.spec.deployment_status,
                "manifest": str(Path(manifest).resolve()),
                "cache": str(cache_path.resolve()),
                "cache_sha256": cache_hash,
                "train_samples": int(len(train_indices)),
                "val_samples": int(len(val_indices)),
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "eval_every": eval_every,
                "seed": seed,
                "loss_weights": asdict(loss_weights),
                "residual_contract": asdict(artifact.spec.residual),
            },
        )
        input_artifact = wandb.Artifact(
            f"{artifact.spec.adapter_id}-inputs-{cache_hash[:8]}",
            type="embodiment-adapter-inputs",
            metadata={
                "cache_sha256": cache_hash,
                "deployment_status": artifact.spec.deployment_status,
            },
        )
        input_artifact.add_file(str(Path(manifest).resolve()), name="manifest.json")
        input_artifact.add_file(str(cache_path.resolve()), name="cache.npz")
        cache_report = cache_path.with_suffix(cache_path.suffix + ".json")
        if cache_report.is_file():
            input_artifact.add_file(str(cache_report.resolve()), name="cache-report.json")
        wandb_run.log_artifact(input_artifact)

    def evaluate(indices: np.ndarray) -> tuple[float, dict[str, float]]:
        residual.eval()
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                state_batch = current[selected].to(device_value)
                bridge_batch = bridge[selected].to(device_value)
                target_batch = target[selected].to(device_value)
                corrected, correction = residual(state_batch, bridge_batch)
                loss, terms = compute_adapter_loss(
                    corrected=corrected,
                    residual=correction,
                    target=target_batch,
                    target_ranges=target_ranges,
                    target_model=artifact.bridge.target_spec,
                    correction_bounds=correction_bounds,
                    weights=loss_weights,
                )
                size = len(selected)
                count += size
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * size
        residual.train()
        metrics = {name: value / count for name, value in totals.items()}
        return metrics["loss"], metrics

    def evaluate_quality(indices: np.ndarray) -> dict[str, float]:
        residual.eval()
        corrected_rows: list[np.ndarray] = []
        correction_rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                corrected, correction = residual(
                    current[selected].to(device_value),
                    bridge[selected].to(device_value),
                )
                corrected_rows.append(_numpy(corrected))
                correction_rows.append(_numpy(correction))
        residual.train()

        corrected_values = np.concatenate(corrected_rows)
        correction_values = np.concatenate(correction_rows)
        selected_current = arrays["current_state"][indices]
        selected_bridge = arrays["bridge_chunk"][indices]
        selected_target = arrays["target_chunk"][indices]
        adapter_metrics = _prediction_metrics(
            corrected_values,
            selected_target,
            selected_current,
            artifact,
        )
        bridge_metrics = _prediction_metrics(
            selected_bridge,
            selected_target,
            selected_current,
            artifact,
        )
        active = np.asarray(ACTIVE_JOINT_INDICES)
        active_bounds = _numpy(correction_bounds)[active]
        bound_fraction = np.abs(correction_values[..., active]) / np.maximum(
            active_bounds,
            1e-8,
        )
        quality: dict[str, float] = {
            "residual_mean_bound_fraction": float(np.mean(bound_fraction)),
            "residual_max_bound_fraction": float(np.max(bound_fraction)),
            "residual_saturation_fraction": float(np.mean(bound_fraction >= 0.95)),
        }
        for name, value in adapter_metrics.items():
            quality[f"adapter_{name}"] = value
        for name, value in bridge_metrics.items():
            quality[f"bridge_{name}"] = value
        error_metrics = (
            "normalized_joint_mae",
            "joint_rmse_rad",
            "shoulder_mae_rad",
            "tcp_position_mae_m",
            "tcp_orientation_mae_deg",
        )
        for name in error_metrics:
            quality[f"improvement_{name}"] = bridge_metrics[name] - adapter_metrics[name]
        quality["improvement_motion_ratio"] = abs(bridge_metrics["motion_ratio"] - 1.0) - abs(
            adapter_metrics["motion_ratio"] - 1.0
        )
        quality["quality_gate_joint"] = float(quality["improvement_normalized_joint_mae"] > 0.0)
        quality["quality_gate_shoulder"] = float(quality["improvement_shoulder_mae_rad"] > 0.0)
        quality["quality_gate_tcp"] = float(quality["improvement_tcp_position_mae_m"] > 0.0)
        quality["quality_gate_motion"] = float(quality["improvement_motion_ratio"] >= 0.0)
        quality["quality_gate_residual"] = float(quality["residual_saturation_fraction"] <= 0.05)
        quality["quality_gate_pass"] = float(
            quality["quality_gate_joint"]
            and quality["quality_gate_shoulder"]
            and quality["quality_gate_tcp"]
            and quality["quality_gate_motion"]
            and quality["quality_gate_residual"]
        )
        return quality

    for step in range(1, steps + 1):
        sampled = train_indices[
            torch.randint(len(train_indices), (batch_size,), generator=generator).numpy()
        ]
        state_batch = current[sampled].to(device_value)
        bridge_batch = bridge[sampled].to(device_value)
        target_batch = target[sampled].to(device_value)
        corrected, correction = residual(state_batch, bridge_batch)
        loss, train_terms = compute_adapter_loss(
            corrected=corrected,
            residual=correction,
            target=target_batch,
            target_ranges=target_ranges,
            target_model=artifact.bridge.target_spec,
            correction_bounds=correction_bounds,
            weights=loss_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            val_loss, metrics = evaluate(val_indices)
            quality = evaluate_quality(val_indices)
            record = {"step": float(step), **metrics, **quality}
            history.append(record)
            print(
                json.dumps(
                    {
                        "event": "adapter_validation",
                        "step": step,
                        "train_loss": float(train_terms["loss"]),
                        **record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{name}": float(value) for name, value in train_terms.items()},
                        **{f"val/{name}": value for name, value in metrics.items()},
                        **{f"quality/val/{name}": value for name, value in quality.items()},
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in residual.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    residual.load_state_dict(best_state)
    artifact.residual.eval()
    provenance = {
        "schema_version": 1,
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_hash,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_step": best_step,
        "best_val_loss": best_loss,
        "loss_weights": asdict(loss_weights),
        "history": history,
        "wandb": (
            {
                "project": wandb_project,
                "entity": wandb_entity,
                "run_id": wandb_run.id,
                "run_name": wandb_run.name,
                "run_path": wandb_run.path,
                "mode": wandb_mode,
            }
            if wandb_run is not None
            else None
        ),
    }
    output_path = Path(output)
    artifact.save(output_path, training_provenance=provenance)
    evaluation_path = output_path / "offline_evaluation.json"
    final_report = evaluate_adapter_cache(
        adapter=output_path,
        cache=cache_path,
        device=device,
        output=evaluation_path,
    )
    if wandb_run is not None:
        final_metrics: dict[str, float] = {}
        for split, split_values in final_report["splits"].items():
            for policy_name, policy_values in split_values.items():
                if isinstance(policy_values, dict):
                    for metric_name, value in policy_values.items():
                        final_metrics[f"final/{split}/{policy_name}/{metric_name}"] = float(value)
                else:
                    final_metrics[f"final/{split}/{policy_name}"] = float(policy_values)
        wandb_run.log(final_metrics, step=steps)
        best_record = next(item for item in history if int(item["step"]) == best_step)
        wandb_run.summary["best_step"] = best_step
        wandb_run.summary["best_val_loss"] = best_loss
        wandb_run.summary["quality_gate_pass"] = best_record["quality_gate_pass"]
        for name, value in best_record.items():
            if name.startswith(("adapter_", "bridge_", "improvement_", "residual_")):
                wandb_run.summary[f"best/{name}"] = value

        import wandb

        model_artifact = wandb.Artifact(
            f"{artifact.spec.adapter_id}-model-{wandb_run.id}",
            type="model",
            metadata={
                "cache_sha256": cache_hash,
                "best_step": best_step,
                "best_val_loss": best_loss,
                "deployment_status": artifact.spec.deployment_status,
            },
        )
        model_artifact.add_dir(str(output_path.resolve()))
        wandb_run.log_artifact(model_artifact)
        wandb_run.finish()
    return provenance


def _geodesic_degrees(predicted: Tensor, target: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = torch.clamp(
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return torch.rad2deg(torch.acos(cosine))


def _prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    artifact: AdapterArtifact,
) -> dict[str, float]:
    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    expected = torch.as_tensor(target, dtype=torch.float64)
    active = torch.tensor(ACTIVE_JOINT_INDICES)
    ranges = torch.as_tensor(artifact.bridge.target_joint_ranges, dtype=torch.float64)
    widths = (ranges[:, 1] - ranges[:, 0]).clamp_min(1e-8)
    error = predicted - expected
    normalized_mae = torch.mean(torch.abs(error[..., active]) / widths[active])
    shoulder = torch.tensor([0, 1, 8, 9])
    shoulder_mae = torch.mean(torch.abs(error[..., shoulder]))
    current_tensor = torch.as_tensor(current, dtype=torch.float64)[:, None, :]
    predicted_motion = torch.mean(torch.abs(predicted[..., active] - current_tensor[..., active]))
    target_motion = torch.mean(torch.abs(expected[..., active] - current_tensor[..., active]))

    position_errors = []
    orientation_errors = []
    for side_index, side in enumerate(("right", "left")):
        start = side_index * 8
        pred_pos, pred_rot = torch_forward_kinematics(
            artifact.bridge.target_spec, side, predicted[..., start : start + 7]
        )
        target_pos, target_rot = torch_forward_kinematics(
            artifact.bridge.target_spec, side, expected[..., start : start + 7]
        )
        position_errors.append(torch.linalg.vector_norm(pred_pos - target_pos, dim=-1))
        orientation_errors.append(_geodesic_degrees(pred_rot, target_rot))
    return {
        "normalized_joint_mae": float(normalized_mae),
        "joint_rmse_rad": float(torch.sqrt(torch.mean(error[..., active] ** 2))),
        "shoulder_mae_rad": float(shoulder_mae),
        "tcp_position_mae_m": float(torch.stack(position_errors).mean()),
        "tcp_orientation_mae_deg": float(torch.stack(orientation_errors).mean()),
        "mean_commanded_motion_rad": float(predicted_motion),
        "target_motion_rad": float(target_motion),
        "motion_ratio": float(predicted_motion / target_motion.clamp_min(1e-8)),
    }


def evaluate_adapter_cache(
    *,
    adapter: str | Path,
    cache: str | Path,
    device: str = "cpu",
    output: str | Path | None = None,
) -> dict[str, Any]:
    arrays = _cache_arrays(cache)
    artifact = load_adapter_artifact(adapter, device=device, require_weights=True)
    current = torch.as_tensor(arrays["current_state"], dtype=torch.float32, device=device)
    bridge = torch.as_tensor(arrays["bridge_chunk"], dtype=torch.float32, device=device)
    with torch.no_grad():
        corrected, _ = artifact.residual(current, bridge)
    predictions = {
        "hold": np.repeat(arrays["current_state"][:, None, :], bridge.shape[1], axis=1),
        "bridge": arrays["bridge_chunk"],
        "adapter": corrected.cpu().numpy(),
    }
    if "baseline_chunk" in arrays:
        predictions["current_5k"] = arrays["baseline_chunk"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "adapter_id": artifact.spec.adapter_id,
        "cache_sha256": sha256_file(Path(cache)),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        selected = arrays["split"] == split
        if not np.any(selected):
            continue
        report["splits"][split] = {
            name: _prediction_metrics(
                prediction[selected],
                arrays["target_chunk"][selected],
                arrays["current_state"][selected],
                artifact,
            )
            for name, prediction in predictions.items()
        }
        report["splits"][split]["samples"] = int(np.sum(selected))
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def validate_adapter_contract(
    *,
    manifest: str | Path,
    base_policy: str | Path | None = None,
    dataset_path: str | Path | None = None,
    stride: int = 500,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    artifact = load_adapter_artifact(
        manifest,
        base_policy_dir=_pretrained_dir(base_policy) if base_policy else None,
        require_weights=False,
    )
    report: dict[str, Any] = {
        "adapter_id": artifact.spec.adapter_id,
        "deployment_status": artifact.spec.deployment_status,
        "reference_model_sha256": artifact.spec.reference_model.sha256,
        "target_model_sha256": artifact.spec.target_model.sha256,
        "base_policy_verified": base_policy is not None,
    }
    if dataset_path is None:
        return report
    dataset_wrapper = EvaluationDataset(Path(dataset_path), video_backend=video_backend)
    if tuple(dataset_wrapper.joint_names) != artifact.spec.target_vector.names:
        raise EmbodimentError("dataset action names do not match target vector contract")
    states = dataset_wrapper.dataset.hf_dataset["observation.state"]
    accepted = 0
    rejected: list[dict[str, Any]] = []
    for index in range(0, len(states), stride):
        try:
            artifact.bridge.target_state_to_policy(np.asarray(states[index]))
            accepted += 1
        except Exception as exc:
            rejected.append({"index": index, "error": str(exc)})
    total = accepted + len(rejected)
    report["dataset"] = {
        "path": str(Path(dataset_path).resolve()),
        "sample_stride": stride,
        "accepted": accepted,
        "rejected": len(rejected),
        "acceptance_rate": accepted / total if total else 0.0,
        "rejections": rejected,
    }
    return report
