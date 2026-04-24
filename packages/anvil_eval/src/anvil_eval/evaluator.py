"""Core evaluation logic — replay dataset episodes through a trained policy."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def _ensure_model_loader_importable() -> None:
    """Add lerobot_control to sys.path for ModelLoader import (zero ROS2 deps)."""
    env_path = os.environ.get("LEROBOT_CONTROL_PATH")
    if env_path:
        target = str(Path(env_path))
    else:
        # Repo-relative: packages/anvil_eval/src/anvil_eval/evaluator.py -> repo root
        repo_root = Path(__file__).resolve().parents[4]
        target = str(repo_root / "ros2" / "src" / "lerobot_control")

    if target not in sys.path:
        sys.path.insert(0, target)


@dataclass
class EpisodeResult:
    """Raw results from evaluating a single episode."""

    episode_idx: int
    split_label: str
    predicted: np.ndarray     # (T, D) absolute actions (after delta restore)
    ground_truth: np.ndarray  # (T, D)
    joint_names: list[str]
    raw_output: np.ndarray | None = None  # (T, D) model raw output before delta restore


class EpisodeEvaluator:
    """Evaluate a trained policy by replaying dataset episodes."""

    def __init__(
        self,
        model,
        preprocessor,
        postprocessor,
        model_type: str,
        device: str,
        anvil_cfg: dict,
        task_description: str | None,
        joint_names: list[str],
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_type = model_type
        self.device = device
        self.action_type: str = anvil_cfg.get("action_type", "absolute")
        if self.action_type == "absolute" and anvil_cfg.get("use_delta_actions", False):
            self.action_type = "delta_obs_t"
        self.use_delta_actions: bool = self.action_type in ("delta_obs_t", "delta_sequential")
        self.delta_exclude_joints = anvil_cfg.get("delta_exclude_joints", [])
        self.task_description = task_description
        self.joint_names = joint_names
        self._is_vla = model_type in ("pi0", "pi05", "smolvla")
        self._exclude_indices: set[int] | None = None
        self._delta_ref_state: np.ndarray | None = None
        self._prev_gt_action: np.ndarray | None = None  # for delta_sequential GT computation

    def evaluate_episode(
        self,
        dataset,
        frame_indices: list[int],
        episode_idx: int,
        split_label: str,
    ) -> EpisodeResult:
        """Evaluate model predictions for a single episode."""
        _ensure_model_loader_importable()
        from lerobot_control.model_loader import reset_model_state
        from lerobot_control.delta_restore import restore_delta_chunk

        predicted_actions: list[np.ndarray] = []
        ground_truth_actions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []

        # Reset model state (clears action queues for ACT)
        reset_model_state(self.model)
        self._delta_ref_state = None   # reset per-episode delta reference state
        self._prev_gt_action = None    # reset sequential GT reference

        exclude_indices = list(self._resolve_exclude_indices())

        for rel_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            item = dataset[rel_idx]

            # Ground truth action (always absolute from dataset)
            gt_action = item["action"].numpy()

            # Build observation dict (observation.* keys only)
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}

            # Observation state for delta restore (raw, before preprocessing)
            obs_state = item["observation.state"].numpy() if "observation.state" in item else None
            _obs_flat = (obs_state[-1] if obs_state is not None and obs_state.ndim > 1 else obs_state)

            # Detect whether a new action chunk is about to be generated.
            # When queue is empty, select_action runs model and fills queue.
            # We capture obs_state as the delta reference for the whole chunk.
            _is_new_chunk = (
                self.use_delta_actions
                and hasattr(self.model, "_queues")
                and len(self.model._queues.get("action", [])) == 0
            )

            # Capture ref state BEFORE inference (queue will be filled after select_action)
            if _is_new_chunk and _obs_flat is not None:
                self._delta_ref_state = _obs_flat.copy()

            # Preprocess + inference
            with torch.inference_mode():
                if self._is_vla:
                    processed = self._preprocess_vla(obs)
                else:
                    if self.preprocessor:
                        processed = self.preprocessor(dict(obs))
                    else:
                        processed = obs
                    processed = self._move_to_device(processed)

                action = self.model.select_action(processed)

            # Postprocess
            if self.postprocessor:
                action = self.postprocessor.process_action(action)

            # To numpy
            if isinstance(action, torch.Tensor):
                if action.dim() > 1:
                    action = action.squeeze(0)
                action = action.cpu().numpy()

            # Capture raw model output before delta restore
            raw_actions.append(action.copy())

            # Chunk-level delta restore: on new chunk, restore all queued actions at once
            # so sequential and obs_t modes both work correctly across the chunk.
            if _is_new_chunk and self.use_delta_actions and self._delta_ref_state is not None:
                if hasattr(self.model, "_queues"):
                    _rest = [
                        (a.cpu().numpy().flatten() if hasattr(a, "cpu") else np.asarray(a).flatten())
                        for a in list(self.model._queues.get("action", []))
                    ]
                    _chunk = np.stack([action] + _rest) if _rest else action[np.newaxis]
                    _abs = restore_delta_chunk(
                        _chunk, self._delta_ref_state, self.action_type, exclude_indices
                    )
                    from collections import deque as _deque
                    self.model._queues["action"] = _deque(
                        torch.tensor(_a, dtype=torch.float32) for _a in _abs[1:]
                    )
                    action = _abs[0]
                else:
                    # Fallback for models without _queues (single-step restore)
                    _abs = restore_delta_chunk(
                        action[np.newaxis], self._delta_ref_state, self.action_type, exclude_indices
                    )
                    action = _abs[0]
            elif not _is_new_chunk and self.use_delta_actions and not hasattr(self.model, "_queues"):
                # Models without chunk queue (e.g. ACT w/ temporal ensemble): per-step restore
                _ref = self._delta_ref_state if self._delta_ref_state is not None else _obs_flat
                if _ref is not None:
                    _abs = restore_delta_chunk(
                        action[np.newaxis], _ref, self.action_type, exclude_indices
                    )
                    action = _abs[0]

            predicted_actions.append(action)
            ground_truth_actions.append(gt_action)

        return EpisodeResult(
            episode_idx=episode_idx,
            split_label=split_label,
            predicted=np.stack(predicted_actions),
            ground_truth=np.stack(ground_truth_actions),
            joint_names=self.joint_names,
            raw_output=np.stack(raw_actions) if raw_actions else None,
        )

    def _preprocess_vla(self, obs: dict) -> dict:
        """Preprocess observation for VLA models (pi0, pi05, smolvla)."""
        if self.preprocessor:
            batch = dict(obs)
            if self.task_description:
                batch["task"] = [self.task_description]
            processed = self.preprocessor(batch)
            return self._move_to_device(processed)
        return self._move_to_device(obs)

    def _move_to_device(self, data):
        """Recursively move tensors to the configured device."""
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _resolve_exclude_indices(self) -> set[int]:
        """Resolve delta_exclude_joints to index set (cached)."""
        if self._exclude_indices is not None:
            return self._exclude_indices

        self._exclude_indices = set()
        for name in self.delta_exclude_joints:
            if name in self.joint_names:
                self._exclude_indices.add(self.joint_names.index(name))
        return self._exclude_indices



def load_model(checkpoint: str, device: str):
    """Load model + processors from checkpoint using ModelLoader.

    Returns (model, preprocessor, postprocessor, model_type).
    """
    _ensure_model_loader_importable()
    from lerobot_control.model_loader import ModelLoader

    loader = ModelLoader(
        model_path=checkpoint,
        device=device,
        logger=None,
        deterministic=True,
        seed=42,
    )
    model, preprocessor, postprocessor = loader.load_with_processors()

    # Detect model type
    model_type = getattr(loader, "model_type", "unknown")

    log.info("[anvil-eval] Loaded model: type=%s, device=%s", model_type, device)
    return model, preprocessor, postprocessor, model_type
