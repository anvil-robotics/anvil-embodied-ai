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
    ground_truth: np.ndarray  # (T, D) absolute ground-truth actions
    joint_names: list[str]
    raw_output: np.ndarray | None = None         # (T, D) model raw output before delta restore
    obs_states: np.ndarray | None = None         # (T, D) observation state at each frame
    raw_ground_truth: np.ndarray | None = None   # (T, D) GT in same space as raw_output


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
        _action_type = anvil_cfg.get("action_type", "absolute")
        if _action_type == "absolute" and anvil_cfg.get("use_delta_actions", False):
            _action_type = "delta_obs_t"
        self.action_type = _action_type
        self.use_delta_actions = _action_type in ("delta_obs_t", "delta_sequential")
        self.delta_exclude_joints = anvil_cfg.get("delta_exclude_joints", [])
        self.task_description = task_description
        self.joint_names = joint_names
        self._is_vla = model_type in ("pi0", "pi05", "smolvla")
        self._exclude_indices: set[int] | None = None
        self._delta_ref_state: np.ndarray | None = None

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

        predicted_actions: list[np.ndarray] = []
        ground_truth_actions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        obs_state_list: list[np.ndarray] = []
        raw_gt_list: list[np.ndarray] = []
        _prev_gt: np.ndarray | None = None  # for delta_sequential GT reference

        # Reset model state (clears action queues for ACT)
        reset_model_state(self.model)
        self._delta_ref_state = None  # reset per-episode delta reference state

        for rel_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            item = dataset[rel_idx]

            # Ground truth action (always absolute from dataset)
            gt_action = item["action"].numpy()

            # Build observation dict (observation.* keys only)
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}

            # Observation state for delta restore (raw, before preprocessing)
            obs_state = item["observation.state"].numpy() if "observation.state" in item else None

            # Detect whether a new action chunk is about to be generated.
            # DiffusionPolicy caches n_action_steps actions relative to the state at
            # the time the chunk was generated (state_t0).  Restoring queued steps
            # with the current obs_state (state_t0+k) introduces a drift error of
            # state_{t0+k} − state_t0.  We therefore capture obs_state_t0 once per
            # chunk and reuse it for all subsequent restorations in that chunk.
            # Models without _queues (e.g. ACT) fall back to per-frame obs_state.
            _is_new_chunk = (
                self.use_delta_actions
                and hasattr(self.model, "_queues")
                and len(self.model._queues.get("action", [])) == 0
            )

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

            # Capture reference state right after new chunk is generated
            if _is_new_chunk and obs_state is not None:
                _ref = obs_state[-1] if obs_state.ndim > 1 else obs_state
                self._delta_ref_state = _ref.copy()

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

            # Capture observation state for per-frame diagnostics
            _obs_flat: np.ndarray | None = None
            if obs_state is not None:
                _obs_flat = obs_state[-1] if obs_state.ndim > 1 else obs_state
                obs_state_list.append(_obs_flat.copy())

            # Compute raw ground truth (same space as raw model output)
            if self.use_delta_actions and _obs_flat is not None:
                raw_gt_list.append(self._compute_delta_gt(gt_action, _obs_flat, _prev_gt))
            else:
                raw_gt_list.append(gt_action.copy())
            _prev_gt = gt_action.copy()

            # Delta action restore — use the reference state from when the current
            # chunk was generated (state_t0) rather than the current obs_state, so
            # that all 8 queued steps share the same reference.
            if self.use_delta_actions:
                _obs_ref = (obs_state[-1] if obs_state is not None and obs_state.ndim > 1
                            else obs_state)
                _ref = self._delta_ref_state if self._delta_ref_state is not None else _obs_ref
                if _ref is not None:
                    action = self._restore_delta_action(action, _ref)

            predicted_actions.append(action)
            ground_truth_actions.append(gt_action)

        return EpisodeResult(
            episode_idx=episode_idx,
            split_label=split_label,
            predicted=np.stack(predicted_actions),
            ground_truth=np.stack(ground_truth_actions),
            joint_names=self.joint_names,
            raw_output=np.stack(raw_actions) if raw_actions else None,
            obs_states=np.stack(obs_state_list) if obs_state_list else None,
            raw_ground_truth=np.stack(raw_gt_list) if raw_gt_list else None,
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

    def _compute_delta_gt(
        self,
        gt_action: np.ndarray,
        obs_state: np.ndarray,
        prev_gt: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ground-truth in model-output (delta) space.

        delta_obs_t:     raw_gt[i] = gt_action[i] - obs_state[i]
        delta_sequential: raw_gt[i] = gt_action[i] - prev_gt[i]  (falls back to obs_state for t=0)
        Joints in delta_exclude_joints remain as absolute values (matching training).
        """
        delta_gt = gt_action.copy()
        exclude = self._resolve_exclude_indices()
        if self.action_type == "delta_sequential" and prev_gt is not None:
            ref = prev_gt
        else:
            ref = obs_state
        for i in range(min(len(delta_gt), len(ref))):
            if i not in exclude:
                delta_gt[i] = gt_action[i] - ref[i]
        return delta_gt

    def _restore_delta_action(self, predicted_delta: np.ndarray, obs_state: np.ndarray) -> np.ndarray:
        """Restore absolute action from delta prediction.

        delta = action - observation.state (during training)
        absolute = delta + observation.state (restore)
        Joints in delta_exclude_joints stay as-is (already absolute).
        """
        predicted_abs = predicted_delta.copy()
        exclude = self._resolve_exclude_indices()

        for i in range(min(len(predicted_abs), len(obs_state))):
            if i not in exclude:
                predicted_abs[i] = predicted_delta[i] + obs_state[i]

        return predicted_abs


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
