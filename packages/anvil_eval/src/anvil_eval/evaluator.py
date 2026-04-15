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
    predicted: np.ndarray     # (T, D)
    ground_truth: np.ndarray  # (T, D)
    joint_names: list[str]


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
        self.use_delta_actions = anvil_cfg.get("use_delta_actions", False)
        self.delta_exclude_joints = anvil_cfg.get("delta_exclude_joints", [])
        self.task_description = task_description
        self.joint_names = joint_names
        self._is_vla = model_type in ("pi0", "pi05", "smolvla")
        self._exclude_indices: set[int] | None = None

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

        # Reset model state (clears action queues for ACT)
        reset_model_state(self.model)

        for rel_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            item = dataset[rel_idx]

            # Ground truth action (always absolute from dataset)
            gt_action = item["action"].numpy()

            # Build observation dict (observation.* keys only)
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}

            # Observation state for delta restore (raw, before preprocessing)
            obs_state = item["observation.state"].numpy() if "observation.state" in item else None

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

            # Delta action restore
            if self.use_delta_actions and obs_state is not None:
                action = self._restore_delta_action(action, obs_state)

            predicted_actions.append(action)
            ground_truth_actions.append(gt_action)

        return EpisodeResult(
            episode_idx=episode_idx,
            split_label=split_label,
            predicted=np.stack(predicted_actions),
            ground_truth=np.stack(ground_truth_actions),
            joint_names=self.joint_names,
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
