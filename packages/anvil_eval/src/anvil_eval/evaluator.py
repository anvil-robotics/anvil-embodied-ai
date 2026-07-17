"""Core evaluation logic — replay dataset episodes through a trained policy."""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
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


def _ensure_anvil_shared() -> None:
    """Add packages/anvil_shared/src to sys.path for ee_transform helpers."""
    env_path = os.environ.get("ANVIL_SHARED_PATH")
    if env_path:
        target = str(Path(env_path))
    else:
        repo_root = Path(__file__).resolve().parents[4]
        target = str(repo_root / "packages" / "anvil_shared" / "src")
    if target not in sys.path:
        sys.path.insert(0, target)


@dataclass
class EpisodeResult:
    """Raw results from evaluating a single episode."""

    episode_idx: int
    split_label: str
    predicted: np.ndarray     # (T, D) absolute actions (after ee_relative restore if needed)
    ground_truth: np.ndarray  # (T, D) absolute ground-truth actions
    joint_names: list[str]
    raw_output: np.ndarray | None = None         # (T, D) model raw output before restore
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
        _ensure_anvil_shared()
        from anvil_shared.action_types import normalize_action_type

        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_type = model_type
        self.device = device
        # normalize_action_type() maps the legacy "ee_rel" alias to the
        # canonical "ee_relative" so everything below only ever compares
        # against the canonical value.
        self.action_type: str = normalize_action_type(anvil_cfg.get("action_type", "joint_abs"))
        self.is_ee: bool = self.action_type in ("ee_abs", "ee_relative")
        self.is_ee_relative: bool = self.action_type == "ee_relative"
        self.is_ee_abs: bool = self.action_type == "ee_abs"
        self.task_description = task_description
        self.joint_names = joint_names
        self._is_vla = model_type in ("pi0", "pi05", "smolvla")
        self._relative_anchor_state: np.ndarray | None = None

    def evaluate_episode(
        self,
        dataset,
        frame_indices: list[int],
        episode_idx: int,
        split_label: str,
    ) -> EpisodeResult:
        """Evaluate model predictions for a single episode."""
        _ensure_model_loader_importable()
        _ensure_anvil_shared()
        from lerobot_control.model_loader import reset_model_state
        from anvil_shared.ee_transform import ee_obs_abs_forward, ee_obs_relative_forward, ee_relative_forward, ee_relative_inverse

        predicted_actions: list[np.ndarray] = []
        ground_truth_actions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        obs_state_list: list[np.ndarray] = []
        raw_gt_list: list[np.ndarray] = []

        reset_model_state(self.model)
        self._relative_anchor_state = None   # reset per-episode n-0 chunk anchor
        # Shadow queue of pre-restored absolute actions (avoids touching model's normalized queue)
        _abs_shadow_queue: deque[np.ndarray] = deque()

        def _tensor_to_np(a: object) -> np.ndarray:
            if self.postprocessor:
                a = self.postprocessor.process_action(a)  # type: ignore[arg-type]
            if isinstance(a, torch.Tensor):
                if a.dim() > 1:
                    a = a.squeeze(0)
                return a.detach().cpu().numpy()
            return np.asarray(a).flatten()

        for rel_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            item = dataset[rel_idx]

            # Ground truth action (always absolute from dataset)
            gt_action = item["action"].numpy()

            # Build observation dict (observation.* keys only)
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}

            # Observation state for ee_relative restore (raw, before preprocessing)
            obs_state = item["observation.state"].numpy() if "observation.state" in item else None
            _obs_flat = (obs_state[-1] if obs_state is not None and obs_state.ndim > 1 else obs_state)

            # Detect whether a new action chunk is about to be generated.
            # Only ee_relative needs restore; ee_abs/joint_abs do not.
            _needs_restore = self.is_ee_relative
            _is_new_chunk = (
                _needs_restore
                and hasattr(self.model, "_queues")
                and len(self.model._queues.get("action", [])) == 0
            )

            # Capture ref state BEFORE inference (queue will be filled after select_action)
            if _is_new_chunk and _obs_flat is not None:
                self._relative_anchor_state = _obs_flat.copy()

            # Preprocess + inference
            with torch.inference_mode():
                if self._is_vla:
                    processed = self._preprocess_vla(obs)
                else:
                    # EE obs conversion: dataset stores 8-dim quat obs; checkpoint
                    # normaliser stats are 10-dim rot6d (patched during training).
                    # Convert here before normalisation to match training layout.
                    if self.is_ee_relative and "observation.state" in obs:
                        # ee_relative: relativise to the anchor (most recent obs step).
                        obs_np = obs["observation.state"].numpy()  # (n_obs_steps, 8) or (8,)
                        anchor = obs_np[-1] if obs_np.ndim > 1 else obs_np
                        obs_rel = ee_obs_relative_forward(obs_np, anchor)
                        obs = dict(obs)
                        obs["observation.state"] = torch.tensor(
                            obs_rel, dtype=torch.float32
                        )
                    elif self.is_ee_abs and "observation.state" in obs:
                        # ee_abs: convert quat layout (8n) → rot6d layout (10n), absolute.
                        obs_np = obs["observation.state"].numpy()  # (n_obs_steps, 8) or (8,)
                        obs_abs_rot6d = ee_obs_abs_forward(obs_np)
                        obs = dict(obs)
                        obs["observation.state"] = torch.tensor(
                            obs_abs_rot6d, dtype=torch.float32
                        )
                    if self.preprocessor:
                        processed = self.preprocessor(dict(obs))
                    else:
                        processed = obs
                    processed = self._move_to_device(processed)

                action_raw = self.model.select_action(processed)  # normalized tensor

                # On new chunk: collect all remaining normalized queue items BEFORE postprocessing.
                if _is_new_chunk and _needs_restore and hasattr(self.model, "_queues"):
                    _rest_norm = [a.detach().clone() for a in self.model._queues.get("action", [])]
                else:
                    _rest_norm = None

            # Postprocess current action (normalized → denormalized)
            action = _tensor_to_np(action_raw)

            raw_actions.append(action)

            # Capture observation state for per-frame diagnostics
            if obs_state is not None:
                obs_state_list.append(_obs_flat.copy())

            # Compute raw ground truth (same space as raw model output)
            if self.is_ee_relative and _obs_flat is not None:
                raw_gt_list.append(self._compute_ee_relative_gt(gt_action, _obs_flat, ee_relative_forward))
            else:
                raw_gt_list.append(gt_action)

            # Chunk-level ee_relative restore using shadow queue.
            if _needs_restore:
                _restore_fn = lambda chunk, ref: ee_relative_inverse(chunk, ref)
                if _is_new_chunk and self._relative_anchor_state is not None:
                    if _rest_norm is not None:
                        _rest_denorm = [_tensor_to_np(a) for a in _rest_norm]
                        _chunk = np.stack([action] + _rest_denorm) if _rest_denorm else action[np.newaxis]
                    else:
                        _chunk = action[np.newaxis]
                    _abs = _restore_fn(_chunk, self._relative_anchor_state)
                    _abs_shadow_queue = deque(_abs[1:])
                    action = _abs[0]
                elif _abs_shadow_queue:
                    action = _abs_shadow_queue.popleft()
                elif not hasattr(self.model, "_queues"):
                    _ref = self._relative_anchor_state if self._relative_anchor_state is not None else _obs_flat
                    if _ref is not None:
                        _abs = _restore_fn(action[np.newaxis], _ref)
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

    def _compute_ee_relative_gt(
        self,
        gt_action: np.ndarray,
        obs_state: np.ndarray,
        ee_relative_forward_fn,
    ) -> np.ndarray:
        """Compute EE ground-truth in model-output (relative) space.

        Mirrors EERelativeTransform applied at training time.
        Uses the vectorised ``ee_relative_forward`` from anvil_shared.ee_transform.
        """
        # gt_action shape: (10*n_arms,) — single frame
        # obs_state shape: (8*n_arms,)
        return ee_relative_forward_fn(gt_action[np.newaxis, :], obs_state)[0]


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
