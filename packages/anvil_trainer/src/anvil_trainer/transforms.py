"""Dataset transforms applied at ``LeRobotDataset.__getitem__`` time.

Each ``Transform`` subclass is enabled by a field on ``TrainingConfig`` and
runs once per loaded sample.  Transforms can also optionally patch lerobot
metadata before training starts — see ``patch_metadata``.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from anvil_trainer.config import TrainingConfig

log = logging.getLogger(__name__)


# =============================================================================
# Transform ABC
# =============================================================================


class Transform(ABC):
    """
    Abstract base class for dataset transforms.

    Subclasses implement specific transformations applied to dataset items
    during training. Each transform can optionally patch LeRobot internals
    for metadata filtering.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @abstractmethod
    def is_enabled(self, config: TrainingConfig) -> bool:
        """Check if this transform should be applied."""

    @abstractmethod
    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        """
        Apply transform to a single dataset item.

        Args:
            item: Dataset item from LeRobotDataset.__getitem__
            config: Training configuration

        Returns:
            Transformed item
        """

    def patch_metadata(self, config: TrainingConfig) -> None:  # noqa: B027
        """
        Optional: Patch LeRobot metadata/utils before training.

        Override this method if the transform needs to modify how
        LeRobot builds the policy (e.g., filtering input features).
        """


# =============================================================================
# ExcludeObservationTransform
# =============================================================================


class ExcludeObservationTransform(Transform):
    """Exclude observation keys from training via --exclude-observation suffixes.

    Each suffix is prepended with "observation." to form the full dataset key:
      "images.chest"  -> "observation.images.chest"
      "velocity"      -> "observation.velocity"
    """

    @property
    def name(self) -> str:
        return "exclude_observation"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return bool(config.exclude_observation)

    @staticmethod
    def _full_keys(config: TrainingConfig) -> set[str]:
        return {f"observation.{s}" for s in config.exclude_observation}

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        for full_key in self._full_keys(config):
            item.pop(full_key, None)
        return item

    def patch_metadata(self, config: TrainingConfig) -> None:
        """Patch dataset_to_policy_features to exclude the specified observation keys."""
        import lerobot.datasets.feature_utils
        import lerobot.policies.factory
        from lerobot.datasets.feature_utils import dataset_to_policy_features

        original_func = dataset_to_policy_features
        excluded = self._full_keys(config)

        def filtered_func(features: dict) -> dict:
            filtered = {}
            for key, value in features.items():
                if key in excluded:
                    log.info("[exclude_observation] Excluding: %s", key)
                    continue
                filtered[key] = value
            return original_func(filtered)

        # Patch both the definition module and the importer (policies/factory.py)
        lerobot.datasets.feature_utils.dataset_to_policy_features = filtered_func
        lerobot.policies.factory.dataset_to_policy_features = filtered_func


# =============================================================================
# TaskOverrideTransform
# =============================================================================


class TaskOverrideTransform(Transform):
    """Override the task field for all dataset items."""

    @property
    def name(self) -> str:
        return "task_override"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.task_override is not None

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if config.task_override:
            item["task"] = config.task_override
        return item


# =============================================================================
# DeltaActionTransform
# =============================================================================


class DeltaActionTransform(Transform):
    """Convert absolute actions to delta actions (action - observation.state).

    For each joint i: delta[i] = target_position[i] - current_position[i]
    where current_position comes from observation.state (most recent step).

    Joints listed in config.delta_exclude_joints are kept in absolute space —
    useful for grippers whose targets are better expressed as absolute positions.
    Joint names are resolved by name from meta/info.json and cached.

    The configuration is persisted to anvil_config.json in each checkpoint so
    the inference node can apply the correct inverse transform automatically.
    """

    def __init__(self):
        self._exclude_indices: list[int] | None = None  # cached after first lookup
        self._shape_mismatch_warned: bool = False
        self._first_apply: bool = True  # gate for one-time logging

    @property
    def name(self) -> str:
        return "delta_actions"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.use_delta_actions

    def _resolve_exclude_indices(self, config: TrainingConfig) -> list[int]:
        """Resolve joint names → action tensor indices from dataset metadata (cached)."""
        if self._exclude_indices is not None:
            return self._exclude_indices

        if not config.delta_exclude_joints or not config.dataset_root:
            self._exclude_indices = []
            return self._exclude_indices

        info_path = Path(config.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[delta_actions] %s not found — no joints excluded", info_path)
            self._exclude_indices = []
            return self._exclude_indices

        with open(info_path) as f:
            info = json.load(f)

        names = info.get("features", {}).get("action", {}).get("names", [])
        # Flatten if nested list (e.g. [{"motor_names": [...]}])
        if names and isinstance(names[0], dict):
            names = [n for group in names for n in group.get("motor_names", [])]

        indices = []
        for joint in config.delta_exclude_joints:
            if joint in names:
                idx = names.index(joint)
                indices.append(idx)
                log.info("[delta_actions] Excluding joint '%s' (index %d) from delta", joint, idx)
            else:
                log.warning("[delta_actions] Joint '%s' not found in action names %s", joint, names)

        self._exclude_indices = indices
        return self._exclude_indices

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if "action" not in item or "observation.state" not in item:
            return item

        action = item["action"]
        state = item["observation.state"]

        # When state has multiple observation steps (e.g. [n_obs_steps, n_joints]),
        # use only the most recent step as the reference for the delta.
        if state.dim() > 1:
            state = state[-1]

        original_action = action.clone()

        action_last = action.shape[-1]
        state_last = state.shape[-1]

        if action_last == state_last:
            # Last dims match — state broadcasts to action (e.g. [n_joints] → [chunk, n_joints])
            item["action"] = action - state
        else:
            # Partial delta: apply only to the leading min(action_dim, state_dim) joints;
            # remaining joints are kept in absolute space.
            n = min(action_last, state_last)
            if not self._shape_mismatch_warned:
                log.warning(
                    "[delta_actions] action has %d joints but state has %d — "
                    "applying delta to first %d joint(s); remainder kept absolute",
                    action_last, state_last, n,
                )
                self._shape_mismatch_warned = True
            delta = original_action.clone()
            delta[..., :n] = action[..., :n] - state[..., :n]
            item["action"] = delta

        # Restore excluded joints to their original absolute values
        exclude = self._resolve_exclude_indices(config)
        for idx in exclude:
            item["action"][..., idx] = original_action[..., idx]

        if self._first_apply:
            log.info(
                "[delta_actions] active — %d joints total: %d get delta, %d kept absolute %s",
                action_last,
                action_last - len(exclude),
                len(exclude),
                config.delta_exclude_joints or [],
            )
            self._first_apply = False

        return item
