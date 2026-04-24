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


class DataIntegrityError(ValueError):
    """Raised when dataset features violate expected contracts (e.g. action joint missing from obs state)."""


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

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:  # noqa: B027
        """
        Optional: Patch LeRobot metadata/utils before training.

        Override this method if the transform needs to modify how
        LeRobot builds the policy (e.g., filtering input features).

        ``runner`` (when provided) is the owning ``TransformRunner``; use its
        ``_patch(module, attr, new_value)`` method so patches are reverted
        when the :func:`patched_lerobot` context manager exits.
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

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
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

        # Patch both the definition module and the importer (policies/factory.py).
        # Use runner._patch so patches are reverted by patched_lerobot(); fall
        # back to direct assignment for backward compatibility when a transform
        # is used standalone without a runner.
        if runner is not None:
            runner._patch(lerobot.datasets.feature_utils, "dataset_to_policy_features", filtered_func)
            runner._patch(lerobot.policies.factory, "dataset_to_policy_features", filtered_func)
        else:
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

    Joint names are resolved from meta/info.json when dataset_root is set.
    Every action joint (not in delta_exclude_joints) must have a matching name
    in observation.state — if not, a DataIntegrityError is raised at first use.
    When info.json is unavailable, shapes must match exactly.

    The configuration is persisted to anvil_config.json in each checkpoint so
    the inference node can apply the correct inverse transform automatically.
    """

    def __init__(self):
        self._mappings_built: bool = False
        self._exclude_indices: list[int] = []
        # action_idx → state_idx; None means fall back to positional (no info.json)
        self._action_to_state_map: list[int] | None = None
        self._first_apply: bool = True

    @property
    def name(self) -> str:
        return "delta_actions"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.action_type in ("delta_obs_t", "delta_sequential")

    @staticmethod
    def _parse_names(info: dict, feat_key: str) -> list[str]:
        names = info.get("features", {}).get(feat_key, {}).get("names", [])
        if names and isinstance(names[0], dict):
            names = [n for group in names for n in group.get("motor_names", [])]
        return names

    def _build_mappings(self, config: TrainingConfig) -> None:
        """Load info.json once, validate action↔state names, build index maps."""
        if self._mappings_built:
            return
        self._mappings_built = True

        if not config.dataset_root:
            return  # no info.json — positional fallback; shape validated in apply()

        info_path = Path(config.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[delta_actions] %s not found — using positional mapping", info_path)
            return

        with open(info_path) as f:
            info = json.load(f)

        action_names = self._parse_names(info, "action")
        state_names = self._parse_names(info, "observation.state")

        # Resolve exclude indices from action names
        for joint in (config.delta_exclude_joints or []):
            if joint in action_names:
                idx = action_names.index(joint)
                self._exclude_indices.append(idx)
                log.info("[delta_actions] Excluding joint '%s' (index %d) from delta", joint, idx)
            else:
                log.warning("[delta_actions] Joint '%s' not found in action names %s", joint, action_names)

        if not action_names or not state_names:
            return  # names missing from info.json — positional fallback

        # Validate: every non-excluded action joint must exist in observation.state
        exclude_names = set(config.delta_exclude_joints or [])
        missing = [n for n in action_names if n not in exclude_names and n not in state_names]
        if missing:
            raise DataIntegrityError(
                "[delta_actions] Data integrity error: the following action joints have no "
                f"matching entry in observation.state:\n"
                f"  Missing:                 {missing}\n"
                f"  action names:            {action_names}\n"
                f"  observation.state names: {state_names}\n"
                f"  delta_exclude_joints:    {sorted(exclude_names)}\n"
                "Fix: add the joints to --delta-exclude-joints or correct the dataset."
            )

        # Build action_idx → state_idx mapping (excluded joints map to -1)
        state_index = {n: i for i, n in enumerate(state_names)}
        self._action_to_state_map = [
            state_index.get(n, -1) for n in action_names
        ]

    def _resolve_exclude_indices(self, config: TrainingConfig) -> list[int]:
        """Return cached exclude indices (used by TransformRunner stats computation)."""
        self._build_mappings(config)
        return self._exclude_indices

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        if "action" not in item or "observation.state" not in item:
            return item

        self._build_mappings(config)

        action = item["action"]
        state = item["observation.state"]

        # When state has multiple observation steps (e.g. [n_obs_steps, n_joints]),
        # use only the most recent step as the reference for the delta.
        if state.dim() > 1:
            state = state[-1]

        original_action = action.clone()
        action_last = action.shape[-1]
        state_last = state.shape[-1]

        is_sequential = getattr(config, "delta_sequential", False)

        if self._action_to_state_map is not None:
            # Name-based mapping: each action joint → its counterpart in state by name
            delta = original_action.clone()
            exclude_set = set(self._exclude_indices)

            if is_sequential and action.dim() == 2:
                # delta_sequential:
                #   k=0: action[0] - state
                #   k>0: action[k] - action[k-1]
                for a_idx, s_idx in enumerate(self._action_to_state_map):
                    if a_idx not in exclude_set:
                        delta[0, a_idx] = action[0, a_idx] - state[s_idx]
                for k in range(1, action.shape[0]):
                    for a_idx in range(action.shape[1]):
                        if a_idx not in exclude_set:
                            delta[k, a_idx] = action[k, a_idx] - action[k - 1, a_idx]
            else:
                # delta_obs_t: all k relative to obs (existing broadcast logic)
                for a_idx, s_idx in enumerate(self._action_to_state_map):
                    if a_idx not in exclude_set:
                        delta[..., a_idx] = action[..., a_idx] - state[..., s_idx]

            item["action"] = delta
        elif action_last == state_last:
            # Positional fallback (info.json unavailable) — shapes match, safe to subtract
            if is_sequential and action.dim() == 2:
                delta = original_action.clone()
                exclude_set = set(self._exclude_indices)
                for a_idx in range(action.shape[1]):
                    if a_idx not in exclude_set:
                        delta[0, a_idx] = action[0, a_idx] - state[a_idx]
                for k in range(1, action.shape[0]):
                    for a_idx in range(action.shape[1]):
                        if a_idx not in exclude_set:
                            delta[k, a_idx] = action[k, a_idx] - action[k - 1, a_idx]
                item["action"] = delta
            else:
                item["action"] = action - state
                for idx in self._exclude_indices:
                    item["action"][..., idx] = original_action[..., idx]
        else:
            raise DataIntegrityError(
                f"[delta_actions] action has {action_last} joints but observation.state has "
                f"{state_last} joints and no info.json is available for name-based mapping. "
                "Provide dataset_root so joint names can be resolved, or fix the dataset."
            )

        if self._first_apply:
            mode = "sequential" if is_sequential else "obs_t"
            log.info(
                "[delta_actions] active (mode=%s) — %d joints total: %d get delta, %d kept absolute %s",
                mode,
                action_last,
                action_last - len(self._exclude_indices),
                len(self._exclude_indices),
                config.delta_exclude_joints or [],
            )
            self._first_apply = False

        return item
