"""Dataset transforms applied at ``LeRobotDataset.__getitem__`` time.

Each ``Transform`` subclass is enabled by a field on ``TrainingConfig`` and
runs once per loaded sample.  Transforms can also optionally patch lerobot
metadata before training starts — see ``patch_metadata``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from anvil_trainer.config import TrainingConfig


log = logging.getLogger(__name__)


class DataIntegrityError(ValueError):
    """Raised when dataset features violate expected contracts."""


def _parse_names(info: dict, feat_key: str) -> list[str]:
    """Extract feature names from info.json for the given feature key.

    Handles both flat string lists and grouped dicts with ``motor_names``.
    """
    names = info.get("features", {}).get(feat_key, {}).get("names", [])
    if names and isinstance(names[0], dict):
        names = [n for group in names for n in group.get("motor_names", [])]
    return names


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
    """Exclude observation keys from training via --exclude-observs.

    Drop observations by suffix after "observation.":
      "images.chest"  -> "observation.images.chest"
      "velocity"      -> "observation.velocity"
    """

    @property
    def name(self) -> str:
        return "exclude_observs"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return bool(config.exclude_observs)

    @staticmethod
    def _excluded_keys(config: TrainingConfig) -> set[str]:
        if not config.exclude_observs:
            return set()
        return {f"observation.{s}" for s in config.exclude_observs}

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        excluded = self._excluded_keys(config)
        for key in list(item.keys()):
            if key in excluded:
                item.pop(key, None)
        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch dataset_to_policy_features to exclude the specified observation keys."""
        import lerobot.datasets.feature_utils
        import lerobot.policies.factory
        from lerobot.datasets.feature_utils import dataset_to_policy_features

        original_func = dataset_to_policy_features
        excluded = self._excluded_keys(config)

        def filtered_func(features: dict) -> dict:
            filtered = {}
            for key, value in features.items():
                if key in excluded:
                    log.info("[exclude_observs] Excluding: %s", key)
                    continue
                filtered[key] = value
            return original_func(filtered)

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
# EERelativeTransform — SE(3) relative EE actions
# =============================================================================


# =============================================================================
# Shared metadata-patch helper
# =============================================================================


def _patch_obs_state_shape_8n_to_10n(
    config: "TrainingConfig", runner: Any = None
) -> None:
    """Patch dataset_to_policy_features to report observation.state as 10-dim/arm.

    Shared by EEAbsTransform, EEDeltaTransform, and EERelativeTransform — all three
    convert obs.state from the dataset's on-disk ``observation_encoding`` layout
    (quaternion=8/arm, rot6d=10/arm, axis_angle=7/arm) to rot6d layout (10 dims/arm),
    so the policy must be initialised with the correct (post-transform) input
    dimension. Uses ``config.observation_encoding`` (resolved by
    ``TrainingConfig.validate_action_space()``) instead of assuming quaternion's
    8-dim/arm layout — a rot6d-encoded dataset is already 10/arm on disk, so for it
    this patch is a no-op (dimension doesn't change), unlike quaternion/axis_angle.

    When ``runner`` is provided the patch is tracked by the TransformRunner and
    will be reverted automatically by :func:`patched_lerobot`.  Without a runner
    the module attribute is set directly (test / standalone use).
    """
    import lerobot.datasets.feature_utils as _feat_utils
    import lerobot.policies.factory as _factory
    from anvil_shared.ee_encodings import observation_state_dim_per_arm
    from lerobot.datasets.feature_utils import dataset_to_policy_features as _original

    per_arm_in = observation_state_dim_per_arm(config.observation_encoding)

    def _patched(features: dict) -> dict:
        modified = {}
        for key, feat in features.items():
            if key == "observation.state":
                shape = feat.get("shape", ())
                if len(shape) == 1 and shape[0] % per_arm_in == 0:
                    modified[key] = {**feat, "shape": (shape[0] // per_arm_in * 10,)}
                else:
                    modified[key] = feat
            else:
                modified[key] = feat
        return _original(modified)

    if runner is not None:
        runner._patch(_feat_utils, "dataset_to_policy_features", _patched)
        runner._patch(_factory, "dataset_to_policy_features", _patched)
    else:
        _feat_utils.dataset_to_policy_features = _patched
        _factory.dataset_to_policy_features = _patched


# =============================================================================
# EEAbsTransform
# =============================================================================


class EEAbsTransform(Transform):
    """Convert absolute EE obs from quaternion layout (8n) to rot6d layout (10n).

    Only observation.state is converted — action is already in rot6d layout
    from the dataset (per-arm: [x, y, z, r0..r5, gripper], 10 dims).
    No SE(3) relative computation; xyz and gripper are passthrough.

    obs: 8 dims/arm (quat layout) → 10 dims/arm (rot6d layout), absolute
    action: 10 dims/arm (rot6d layout), unchanged
    """

    def __init__(self) -> None:
        self._first_apply: bool = True

    @property
    def name(self) -> str:
        return "ee_abs"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.is_ee_abs

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        import torch
        from anvil_shared.ee_encodings import observation_state_dim_per_arm
        from anvil_shared.ee_transform import ee_obs_abs_forward

        if "observation.state" not in item:
            return item

        obs_full = item["observation.state"]  # (T, state_dim_per_arm*n_arms) or (state_dim_per_arm*n_arms,)
        obs_np = obs_full.detach().cpu().numpy().astype("float64")

        obs_abs_np = ee_obs_abs_forward(
            obs_np, observation_encoding=config.observation_encoding
        )  # (..., 10*n_arms)
        item["observation.state"] = torch.tensor(obs_abs_np, dtype=torch.float32)

        if self._first_apply:
            n_arms = obs_np.shape[-1] // observation_state_dim_per_arm(config.observation_encoding)
            log.info(
                "[ee_abs] active — %d arm(s), obs (%s) → (10n rot6d, absolute)",
                n_arms, config.observation_encoding,
            )
            self._first_apply = False

        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch lerobot's dataset_to_policy_features to report 10-dim obs shape."""
        if not config.is_ee_abs:
            return
        _patch_obs_state_shape_8n_to_10n(config, runner)
        log.info("[ee_abs] patched dataset_to_policy_features: obs.state 8n→10n/arm")


# =============================================================================
# EEDeltaTransform
# =============================================================================


class EEDeltaTransform(Transform):
    """Convert absolute EE obs from quaternion layout (8n) to rot6d layout (10n)
    for the baked-delta ``ee_delta`` action_type.

    ``action`` is NOT transformed here — it is already a baked per-frame
    Delta(n->n+1) value, written to disk by mcap_converter
    (``action_encoding="delta"``) at convert time, one arm-relativization
    per frame against THIS frame's own state, targeting the NEXT frame's
    pose. Re-applying any action-side relativization here would silently
    double-transform every sample; ``action`` passes through completely
    unchanged, exactly as EEAbsTransform already does for its own action
    column.

    Structurally this mirrors EEAbsTransform's obs handling (layout
    conversion only, no relativization) — NOT EERelativeTransform's obs
    handling (which also relativizes obs against a chunk anchor). This is a
    deliberate match to LIBERO's ``native`` convention: observation.state
    stays absolute; only ``action`` carries the delta representation.

    obs: 8 dims/arm (quat layout) → 10 dims/arm (rot6d layout), absolute
    action: 10 dims/arm (rot6d layout), unchanged — already a baked delta
    """

    def __init__(self) -> None:
        self._first_apply: bool = True

    @property
    def name(self) -> str:
        return "ee_delta"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.is_ee_delta

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        import torch
        from anvil_shared.ee_encodings import observation_state_dim_per_arm
        from anvil_shared.ee_transform import ee_obs_abs_forward

        if "observation.state" not in item:
            return item

        obs_full = item["observation.state"]  # (T, state_dim_per_arm*n_arms) or (state_dim_per_arm*n_arms,)
        obs_np = obs_full.detach().cpu().numpy().astype("float64")

        obs_abs_np = ee_obs_abs_forward(
            obs_np, observation_encoding=config.observation_encoding
        )  # (..., 10*n_arms)
        item["observation.state"] = torch.tensor(obs_abs_np, dtype=torch.float32)
        # item["action"] is left untouched — already the baked Delta(n->n+1)
        # target written by mcap_converter; no double-transform here.

        if self._first_apply:
            n_arms = obs_np.shape[-1] // observation_state_dim_per_arm(config.observation_encoding)
            log.info(
                "[ee_delta] active — %d arm(s), obs (%s) → (10n rot6d, absolute); "
                "action untouched (baked per-frame Delta(n->n+1) from mcap_converter)",
                n_arms, config.observation_encoding,
            )
            self._first_apply = False

        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch lerobot's dataset_to_policy_features to report 10-dim obs shape."""
        if not config.is_ee_delta:
            return
        _patch_obs_state_shape_8n_to_10n(config, runner)
        log.info("[ee_delta] patched dataset_to_policy_features: obs.state 8n→10n/arm")


# =============================================================================
# EERelativeTransform
# =============================================================================


class EERelativeTransform(Transform):
    """Convert absolute EE obs and actions to SE(3)-relative representation.

    Both observation.state and action are anchored to the SAME current EE pose
    (last obs step), matching UMI's verified 'relative' mode:
        T_rel = inv(T_anchor) @ T_pose  (full SE(3), translation in body frame)

    obs: 8 dims/arm (quat layout) → 10 dims/arm (rot6d layout), relative to anchor
    action: 10 dims/arm (rot6d layout), unchanged dim, relative to anchor
    """

    def __init__(self):
        self._first_apply: bool = True

    @property
    def name(self) -> str:
        return "ee_relative"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return config.is_ee_relative

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        import torch
        from anvil_shared.ee_transform import ee_obs_relative_forward, ee_relative_forward, n_arms_from_dims

        if "action" not in item or "observation.state" not in item:
            return item

        obs_encoding = config.observation_encoding
        action = item["action"]                   # (horizon, 10*n_arms) or (10*n_arms,)
        obs_full = item["observation.state"]       # (T, state_dim_per_arm*n_arms) or (state_dim_per_arm*n_arms,)

        # Anchor = most recent obs step (state_dim_per_arm*n_arms,)
        if obs_full.dim() > 1:
            anchor_tensor = obs_full[-1]
        else:
            anchor_tensor = obs_full

        anchor_np = anchor_tensor.detach().cpu().numpy().astype("float64")
        obs_np = obs_full.detach().cpu().numpy().astype("float64")
        action_np = action.detach().cpu().numpy().astype("float64")

        # Validate action/state dims
        try:
            n_arms = n_arms_from_dims(anchor_np.shape[-1], action_np.shape[-1], obs_encoding)
        except ValueError as exc:
            raise DataIntegrityError(str(exc)) from exc

        # Transform obs: (T, state_dim_per_arm*n) → (T, 10*n) relative to anchor
        obs_rel_np = ee_obs_relative_forward(obs_np, anchor_np, observation_encoding=obs_encoding)

        # Transform action: (horizon, 10*n) relative to anchor
        single = action_np.ndim == 1
        if single:
            action_np = action_np[None, :]
        delta_np = ee_relative_forward(action_np, anchor_np, observation_encoding=obs_encoding)
        if single:
            delta_np = delta_np[0]

        item["observation.state"] = torch.tensor(obs_rel_np, dtype=torch.float32)
        item["action"] = torch.tensor(delta_np, dtype=action.dtype)

        if self._first_apply:
            log.info(
                "[ee_relative] active — %d arm(s), obs (%s) → (10n rel), action (abs rot6d) → SE(3) relative",
                n_arms, obs_encoding,
            )
            self._first_apply = False

        return item

    def patch_metadata(self, config: TrainingConfig, runner: Any = None) -> None:
        """Patch lerobot's dataset_to_policy_features to report 10-dim obs shape.

        observation.state changes from 8*n_arms (quat layout) to 10*n_arms (rot6d
        relative layout) after this transform. The policy must be initialised with
        the correct input dimension.
        """
        if not config.is_ee_relative:
            return
        _patch_obs_state_shape_8n_to_10n(config, runner)
        log.info("[ee_relative] patched dataset_to_policy_features: obs.state 8n→10n/arm")
