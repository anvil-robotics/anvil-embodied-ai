"""Monkey-patches applied to lerobot at training time.

``TransformRunner`` owns:
    * The active list of :class:`~anvil_trainer.transforms.Transform` instances.
    * Six monkey-patches on lerobot modules:
        - ``apply_dataset_patches`` — patches ``LeRobotDataset.__getitem__``.
        - ``apply_val_loss_patch`` — patches ``make_dataset`` (split creation),
          captures the preprocessor from ``make_pre_post_processors``, and
          injects EE rel action stats into the returned ``train_dataset``.
        - ``apply_checkpoint_patch`` — patches ``save_checkpoint`` to compute
          test loss (raw + EMA) and write EMA weights / ``anvil_config.json`` /
          ``split_info.json`` / ``model_raw.safetensors`` next to each checkpoint.
        - ``apply_val_loss_hook`` — patches ``update_policy`` for periodic val
          loss computation.
        - ``apply_ddpm_ip_patch`` — patches ``DiffusionModel.compute_loss`` to add
          DDPM-IP input perturbation (UMI alpha=0.1, disable with ``--no-ddpm-ip``).
        - ``apply_ema_hook`` — patches ``update_policy`` to maintain an EMA of
          policy weights (UMI-style, enabled by default, disable with ``--no-ema``).
        - ``apply_metadata_patches`` — runs ``Transform.patch_metadata`` hooks
          (currently used by ``ExcludeObservationTransform``).

Patches are installed via :meth:`TransformRunner._patch` which tracks the
original attribute so :meth:`restore_all_patches` can put everything back.
For the typical ``train()`` entry point, use the
:func:`patched_lerobot` context manager which guarantees cleanup even when
training raises.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from anvil_shared.provenance import git_provenance
from anvil_shared.splits import compute_split_episodes, load_split_info, save_split_info

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import (
    DataIntegrityError,
    EEAbsTransform,
    EERelTransform,
    EERelWorldTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

log = logging.getLogger(__name__)

# Sentinel used to mark "patch already installed" in the originals list so we
# can keep insertion order + detect re-entrancy without wrapping in tuples.
_PATCHED_MARKER = object()


# ---------------------------------------------------------------------------
# Module-level helpers shared by multiple patches
# ---------------------------------------------------------------------------

def _force_rot6d_identity(
    min_arr: "np.ndarray", max_arr: "np.ndarray", n_arms: int, dim_per_arm: int = 10
) -> None:
    """Force rot6d dims (index 3–8 per arm) to min=-1, max=1 (identity trick).

    Under MIN_MAX normalization, ``2*(x-(-1))/(1-(-1))-1 = x``, so rot6d passes
    through unchanged — matching UMI's rot6d→identity design.  Position dims
    (0–2) and gripper (9) are left untouched so they get proper range
    normalization from their true data distributions.

    Modifies *min_arr* and *max_arr* in-place.
    """
    for arm in range(n_arms):
        for r in range(3, 9):
            idx = arm * dim_per_arm + r
            min_arr[idx] = -1.0
            max_arr[idx] = 1.0


def _log_wandb(metrics: dict, step: int) -> None:
    """Log *metrics* to the active wandb run at *step*, silently no-op if unavailable."""
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.log(metrics, step=step)
    except Exception:
        pass


def _compute_mean_loss(policy, dataloader, preprocessor) -> float | None:
    """Run *policy* over *dataloader* and return mean loss.

    Handles:
    - Moving tensors to the policy device when no *preprocessor* is provided.
    - Temporarily switching ACTPolicy to train mode (needed to activate the VAE
      for loss computation) and restoring eval mode afterwards.
    - Returns ``None`` if *dataloader* is ``None`` or *policy* is ``None``.
    """
    import torch

    if dataloader is None or policy is None:
        return None

    is_act = "ACTPolicy" in str(type(policy))
    policy.eval()
    if is_act:
        policy.train()

    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            if preprocessor is not None:
                batch = preprocessor(batch)
            else:
                device = next(policy.parameters()).device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            loss, _ = policy.forward(batch)
            total += loss.item()
            n += 1

    if is_act:
        policy.eval()
    return total / max(n, 1)


class TransformRunner:
    """
    Manages and applies dataset transforms.

    Handles:
    - Registration of transforms
    - Metadata patching (before lerobot import)
    - Dataset patching (after lerobot import)
    """

    # Registry of available transforms (add new transforms here).
    # Instantiated fresh per TransformRunner so stateful transforms do not share state across runs.
    TRANSFORMS: list[Transform] = []  # populated in __init__

    def __init__(self, config: TrainingConfig):
        self.config = config
        transforms: list[Transform] = [
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            EEAbsTransform(),
            EERelTransform(),
            EERelWorldTransform(),
        ]
        self.active_transforms = [t for t in transforms if t.is_enabled(config)]
        self._val_dataloader = None   # set by apply_val_loss_patch when make_dataset is called
        self._test_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._split_info: dict = {}   # populated by patched_make_dataset
        self._preprocessor = None     # captured from make_pre_post_processors
        self._val_freq = 0            # set from cfg.log_freq * 5 inside patched_make_dataset
        self._resume_step = 0         # for absolute step tracking in wandb
        # List of (module, attr_name, original_value) — populated by _patch in
        # insertion order so restore_all_patches can revert in reverse.
        self._saved_originals: list[tuple[Any, str, Any]] = []
        # Registry alias names added by apply_processor_compat_aliases — unregistered on restore.
        self._registered_aliases: list[str] = []

    # -------------------------------------------------------------------------
    # Patch install / restore infrastructure
    # -------------------------------------------------------------------------

    def _patch(self, module: Any, attr_name: str, new_value: Any) -> None:
        """Install a monkey-patch and remember the original for later restoration.

        Calling this twice with the same ``(module, attr_name)`` from the same
        TransformRunner instance is a no-op (with a debug log) — it will not
        re-wrap and lose the true original.  Use :meth:`restore_all_patches`
        to put things back; the :func:`patched_lerobot` context manager does
        this automatically.
        """
        already_patched = any(
            m is module and n == attr_name for m, n, _ in self._saved_originals
        )
        if already_patched:
            log.debug(
                "[anvil_trainer] Skipping duplicate patch %s.%s",
                getattr(module, "__name__", module), attr_name,
            )
            return
        original = getattr(module, attr_name)
        self._saved_originals.append((module, attr_name, original))
        setattr(module, attr_name, new_value)

    def restore_all_patches(self) -> None:
        """Restore every attribute touched by :meth:`_patch` (LIFO).

        Called by :func:`patched_lerobot` on context exit.  Safe to call more
        than once — the originals list is cleared after restoration.
        Also unregisters any processor aliases registered by
        :meth:`apply_processor_compat_aliases`.
        """
        while self._saved_originals:
            module, attr_name, original = self._saved_originals.pop()
            try:
                setattr(module, attr_name, original)
            except Exception as e:  # pragma: no cover — extremely defensive
                log.warning(
                    "[anvil_trainer] Failed to restore %s.%s: %s",
                    getattr(module, "__name__", module), attr_name, e,
                )
        # Unregister processor registry aliases added by apply_processor_compat_aliases.
        if self._registered_aliases:
            try:
                from lerobot.processor.pipeline import ProcessorStepRegistry

                for alias in self._registered_aliases:
                    ProcessorStepRegistry.unregister(alias)
                log.debug(
                    "[anvil_trainer] Unregistered processor aliases: %s",
                    self._registered_aliases,
                )
            except Exception as e:  # pragma: no cover
                log.warning("[anvil_trainer] Failed to unregister processor aliases: %s", e)
            finally:
                self._registered_aliases.clear()

    def apply_processor_compat_aliases(self) -> None:
        """Register backward-compatibility aliases for renamed ProcessorStep registry names.

        Some lerobot Hub checkpoints (e.g. ``lerobot/pi05_base``) were published with
        an older registry name ``relative_actions_processor`` that was later renamed to
        ``delta_actions_processor`` in lerobot 0.5.x.  Loading those checkpoints fails
        with an ImportError because the old name is no longer in the registry.

        This method registers the old name as an alias pointing to the same class
        (``RelativeActionsProcessorStep``) so that deserialization succeeds.  Crucially,
        it restores the class's ``_registry_name`` attribute to the *canonical* new name
        after registration, so that any checkpoints saved during this training run still
        use ``delta_actions_processor`` rather than the legacy name.

        The alias is unregistered automatically in :meth:`restore_all_patches`.
        """
        try:
            from lerobot.processor.pipeline import ProcessorStepRegistry
            from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
        except ImportError as e:
            log.warning(
                "[anvil_trainer] Could not import lerobot processor for compat aliases: %s", e
            )
            return

        if "relative_actions_processor" in ProcessorStepRegistry.list():
            return  # Already registered — no-op.

        canonical_name = getattr(RelativeActionsProcessorStep, "_registry_name", "delta_actions_processor")
        try:
            ProcessorStepRegistry.register("relative_actions_processor")(RelativeActionsProcessorStep)
            # register() overwrites _registry_name on the class; restore it so that checkpoints
            # produced during this training run serialize with the current canonical name.
            RelativeActionsProcessorStep._registry_name = canonical_name
            self._registered_aliases.append("relative_actions_processor")
            log.info(
                "[anvil_trainer] Registered compat alias 'relative_actions_processor' → %s",
                canonical_name,
            )
        except Exception as e:  # pragma: no cover
            log.warning("[anvil_trainer] Failed to register processor compat alias: %s", e)

    def log_config(self) -> None:
        """Log active transforms."""
        if not self.active_transforms:
            log.info("[anvil_trainer] Active transforms: (none - pass-through mode)")
            return

        for transform in self.active_transforms:
            details = self._get_transform_details(transform)
            log.info("[anvil_trainer] Active transform: %s — %s", transform.name, details)

    def _get_transform_details(self, transform: Transform) -> str:
        """Get human-readable details for a transform."""
        if isinstance(transform, ExcludeObservationTransform):
            if self.config.exclude_observs:
                return f"excluding: {', '.join(self.config.exclude_observs)}"
            return "enabled"
        elif isinstance(transform, TaskOverrideTransform):
            return f"'{self.config.task_override}'"
        elif isinstance(transform, EERelTransform):
            return "SE(3) relative: delta_xyz + R_state.T @ R_action"
        elif isinstance(transform, EERelWorldTransform):
            return "SE(3) relative (WORLD frame, experimental): delta_xyz + R_action @ R_state.T"
        return "enabled"

    def _compute_ee_rel_stats(self, full_dataset: Any, cfg: Any) -> dict | None:
        """Compute EE relative action AND obs stats for ``ee_rel`` training (UMI body frame)."""
        if not self.config.is_ee_rel:
            return None
        from anvil_shared.ee_transform import ee_obs_rel_forward, ee_rel_forward

        return self._compute_ee_rel_stats_generic(
            full_dataset, cfg, action_forward=ee_rel_forward, obs_forward=ee_obs_rel_forward,
            log_tag="ee_rel_stats",
        )

    def _compute_ee_rel_world_stats(self, full_dataset: Any, cfg: Any) -> dict | None:
        """Compute EE relative action AND obs stats for the experimental
        ``ee_rel_world`` training (WORLD frame — see :func:`ee_rel_world_forward`)."""
        if not self.config.is_ee_rel_world:
            return None
        from anvil_shared.ee_transform import ee_obs_rel_world_forward, ee_rel_world_forward

        return self._compute_ee_rel_stats_generic(
            full_dataset, cfg, action_forward=ee_rel_world_forward, obs_forward=ee_obs_rel_world_forward,
            log_tag="ee_rel_world_stats",
        )

    def _compute_ee_rel_stats_generic(
        self, full_dataset: Any, cfg: Any, *, action_forward: Any, obs_forward: Any, log_tag: str,
    ) -> dict | None:
        """Shared implementation behind :meth:`_compute_ee_rel_stats` and
        :meth:`_compute_ee_rel_world_stats` — identical logic, only the SE(3)
        relativisation functions differ (UMI body frame vs experimental world
        frame). Kept as one implementation so a future stats-computation fix
        can't accidentally be applied to only one of the two frame conventions.

        Both observation.state and action are transformed to SE(3)-relative with the
        SAME anchor (current EE pose).  Stats are computed from the actual training
        distributions so normalization is valid.

        Action target at horizon step k (anchored to state[t]):
            delta[k]       = action_forward(act_xyz[t+k], R_act[t+k]; anchor=state[t])
            gripper[k]     = action_gripper[t+k]   (absolute, stats restored)

        Obs target at obs step j (anchored to state[t], j = 0..n_obs_steps-1):
            identity step (j == n_obs_steps-1): all zeros + [1,0,0,0,1,0] + abs gripper
            prior steps (j < n_obs_steps-1): obs[t-(n_obs_steps-1-j)] rel to obs[t]

        Returns a dict ``{"action": stats, "observation.state": obs_stats}``,
        or ``None`` on failure.
        """
        import numpy as np
        try:
            from anvil_shared.ee_transform import EE_ACTION_DIM_PER_ARM, n_arms_from_dims

            hf = full_dataset.hf_dataset
            actions_np = np.array(hf["action"], dtype=np.float64)       # (N, 10*n_arms)
            states_np = np.array(hf["observation.state"], dtype=np.float64)  # (N, 8*n_arms)
            episode_idx_np = np.array(hf["episode_index"], dtype=np.int64).ravel()

            if states_np.ndim == 3:
                states_np = states_np[:, -1, :]  # multi-step obs → most recent step

            n_arms = n_arms_from_dims(states_np.shape[-1], actions_np.shape[-1])

            # ------------------------------------------------------------------ #
            # Action stats (relative to current state, per-sample anchor)        #
            # ------------------------------------------------------------------ #
            # Use action_delta_indices directly so the distribution exactly matches
            # what EERelTransform.apply produces at __getitem__ time.
            # lerobot default: action_delta_indices = range(1-n_obs_steps, 1-n_obs_steps+horizon)
            #   e.g. [-1, 0, …, 14] for n_obs_steps=2, horizon=16.
            # Using range(n_steps) instead would shift origin by +1 (omit t-1, include t+15),
            # giving wrong min/max for MIN_MAX normalization.
            action_delta_indices = getattr(cfg.policy, "action_delta_indices", None)
            if not action_delta_indices:
                action_delta_indices = [0]
            N = len(actions_np)

            def _rel_action_for_delta(d: int) -> np.ndarray:
                """Return SE(3) relative array for action offset d (may be negative), episode-bounded."""
                if d == 0:
                    act, sta, mask = actions_np, states_np, np.ones(N, dtype=bool)
                elif d > 0:
                    act = actions_np[d:]
                    sta = states_np[:-d]
                    mask = episode_idx_np[d:] == episode_idx_np[:-d]
                else:
                    k = -d
                    act = actions_np[:-k]
                    sta = states_np[k:]
                    mask = episode_idx_np[:-k] == episode_idx_np[k:]
                return action_forward(act, sta)[mask]

            all_deltas = np.concatenate(
                [_rel_action_for_delta(d) for d in action_delta_indices], axis=0
            )  # (N_valid_pairs, 10*n_arms)

            orig_action = full_dataset.meta.stats.get("action", {})
            delta_mean = all_deltas.mean(axis=0)
            delta_std = np.where(all_deltas.std(axis=0) < 1e-6, 1e-6, all_deltas.std(axis=0))
            delta_min = all_deltas.min(axis=0)
            delta_max = all_deltas.max(axis=0)

            # Restore gripper stats to absolute range
            orig_arr = lambda key, fallback: np.array(orig_action.get(key, fallback))
            for arm in range(n_arms):
                grip_idx = arm * EE_ACTION_DIM_PER_ARM + 9
                for arr, key in [
                    (delta_mean, "mean"), (delta_std, "std"),
                    (delta_min, "min"), (delta_max, "max"),
                ]:
                    orig_vals = orig_arr(key, arr)
                    if grip_idx < len(orig_vals):
                        arr[grip_idx] = orig_vals[grip_idx]

            # rot6d identity trick: force min=-1/max=1 for rot6d dims (index 3-8 per arm).
            # With MIN_MAX normalization, 2*(x-(-1))/(1-(-1))-1 = x, so rot6d passes through
            # unchanged — matching UMI's rot6d→identity design.  pos (0-2) and gripper (9)
            # retain their real distribution and get range-normalised to [-1,1] as intended.
            _force_rot6d_identity(delta_min, delta_max, n_arms)

            action_patched_stats = {
                "mean": delta_mean.tolist(),
                "std": delta_std.tolist(),
                "min": delta_min.tolist(),
                "max": delta_max.tolist(),
                "count": orig_action.get("count", len(all_deltas)),
            }
            full_dataset.meta.stats["action"] = action_patched_stats

            # ------------------------------------------------------------------ #
            # Obs stats (relative to current state, 10-dim rot6d layout)         #
            # ------------------------------------------------------------------ #
            # Identity step: obs[t] relative to obs[t] — always zeros + identity rot6d
            # Prior steps: obs[t-j] relative to obs[t]  (j = 1..n_obs_steps-1)
            # We compute mean/std/min/max over all; rot6d dims are then forced to
            # min=-1/max=1 so they pass through unchanged under MIN_MAX normalization.
            n_obs_steps = getattr(cfg.policy, "n_obs_steps", 2)

            obs_rel_samples = []
            # Identity steps — all N frames (obs relative to itself)
            identity = obs_forward(states_np, states_np)  # zeros+[1,0,0,0,1,0]+grip
            obs_rel_samples.append(identity)

            # Prior steps — obs[t-j] relative to obs[t], episode-bounded
            for j in range(1, n_obs_steps):
                past = states_np[:-j]
                anchor = states_np[j:]
                mask = episode_idx_np[:-j] == episode_idx_np[j:]
                rel = obs_forward(past, anchor)
                obs_rel_samples.append(rel[mask])

            all_obs_rel = np.concatenate(obs_rel_samples, axis=0)  # (N_total, 10*n_arms)

            obs_mean = all_obs_rel.mean(axis=0)
            obs_std = np.where(all_obs_rel.std(axis=0) < 1e-6, 1e-6, all_obs_rel.std(axis=0))
            obs_min = all_obs_rel.min(axis=0)
            obs_max = all_obs_rel.max(axis=0)

            # rot6d identity trick (same as action): force min=-1/max=1 for rot6d dims.
            _force_rot6d_identity(obs_min, obs_max, n_arms)

            obs_patched_stats = {
                "mean": obs_mean.tolist(),
                "std": obs_std.tolist(),
                "min": obs_min.tolist(),
                "max": obs_max.tolist(),
                "count": len(all_obs_rel),
            }
            full_dataset.meta.stats["observation.state"] = obs_patched_stats

            log.info(
                "[%s] action: %d samples (n_delta_steps=%d); obs: %d samples "
                "(n_obs_steps=%d); %d arm(s)",
                log_tag, len(all_deltas), len(action_delta_indices), len(all_obs_rel), n_obs_steps, n_arms,
            )
            return {"action": action_patched_stats, "observation.state": obs_patched_stats}
        except DataIntegrityError:
            raise
        except Exception as e:
            log.warning("[%s] Failed: %s — falling back to absolute stats", log_tag, e)
            return None

    def _compute_ee_abs_stats(self, full_dataset: Any, cfg: Any) -> dict | None:
        """Compute EE absolute action AND obs stats for ``ee_abs`` training.

        Action stats are taken from the dataset as-is (already absolute rot6d),
        with rot6d dims (index 3–8 per arm) forced to min=-1/max=1 so they pass
        through MIN_MAX normalization unchanged (identity trick).

        Obs stats are computed by converting every state frame from quaternion
        layout (8n) to rot6d layout (10n) via ``ee_obs_abs_forward``, then
        similarly clamping rot6d dims to ±1.  xyz and gripper dims keep their
        true absolute distributions.

        Returns a dict ``{"action": stats, "observation.state": obs_stats}``,
        or ``None`` on failure.
        """
        if not self.config.is_ee_abs:
            return None

        import numpy as np
        try:
            from anvil_shared.ee_transform import (
                ee_obs_abs_forward,
                n_arms_from_dims,
                EE_ACTION_DIM_PER_ARM,
            )

            hf = full_dataset.hf_dataset
            actions_np = np.array(hf["action"], dtype=np.float64)       # (N, 10*n_arms)
            states_np = np.array(hf["observation.state"], dtype=np.float64)  # (N, 8*n_arms)

            if states_np.ndim == 3:
                states_np = states_np[:, -1, :]  # multi-step obs → most recent step

            n_arms = n_arms_from_dims(states_np.shape[-1], actions_np.shape[-1])

            # ------------------------------------------------------------------ #
            # Action stats: dataset already has absolute rot6d; just clamp rot6d #
            # dims to ±1 so Gram-Schmidt reconstruction stays geometrically valid.#
            # ------------------------------------------------------------------ #
            orig_action = full_dataset.meta.stats.get("action", {})
            act_mean = np.array(orig_action.get("mean", actions_np.mean(axis=0)))
            act_std = np.array(orig_action.get("std", np.where(
                actions_np.std(axis=0) < 1e-6, 1e-6, actions_np.std(axis=0)
            )))
            act_min = np.array(orig_action.get("min", actions_np.min(axis=0)))
            act_max = np.array(orig_action.get("max", actions_np.max(axis=0)))

            _force_rot6d_identity(act_min, act_max, n_arms)

            action_patched_stats = {
                "mean": act_mean.tolist(),
                "std": act_std.tolist(),
                "min": act_min.tolist(),
                "max": act_max.tolist(),
                "count": orig_action.get("count", len(actions_np)),
            }
            full_dataset.meta.stats["action"] = action_patched_stats

            # ------------------------------------------------------------------ #
            # Obs stats: convert all frames to rot6d, then clamp rot6d dims.     #
            # xyz and gripper retain their true absolute distributions.           #
            # ------------------------------------------------------------------ #
            all_obs_abs = ee_obs_abs_forward(states_np)  # (N, 10*n_arms)

            obs_mean = all_obs_abs.mean(axis=0)
            obs_std = np.where(all_obs_abs.std(axis=0) < 1e-6, 1e-6, all_obs_abs.std(axis=0))
            obs_min = all_obs_abs.min(axis=0)
            obs_max = all_obs_abs.max(axis=0)

            _force_rot6d_identity(obs_min, obs_max, n_arms)

            obs_patched_stats = {
                "mean": obs_mean.tolist(),
                "std": obs_std.tolist(),
                "min": obs_min.tolist(),
                "max": obs_max.tolist(),
                "count": len(all_obs_abs),
            }
            full_dataset.meta.stats["observation.state"] = obs_patched_stats

            log.info(
                "[ee_abs_stats] action rot6d dims clamped ±1; obs: %d frames → 10-dim rot6d; %d arm(s)",
                len(all_obs_abs), n_arms,
            )
            return {"action": action_patched_stats, "observation.state": obs_patched_stats}
        except DataIntegrityError:
            raise
        except Exception as e:
            log.warning("[ee_abs_stats] Failed: %s — falling back to dataset stats", e)
            return None

    def apply_metadata_patches(self) -> None:
        """Apply metadata patches before importing lerobot training."""
        for transform in self.active_transforms:
            transform.patch_metadata(self.config, runner=self)

    def apply_dataset_patches(self) -> None:
        """Patch LeRobotDataset.__getitem__ to apply transforms and fix index mapping.

        This patch is always installed (even without active_transforms) because
        EpisodeAwareSampler yields absolute frame indices that must be remapped to
        relative indices for filtered (split) datasets. The mapping is only applied
        to the train dataset instance (flagged via _anvil_uses_abs_sampler).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # We must capture the original __getitem__ to use it in our patch.
        # LeRobotDataset.__getitem__ in v0.5.1 does not perform index mapping,
        # but EpisodeAwareSampler yields absolute indices. We add the mapping
        # logic here to support filtered datasets (splits).
        original_getitem = LeRobotDataset.__getitem__
        transforms = self.active_transforms
        config = self.config

        def patched_getitem(self, idx):
            # 1. Resolve relative index if the dataset is filtered by episodes.
            # Only the train dataset uses EpisodeAwareSampler (absolute indices).
            # Val/test datasets use DataLoader without a sampler (relative indices
            # 0..N-1) and must NOT be remapped — doing so would corrupt reads when
            # relative indices overlap with the absolute frame index space.
            reader = self._ensure_reader()
            if getattr(self, '_anvil_uses_abs_sampler', False) and reader._absolute_to_relative_idx is not None:
                # Map from absolute HF frame index to relative filtered index
                idx = reader._absolute_to_relative_idx.get(idx, idx)

            # 2. Call original __getitem__ (which calls reader.get_item)
            item = original_getitem(self, idx)

            # 3. Apply transforms (no-op when transforms list is empty)
            for transform in transforms:
                item = transform.apply(item, config)
            return item

        self._patch(LeRobotDataset, "__getitem__", patched_getitem)
        log.info("[anvil_trainer] Patched LeRobotDataset.__getitem__ (%d transform(s))", len(transforms))

    def apply_val_loss_patch(self) -> None:
        """Monkey-patch make_dataset to create train/val/test splits, and capture preprocessor."""
        s = self.config.split_ratio
        total_r = sum(s)
        if total_r <= 0 or (s[1] <= 0 and (len(s) < 3 or s[2] <= 0)):
            return  # no val or test, skip patching

        import lerobot.datasets.factory as factory_mod
        import lerobot.policies.factory as policy_factory_mod
        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import torch
        from lerobot.datasets.factory import make_dataset as original_make_dataset

        val_state = self
        _patched = {"done": False}

        def patched_make_dataset(cfg):
            # Only intercept the first call (main process dataset creation).
            if _patched["done"]:
                return original_make_dataset(cfg)
            _patched["done"] = True

            # Capture val_freq and resume_step from lerobot cfg
            val_state._val_freq = cfg.log_freq * 5 if cfg.log_freq > 0 else 0
            if cfg.resume and hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
                try:
                    step_file = Path(cfg.checkpoint_path) / "training_state" / "training_step.json"
                    if step_file.exists():
                        val_state._resume_step = json.loads(step_file.read_text()).get("step", 0)
                except Exception:
                    val_state._resume_step = 0

            # Full dataset to determine total episode count
            full_dataset = original_make_dataset(cfg)
            total_ep = full_dataset.num_episodes

            # Compute stats patches for EE action types.
            # ee_rel: SE(3)-relative stats with rot6d identity trick.
            # ee_abs: absolute rot6d obs stats + rot6d identity trick on action.
            # joint_abs: use dataset stats as-is (no patching needed).
            if val_state.config.is_ee_rel:
                _patched_ee_stats = val_state._compute_ee_rel_stats(full_dataset, cfg)
            elif val_state.config.is_ee_rel_world:
                _patched_ee_stats = val_state._compute_ee_rel_world_stats(full_dataset, cfg)
            elif val_state.config.is_ee_abs:
                _patched_ee_stats = val_state._compute_ee_abs_stats(full_dataset, cfg)
            else:
                _patched_ee_stats = None

            # Check if split_info.json already exists in last checkpoint (for resume)
            split_info_path = Path(cfg.output_dir) / "checkpoints" / "last" / "pretrained_model" / "split_info.json"
            loaded_split = load_split_info(split_info_path)
            if loaded_split is not None:
                train_ep = loaded_split.get("train_episodes", [])
                val_ep = loaded_split.get("val_episodes", [])
                test_ep = loaded_split.get("test_episodes", [])
                log.info("[split] Loaded random splits from %s", split_info_path)
            else:
                train_ep = val_ep = test_ep = None

            if train_ep is None:
                # Optional: subsample N episodes before splitting
                import random as _random
                pool = list(range(total_ep))
                max_ep = val_state.config.max_episodes
                if max_ep is not None and max_ep < total_ep:
                    _rng = _random.Random(cfg.seed)
                    pool = sorted(_rng.sample(pool, max_ep))
                    log.info("[split] Subsampled %d / %d episodes (--max-episodes)", len(pool), total_ep)

                # Random three-way split via shared helper (seeded for reproducibility)
                splits = compute_split_episodes(len(pool), s, seed=cfg.seed)
                train_ep = [pool[i] for i in splits["train"]]
                val_ep = [pool[i] for i in splits["val"]]
                test_ep = [pool[i] for i in splits["test"]]

                if len(train_ep) < 1:
                    log.warning("[split] Not enough episodes (%d) for split %s, using all for training", total_ep, s)
                    return full_dataset
                log.info("[split] Generated random splits")

            # Store split info for anvil_config.json (as full lists now)
            val_state._split_info = {
                "split_ratio": list(s),
                "total_episodes": total_ep,
                "train_episodes": train_ep,
                "val_episodes": val_ep,
                "test_episodes": test_ep,
                **({"max_episodes": val_state.config.max_episodes} if val_state.config.max_episodes is not None else {}),
            }

            def _make_dataloader(dataset):
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    prefetch_factor=2 if cfg.num_workers > 0 else None,
                )

            # Val dataloader
            if val_ep:
                cfg.dataset.episodes = val_ep
                val_dataset = original_make_dataset(cfg)
                val_state._val_dataloader = _make_dataloader(val_dataset)
                log.info("[split] val=%d ep (randomly selected, %d frames)", len(val_ep), val_dataset.num_frames)

            # Test dataloader
            if test_ep:
                cfg.dataset.episodes = test_ep
                test_dataset = original_make_dataset(cfg)
                val_state._test_dataloader = _make_dataloader(test_dataset)
                log.info("[split] test=%d ep (randomly selected, %d frames)", len(test_ep), test_dataset.num_frames)

            # Train dataset
            cfg.dataset.episodes = train_ep
            train_dataset = original_make_dataset(cfg)
            # Flag this instance so patched_getitem applies absolute→relative mapping.
            # EpisodeAwareSampler (used by ACT and similar policies) yields absolute
            # frame indices; val/test dataloaders use relative indices and must NOT
            # be remapped.
            train_dataset._anvil_uses_abs_sampler = True
            # Inject EE rel stats so lerobot's normalizer uses relative distributions.
            if _patched_ee_stats is not None:
                train_dataset.meta.stats["action"] = _patched_ee_stats["action"]
                train_dataset.meta.stats["observation.state"] = _patched_ee_stats["observation.state"]
                log.info("[anvil_trainer] Patched train_dataset.meta.stats [action + observation.state]")
            log.info("[split] train=%d ep (randomly selected)", len(train_ep))
            return train_dataset

        self._patch(factory_mod, "make_dataset", patched_make_dataset)
        self._patch(lerobot_train_mod, "make_dataset", patched_make_dataset)
        log.info("[split] Patched make_dataset (split_ratio=%s, random=True)", s)

        # Capture preprocessor when it's created by lerobot
        original_make_processors = policy_factory_mod.make_pre_post_processors

        def capturing_make_processors(*args, **kwargs):
            preprocessor, postprocessor = original_make_processors(*args, **kwargs)
            val_state._preprocessor = preprocessor
            return preprocessor, postprocessor

        self._patch(policy_factory_mod, "make_pre_post_processors", capturing_make_processors)
        self._patch(lerobot_train_mod, "make_pre_post_processors", capturing_make_processors)
        log.info("[split] Patched make_pre_post_processors to capture preprocessor")

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log test loss (if test split is active) at save_freq.
        2. Write anvil_config.json (with split info) into each checkpoint's pretrained_model/ directory.
        """
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg_base: dict = {
            "action_type": self.config.action_type,
            "is_ee": self.config.is_ee,
            "is_ee_rel": self.config.is_ee_rel,
            "is_ee_rel_world": self.config.is_ee_rel_world,
            **git_provenance(),
        }
        if self.config.task_override:
            anvil_cfg_base["task_description"] = self.config.task_override
        if self.config.note:
            anvil_cfg_base["note"] = self.config.note

        val_state = self

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            policy = kwargs.get("policy")
            preprocessor = kwargs.get("preprocessor") or val_state._preprocessor
            step = kwargs.get("step", "?")

            # EMA model — present only after the first optimizer step (lazy init).
            ema = getattr(val_state, "_ema", None)
            has_test = val_state._test_dataloader is not None and policy is not None

            # --- Test loss (raw policy weights) ---
            if has_test:
                t0 = time.perf_counter()
                test_loss = _compute_mean_loss(policy, val_state._test_dataloader, preprocessor)
                log.info("[eval] test_loss=%.6f @ step %s (%.1fs)", test_loss, step, time.perf_counter() - t0)
                _log_wandb({"eval/test_loss": test_loss}, step=int(step))

            # --- EMA: single swap for both EMA test loss AND checkpoint save ---
            # Clone raw weights once; swap to EMA; compute EMA loss + save; restore.
            raw_sd = None
            if ema is not None and policy is not None:
                raw_sd = {k: v.clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(ema.averaged_model.state_dict())

                # Test loss on EMA weights (policy is now EMA).
                if has_test:
                    t0 = time.perf_counter()
                    ema_test_loss = _compute_mean_loss(policy, val_state._test_dataloader, preprocessor)
                    log.info(
                        "[eval] test_loss_ema=%.6f @ step %s (%.1fs)",
                        ema_test_loss, step, time.perf_counter() - t0,
                    )
                    _log_wandb({"eval/test_loss_ema": ema_test_loss}, step=int(step))

            # --- Original save (pretrained_model/ = EMA weights when EMA active) ---
            try:
                original_save_checkpoint(checkpoint_dir, **kwargs)
            finally:
                # Always restore raw weights so training continues from raw trajectory.
                if raw_sd is not None and policy is not None:
                    policy.load_state_dict(raw_sd)
                    del raw_sd

            # --- Persist raw weights + EMA counter to training_state/ (for resume) ---
            if ema is not None and policy is not None:
                from safetensors.torch import save_file as _st_save

                training_dir = checkpoint_dir / "training_state"
                training_dir.mkdir(parents=True, exist_ok=True)
                # policy is now back to raw weights (restored above).
                _st_save(
                    {k: v.contiguous() for k, v in policy.state_dict().items()},
                    str(training_dir / "model_raw.safetensors"),
                )
                (training_dir / "ema_state.json").write_text(json.dumps(ema.state_dict()))
                log.info("[ema] Saved raw weights + EMA state to %s", training_dir)

            # --- Save split_info.json and anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                # 1. anvil_config.json: only non-split flags
                (pretrained_dir / "anvil_config.json").write_text(json.dumps(anvil_cfg_base, indent=2))

                # 2. split_info.json: all split metadata
                if val_state._split_info:
                    save_split_info(pretrained_dir / "split_info.json", val_state._split_info)

                log.info("[anvil_trainer] Saved configs to %s", pretrained_dir)

        # Patch both the module and the already-imported reference in lerobot_train
        self._patch(train_utils_mod, "save_checkpoint", patched_save_checkpoint)
        self._patch(lerobot_train_mod, "save_checkpoint", patched_save_checkpoint)
        log.info("[anvil_trainer] Patched save_checkpoint for test loss + anvil_config.json")

    def apply_val_loss_hook(self) -> None:
        """Monkey-patch update_policy for periodic val loss computation at val_freq intervals."""
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod

        original_update_policy = lerobot_train_mod.update_policy
        val_state = self
        _counter = {"n": 0}

        def patched_update_policy(
            train_metrics, policy, batch, optimizer, grad_clip_norm,
            accelerator=None, lr_scheduler=None, lock=None, rabc_weights_provider=None,
        ):
            result = original_update_policy(
                train_metrics, policy, batch, optimizer, grad_clip_norm,
                accelerator=accelerator, lr_scheduler=lr_scheduler,
                lock=lock, rabc_weights_provider=rabc_weights_provider,
            )

            _counter["n"] += 1
            val_freq = val_state._val_freq
            if not val_freq or val_freq <= 0 or val_state._val_dataloader is None:
                return result
            if _counter["n"] % val_freq != 0:
                return result

            abs_step = val_state._resume_step + _counter["n"]
            preprocessor = val_state._preprocessor

            # Unwrap accelerator-wrapped policy for eval
            if accelerator is not None:
                unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            else:
                unwrapped = policy

            t0 = time.perf_counter()
            val_loss = _compute_mean_loss(unwrapped, val_state._val_dataloader, preprocessor)
            val_s = time.perf_counter() - t0
            log.info("[eval] val_loss=%.6f @ step %s (%.1fs)", val_loss, abs_step, val_s)
            _log_wandb({"eval/val_loss": val_loss}, step=abs_step)

            return result

        self._patch(lerobot_train_mod, "update_policy", patched_update_policy)
        log.info("[eval] Patched update_policy for periodic val loss (val_freq will be log_freq*5)")

    def apply_ddpm_ip_patch(self) -> None:
        """Monkey-patch DiffusionModel.compute_loss to add DDPM-IP input perturbation.

        DDPM-IP (Input Perturbation, Ning et al. 2023) adds a small extra noise
        perturbation to the noise sample used for ``add_noise`` during training:

            eps_perturbed = eps + alpha * randn_like(eps)
            noisy = add_noise(x_0, eps_perturbed, t)
            target = eps                               # original noise, NOT perturbed

        This forces the model to handle slightly-off inputs, reducing exposure bias
        in closed-loop inference where each step receives the previous step's
        (imperfect) output rather than ground-truth.

        Matches UMI's ``input_pertub=0.1`` exactly.  Disabled with ``--no-ddpm-ip``.
        """
        if not self.config.use_ddpm_ip:
            log.info("[ddpm-ip] DDPM-IP disabled (--no-ddpm-ip). Skipping patch.")
            return

        import torch
        import torch.nn.functional as F
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

        alpha = self.config.ddpm_ip_alpha
        original_compute_loss = DiffusionModel.compute_loss

        def patched_compute_loss(model_self, batch):
            # --- Input validation (unchanged from original) ---
            assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
            assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
            assert batch[ACTION].shape[1] == model_self.config.horizon
            assert batch[OBS_STATE].shape[1] == model_self.config.n_obs_steps

            global_cond = model_self._prepare_global_conditioning(batch)

            trajectory = batch[ACTION]
            eps = torch.randn(trajectory.shape, device=trajectory.device)
            timesteps = torch.randint(
                low=0,
                high=model_self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()

            # DDPM-IP: perturb the noise used for add_noise, keep original eps as target.
            # Reference: https://github.com/forever208/DDPM-IP  (matches UMI input_pertub=0.1)
            eps_perturbed = eps + alpha * torch.randn_like(eps)
            noisy_trajectory = model_self.noise_scheduler.add_noise(trajectory, eps_perturbed, timesteps)

            pred = model_self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

            # Target is always the ORIGINAL eps (not eps_perturbed) — this is the key.
            if model_self.config.prediction_type == "epsilon":
                target = eps
            elif model_self.config.prediction_type == "sample":
                target = batch[ACTION]
            else:
                raise ValueError(f"Unsupported prediction type {model_self.config.prediction_type}")

            loss = F.mse_loss(pred, target, reduction="none")

            if model_self.config.do_mask_loss_for_padding:
                if "action_is_pad" not in batch:
                    raise ValueError(
                        "You need to provide 'action_is_pad' in the batch when "
                        f"{model_self.config.do_mask_loss_for_padding=}."
                    )
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)

            return loss.mean()

        self._patch(DiffusionModel, "compute_loss", patched_compute_loss)
        log.info("[ddpm-ip] Patched DiffusionModel.compute_loss with DDPM-IP (alpha=%.2f)", alpha)

    def apply_ema_hook(self) -> None:
        """Monkey-patch update_policy to maintain an EMA of policy weights.

        Enabled by default; disable with ``--no-ema`` (``config.use_ema=False``).

        On the **first** call to ``update_policy`` (lazy init):
        - **Fresh run**: deepcopy the live policy as the EMA seed.
        - **Resume**: load ``training_state/model_raw.safetensors`` back into the
          live model (aligning it with optimizer moments); seed the EMA averaged_model
          from the current live policy (= EMA weights in ``pretrained_model/``);
          restore the EMA counter from ``training_state/ema_state.json``.

        On every subsequent call: ``ema.step(unwrapped_policy)`` after the
        original ``update_policy`` returns.

        The EMA model is stored on ``self._ema`` so ``patched_save_checkpoint``
        can access it to swap weights and persist state.
        """
        if not self.config.use_ema:
            log.info("[ema] EMA disabled (--no-ema). Skipping apply_ema_hook.")
            return

        import copy

        import lerobot.scripts.lerobot_train as lerobot_train_mod

        from anvil_trainer.ema import EMAModel

        original_update_policy = lerobot_train_mod.update_policy
        _state: dict = {"initialized": False}
        runner = self

        def _build_fresh_ema(model) -> EMAModel:
            """Seed a new EMAModel from *model* (deepcopy) with runner config hyperparams."""
            ema = EMAModel(
                copy.deepcopy(model),
                inv_gamma=runner.config.ema_inv_gamma,
                power=runner.config.ema_power,
                max_value=runner.config.ema_max_value,
            )
            ema.averaged_model.to(next(model.parameters()).device)
            return ema

        def patched_update_policy(
            train_metrics, policy, batch, optimizer, grad_clip_norm,
            accelerator=None, lr_scheduler=None, lock=None, rabc_weights_provider=None,
        ):
            result = original_update_policy(
                train_metrics, policy, batch, optimizer, grad_clip_norm,
                accelerator=accelerator, lr_scheduler=lr_scheduler,
                lock=lock, rabc_weights_provider=rabc_weights_provider,
            )

            unwrapped = (
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                if accelerator is not None
                else policy
            )

            # Lazy EMA initialisation on the first call.
            if not _state["initialized"]:
                _state["initialized"] = True

                if runner.config.resume_job_path:
                    # Resume path: pretrained_model/ holds EMA weights (already loaded
                    # by lerobot into the live policy).  We deepcopy those as the EMA
                    # averaged_model seed, then reload raw weights into the live policy.
                    from pathlib import Path

                    ckpt_root = (
                        Path(runner.config.resume_job_path)
                        / "checkpoints"
                        / runner.config.resume_checkpoint
                    )
                    training_state_dir = ckpt_root / "training_state"
                    ema = EMAModel.load_from_dir(training_state_dir, unwrapped)
                    if ema is None:
                        # Old checkpoint without EMA state — start fresh from current weights.
                        log.warning(
                            "[ema] No ema_state.json found at %s. "
                            "Starting EMA from scratch (counter at 0).",
                            training_state_dir,
                        )
                        ema = _build_fresh_ema(unwrapped)
                else:
                    # Fresh run: seed EMA from initial policy weights.
                    ema = _build_fresh_ema(unwrapped)
                    log.info(
                        "[ema] EMAModel initialised: power=%.4f  max_value=%.4f  inv_gamma=%.1f",
                        runner.config.ema_power,
                        runner.config.ema_max_value,
                        runner.config.ema_inv_gamma,
                    )

                runner._ema = ema

            runner._ema.step(unwrapped)
            return result

        self._patch(lerobot_train_mod, "update_policy", patched_update_policy)
        log.info(
            "[ema] Patched update_policy for EMA (power=%.4f max_value=%.4f)",
            self.config.ema_power,
            self.config.ema_max_value,
        )


# =============================================================================
# Context manager
# =============================================================================


@contextlib.contextmanager
def patched_lerobot(config: TrainingConfig):
    """Install every anvil-trainer patch for the duration of the ``with`` block.

    Yields the constructed :class:`TransformRunner` so the caller can inspect
    split info, the captured preprocessor, etc.  On exit (normal or via an
    exception), every touched lerobot attribute is restored — so training
    failures no longer leave lerobot's module state permanently mutated, and
    tests run back-to-back without polluting each other.

    Example::

        with patched_lerobot(config) as runner:
            from lerobot.scripts.lerobot_train import train as lerobot_train
            lerobot_train()
    """
    runner = TransformRunner(config)
    runner.log_config()
    runner.apply_metadata_patches()
    runner.apply_processor_compat_aliases()
    # Note: the dataset/val_loss/checkpoint patches need lerobot imported,
    # which apply_metadata_patches typically triggers indirectly via
    # Transform.patch_metadata.  Keep the same install order as train().
    runner.apply_dataset_patches()
    runner.apply_val_loss_patch()
    runner.apply_checkpoint_patch()
    runner.apply_val_loss_hook()
    runner.apply_ddpm_ip_patch()
    runner.apply_ema_hook()
    try:
        yield runner
    finally:
        runner.restore_all_patches()
