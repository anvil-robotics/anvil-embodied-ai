"""Monkey-patches applied to lerobot at training time.

``TransformRunner`` owns:
    * The active list of :class:`~anvil_trainer.transforms.Transform` instances.
    * Five monkey-patches on lerobot modules:
        - ``apply_dataset_patches`` — patches ``LeRobotDataset.__getitem__``.
        - ``apply_val_loss_patch`` — patches ``make_dataset`` (split creation),
          captures the preprocessor from ``make_pre_post_processors``, and
          injects delta action stats into the returned ``train_dataset``.
        - ``apply_checkpoint_patch`` — patches ``save_checkpoint`` to compute
          test loss and write ``anvil_config.json`` / ``split_info.json`` next
          to each checkpoint.
        - ``apply_val_loss_hook`` — patches ``update_policy`` for periodic val
          loss computation.
        - ``apply_metadata_patches`` — runs ``Transform.patch_metadata`` hooks
          (currently used by ``ExcludeObservationTransform``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from anvil_trainer.config import TrainingConfig
from anvil_trainer.transforms import (
    DeltaActionTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

log = logging.getLogger(__name__)


class TransformRunner:
    """
    Manages and applies dataset transforms.

    Handles:
    - Registration of transforms
    - Metadata patching (before lerobot import)
    - Dataset patching (after lerobot import)
    """

    # Registry of available transforms (add new transforms here).
    # Instantiated fresh per TransformRunner so stateful transforms (e.g. DeltaActionTransform
    # which caches joint indices) do not share state across runs.
    TRANSFORMS: list[Transform] = []  # populated in __init__

    def __init__(self, config: TrainingConfig):
        self.config = config
        transforms: list[Transform] = [
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            DeltaActionTransform(),
        ]
        self.active_transforms = [t for t in transforms if t.is_enabled(config)]
        self._val_dataloader = None   # set by apply_val_loss_patch when make_dataset is called
        self._test_dataloader = None  # set by apply_val_loss_patch when make_dataset is called
        self._split_info: dict = {}   # populated by patched_make_dataset
        self._preprocessor = None     # captured from make_pre_post_processors
        self._val_freq = 0            # set from cfg.log_freq * 5 inside patched_make_dataset
        self._resume_step = 0         # for absolute step tracking in wandb

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
            return f"excluding: {', '.join(self.config.exclude_observation)}"
        elif isinstance(transform, TaskOverrideTransform):
            return f"'{self.config.task_override}'"
        elif isinstance(transform, DeltaActionTransform):
            return "action = action - observation.state"
        return "enabled"

    def _compute_delta_action_stats(self, full_dataset: Any) -> dict | None:
        """Compute delta action stats when DeltaActionTransform is active.

        LeRobot reads ``dataset.meta.stats`` after ``make_dataset()`` returns
        and uses those stats to build the normalizer.  Because
        ``DeltaActionTransform`` runs at ``__getitem__`` time (after stats are
        read), a vanilla setup normalises delta actions with *absolute* action
        statistics — producing a large constant offset in normalised space
        (e.g. −2.66 σ for joint j4) that causes early overfitting.

        This method computes delta action stats from the HF dataset arrays
        (vectorised, fast — ~5 MB for 69K frames) and returns the dict.  It
        also patches ``full_dataset.meta.stats["action"]`` in place so that
        the early-return path (``n_train < 1`` case) inherits the correct
        stats.

        Args:
            full_dataset: LeRobotDataset spanning all episodes.

        Returns:
            Patched action stats dict, or ``None`` when DeltaActionTransform
            is inactive or the computation fails (logged as a warning).
        """
        delta_transform = next(
            (t for t in self.active_transforms if isinstance(t, DeltaActionTransform)),
            None,
        )
        if delta_transform is None:
            return None

        import numpy as np  # numpy is not top-level; keep import local
        try:
            hf = full_dataset.hf_dataset
            actions_np = np.array(hf["action"], dtype=np.float64)
            states_np = np.array(hf["observation.state"], dtype=np.float64)
            # Use most recent obs step if states are stacked (n_obs_steps > 1)
            if states_np.ndim == 3:
                states_np = states_np[:, -1, :]
            d_action = actions_np.shape[-1]
            d_state = states_np.shape[-1]
            n = min(d_action, d_state)

            deltas = actions_np.copy()
            deltas[:, :n] = actions_np[:, :n] - states_np[:, :n]

            exclude_indices = delta_transform._resolve_exclude_indices(self.config)
            orig = full_dataset.meta.stats.get("action", {})
            orig_mean = np.array(orig.get("mean", deltas.mean(axis=0)))
            orig_std = np.array(orig.get("std", deltas.std(axis=0)))
            orig_min = np.array(orig.get("min", deltas.min(axis=0)))
            orig_max = np.array(orig.get("max", deltas.max(axis=0)))

            # Keep excluded joints' values in absolute space before taking stats
            for idx in exclude_indices:
                if idx < d_action:
                    deltas[:, idx] = actions_np[:, idx]

            delta_mean = deltas.mean(axis=0)
            delta_std = np.where(deltas.std(axis=0) < 1e-6, 1e-6, deltas.std(axis=0))
            delta_min = deltas.min(axis=0)
            delta_max = deltas.max(axis=0)

            # Restore excluded joints to their original absolute stats
            for idx in exclude_indices:
                if idx < d_action:
                    delta_mean[idx] = orig_mean[idx]
                    delta_std[idx] = orig_std[idx]
                    delta_min[idx] = orig_min[idx]
                    delta_max[idx] = orig_max[idx]

            patched_stats = {
                "mean": delta_mean.tolist(),
                "std": delta_std.tolist(),
                "min": delta_min.tolist(),
                "max": delta_max.tolist(),
                "count": orig.get("count", len(deltas)),
            }
            # Patch full_dataset in place so the early-return path is covered
            full_dataset.meta.stats["action"] = patched_stats

            log.info(
                "[delta_stats] Computed delta action stats over %d frames, %d joints "
                "(%d kept absolute: %s). "
                "j4 abs_mean=%.3f → delta_mean=%.4f, abs_std=%.3f → delta_std=%.4f",
                len(deltas), d_action, len(exclude_indices),
                self.config.delta_exclude_joints or [],
                orig_mean[4] if len(orig_mean) > 4 else float("nan"),
                delta_mean[4] if len(delta_mean) > 4 else float("nan"),
                orig_std[4] if len(orig_std) > 4 else float("nan"),
                delta_std[4] if len(delta_std) > 4 else float("nan"),
            )
            return patched_stats
        except Exception as e:
            log.warning(
                "[delta_stats] Failed to compute delta action stats: %s — "
                "falling back to absolute stats (training may be suboptimal)", e
            )
            return None

    def apply_metadata_patches(self) -> None:
        """Apply metadata patches before importing lerobot training."""
        for transform in self.active_transforms:
            transform.patch_metadata(self.config)

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

        LeRobotDataset.__getitem__ = patched_getitem
        log.info("[anvil_trainer] Patched LeRobotDataset.__getitem__ (%d transform(s))", len(transforms))

    def apply_val_loss_patch(self) -> None:
        """Monkey-patch make_dataset to create train/val/test splits, and capture preprocessor."""
        s = self.config.split_ratio
        total_r = sum(s)
        if total_r <= 0 or (s[1] <= 0 and (len(s) < 3 or s[2] <= 0)):
            return  # no val or test, skip patching

        import random

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

            # Compute delta action stats when DeltaActionTransform is active so
            # that lerobot's normalizer is built against delta statistics rather
            # than the absolute-action stats stored in meta/stats.json.  The
            # helper patches full_dataset.meta.stats in place for the early-
            # return path and returns the dict so we can re-apply it to the
            # filtered train_dataset below.
            _patched_action_stats = val_state._compute_delta_action_stats(full_dataset)

            # Check if split_info.json already exists in last checkpoint (for resume)
            split_info_path = Path(cfg.output_dir) / "checkpoints" / "last" / "pretrained_model" / "split_info.json"
            if split_info_path.exists():
                try:
                    loaded_split = json.loads(split_info_path.read_text())
                    train_ep = loaded_split.get("train_episodes", [])
                    val_ep = loaded_split.get("val_episodes", [])
                    test_ep = loaded_split.get("test_episodes", [])
                    log.info("[split] Loaded random splits from %s", split_info_path)
                except Exception as e:
                    log.warning("[split] Failed to load %s: %s. Re-splitting...", split_info_path, e)
                    train_ep = val_ep = test_ep = None
            else:
                train_ep = val_ep = test_ep = None

            if train_ep is None:
                # Random three-way split
                n_test = max(1, round(total_ep * s[2] / total_r)) if len(s) > 2 and s[2] > 0 else 0
                n_val = max(1, round(total_ep * s[1] / total_r)) if s[1] > 0 else 0
                n_train = total_ep - n_val - n_test

                if n_train < 1:
                    log.warning("[split] Not enough episodes (%d) for split %s, using all for training", total_ep, s)
                    # full_dataset already has patched action stats if _patched_action_stats is set
                    return full_dataset

                all_eps = list(range(total_ep))
                # Use a fixed seed for splitting to ensure consistency if re-run without split_info.json
                rng = random.Random(cfg.seed)
                rng.shuffle(all_eps)

                train_ep = sorted(all_eps[:n_train])
                val_ep = sorted(all_eps[n_train : n_train + n_val])
                test_ep = sorted(all_eps[n_train + n_val :])
                log.info("[split] Generated random splits")

            # Store split info for anvil_config.json (as full lists now)
            val_state._split_info = {
                "split_ratio": list(s),
                "total_episodes": total_ep,
                "train_episodes": train_ep,
                "val_episodes": val_ep,
                "test_episodes": test_ep,
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
            # Inject delta action stats so lerobot's normalizer uses the correct
            # distribution.  train_dataset is a fresh LeRobotDataset instance that
            # re-reads stats.json, so we must patch it explicitly here.
            if _patched_action_stats is not None:
                train_dataset.meta.stats["action"] = _patched_action_stats
                log.info("[delta_stats] Patched train_dataset.meta.stats['action'] with delta stats")
            log.info("[split] train=%d ep (randomly selected)", len(train_ep))
            return train_dataset

        factory_mod.make_dataset = patched_make_dataset
        lerobot_train_mod.make_dataset = patched_make_dataset
        log.info("[split] Patched make_dataset (split_ratio=%s, random=True)", s)

        # Capture preprocessor when it's created by lerobot
        original_make_processors = policy_factory_mod.make_pre_post_processors

        def capturing_make_processors(*args, **kwargs):
            preprocessor, postprocessor = original_make_processors(*args, **kwargs)
            val_state._preprocessor = preprocessor
            return preprocessor, postprocessor

        policy_factory_mod.make_pre_post_processors = capturing_make_processors
        lerobot_train_mod.make_pre_post_processors = capturing_make_processors
        log.info("[split] Patched make_pre_post_processors to capture preprocessor")

    def apply_checkpoint_patch(self) -> None:
        """Monkey-patch lerobot save_checkpoint to:
        1. Compute and log test loss (if test split is active) at save_freq.
        2. Write anvil_config.json (with split info) into each checkpoint's pretrained_model/ directory.
        """
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import lerobot.utils.train_utils as train_utils_mod
        import torch
        from lerobot.utils.train_utils import save_checkpoint as original_save_checkpoint

        anvil_cfg_base: dict = {"use_delta_actions": self.config.use_delta_actions}
        if self.config.delta_exclude_joints:
            anvil_cfg_base["delta_exclude_joints"] = self.config.delta_exclude_joints
        if self.config.task_override:
            anvil_cfg_base["task_description"] = self.config.task_override
        if self.config.note:
            anvil_cfg_base["note"] = self.config.note

        val_state = self

        def patched_save_checkpoint(checkpoint_dir, **kwargs):
            # --- Test loss (computed at save_freq) ---
            if val_state._test_dataloader is not None:
                policy = kwargs.get("policy")
                preprocessor = kwargs.get("preprocessor") or val_state._preprocessor
                step = kwargs.get("step", "?")

                if policy is not None:
                    policy.eval()
                    t0 = time.perf_counter()
                    total_loss = 0.0
                    n_batches = 0

                    # ACTPolicy in evaluation mode has no VAE, but test_loss
                    # needs to calculate the full loss. We set back to train mode
                    # to get the VAE loss if needed.
                    is_act = "ACTPolicy" in str(type(policy))
                    if is_act:
                        policy.train()

                    with torch.no_grad():
                        for batch in val_state._test_dataloader:
                            if preprocessor is not None:
                                batch = preprocessor(batch)
                            else:
                                device = next(policy.parameters()).device
                                batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                            loss, _ = policy.forward(batch)
                            total_loss += loss.item()
                            n_batches += 1

                    if is_act:
                        policy.eval()

                    test_loss = total_loss / max(n_batches, 1)
                    test_s = time.perf_counter() - t0
                    log.info("[eval] test_loss=%.6f @ step %s (%.1fs)", test_loss, step, test_s)

                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            _wandb.log({"eval/test_loss": test_loss}, step=int(step))
                    except Exception:
                        pass

            # --- Original save ---
            original_save_checkpoint(checkpoint_dir, **kwargs)

            # --- Save split_info.json and anvil_config.json ---
            pretrained_dir = checkpoint_dir / "pretrained_model"
            if pretrained_dir.exists():
                # 1. anvil_config.json: only non-split flags
                (pretrained_dir / "anvil_config.json").write_text(json.dumps(anvil_cfg_base, indent=2))

                # 2. split_info.json: all split metadata
                if val_state._split_info:
                    (pretrained_dir / "split_info.json").write_text(json.dumps(val_state._split_info, indent=2))

                log.info("[anvil_trainer] Saved configs to %s", pretrained_dir)

        # Patch both the module and the already-imported reference in lerobot_train
        train_utils_mod.save_checkpoint = patched_save_checkpoint
        lerobot_train_mod.save_checkpoint = patched_save_checkpoint
        log.info("[anvil_trainer] Patched save_checkpoint for test loss + anvil_config.json")

    def apply_val_loss_hook(self) -> None:
        """Monkey-patch update_policy for periodic val loss computation at val_freq intervals."""
        import time

        import lerobot.scripts.lerobot_train as lerobot_train_mod
        import torch

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

            unwrapped.eval()
            t0 = time.perf_counter()
            total_loss = 0.0
            n_batches = 0

            # ACTPolicy in evaluation mode has no VAE, but val_loss
            # needs to calculate the full loss.
            is_act = "ACTPolicy" in str(type(unwrapped))
            if is_act:
                unwrapped.train()

            with torch.no_grad():
                for val_batch in val_state._val_dataloader:
                    if preprocessor is not None:
                        val_batch = preprocessor(val_batch)
                    else:
                        device = next(unwrapped.parameters()).device
                        val_batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_batch.items()
                        }
                    loss, _ = unwrapped.forward(val_batch)
                    total_loss += loss.item()
                    n_batches += 1

            if is_act:
                unwrapped.eval()

            val_loss = total_loss / max(n_batches, 1)
            val_s = time.perf_counter() - t0
            log.info("[eval] val_loss=%.6f @ step %s (%.1fs)", val_loss, abs_step, val_s)

            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({"eval/val_loss": val_loss}, step=abs_step)
            except Exception:
                pass

            return result

        lerobot_train_mod.update_policy = patched_update_policy
        log.info("[eval] Patched update_policy for periodic val loss (val_freq will be log_freq*5)")
