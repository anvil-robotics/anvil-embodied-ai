"""Shared delta action restore utility.

restore_delta_chunk(chunk_np, obs_t, action_type, exclude_indices)
converts a chunk of raw model delta outputs to absolute joint positions.

  chunk_np        : (chunk_size, D) float64 ndarray — raw model output (delta space)
  obs_t           : (D,) float64 ndarray — observation state at chunk generation time
  action_type     : "delta_obs_t" | "delta_sequential"
  exclude_indices : set[int] | list[int] — joints kept absolute (not deltaized during training)
  returns         : (chunk_size, D) float64 ndarray — absolute actions

Both modes leave exclude_indices joints unchanged (they are already absolute in model output).

resolve_action_type(cfg) normalises the action_type field from a checkpoint anvil_config dict,
applying backward-compat mapping from the old use_delta_actions boolean.
"""
from __future__ import annotations

import numpy as np


def resolve_action_type(cfg: dict) -> str:
    """Return normalised action_type string from an anvil_config dict.

    Handles backward compat: old checkpoints without action_type but with
    use_delta_actions=True are mapped to "delta_obs_t".
    """
    action_type = cfg.get("action_type", "absolute")
    if action_type == "absolute" and cfg.get("use_delta_actions", False):
        return "delta_obs_t"
    return action_type


def restore_delta_chunk(
    chunk_np: np.ndarray,
    obs_t: np.ndarray,
    action_type: str,
    exclude_indices: set[int] | list[int],
) -> np.ndarray:
    """Restore a chunk of delta actions to absolute joint positions.

    Args:
        chunk_np: Raw model output, shape (chunk_size, D).
        obs_t: Observation state at chunk generation time, shape (D,).
        action_type: "delta_obs_t" or "delta_sequential".
        exclude_indices: Joint indices kept in absolute space during training.
            Pass a pre-computed set to avoid repeated conversion.

    Returns:
        Absolute actions, shape (chunk_size, D).
    """
    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)

    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    chunk_size, D = chunk_np.shape
    excl = exclude_indices if isinstance(exclude_indices, set) else set(exclude_indices)
    non_excl = [i for i in range(D) if i not in excl]

    abs_chunk = chunk_np.copy()  # excluded joints: keep raw value (already absolute)

    if action_type == "delta_obs_t":
        # All non-excluded joints: absolute = obs_t + delta (same reference for all k)
        if non_excl:
            abs_chunk[:, non_excl] = obs_t[non_excl] + chunk_np[:, non_excl]

    elif action_type == "delta_sequential":
        # Non-excluded joints: cumulative sum along time axis, then + obs_t
        if non_excl:
            abs_chunk[:, non_excl] = obs_t[non_excl] + np.cumsum(chunk_np[:, non_excl], axis=0)

    return abs_chunk
