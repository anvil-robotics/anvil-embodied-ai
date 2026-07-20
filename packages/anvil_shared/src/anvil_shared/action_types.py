"""Canonical ``action_type`` strings and legacy-alias normalization.

Shared across ``anvil_trainer`` (training-time config), ``anvil_eval`` /
``anvil_eval_ros`` (offline + ROS evaluation), and the ROS2 inference node so
the alias mapping is defined exactly once and cannot drift between call
sites.

Background
----------
``"ee_rel"`` was the original name for the chunk-anchor SE(3)-relative
mechanism: the action/obs window is relativized once per generated chunk,
anchored to the observation at chunk-generation time (relative to step
``n-0``). It was renamed to ``"ee_relative"`` to free up "rel"/"delta"
wording for a structurally different mechanism (per-frame anchor, relative
to step ``n->n+1``) built alongside it.

``"ee_rel"`` is a PERMANENT alias, not a deprecated value to migrate away
from: existing on-disk checkpoints persist ``action_type="ee_rel"`` in their
``anvil_config.json`` and must keep loading and behaving identically
forever.
"""
from __future__ import annotations

# Legacy → canonical action_type string aliases.
ACTION_TYPE_ALIASES: dict[str, str] = {
    "ee_rel": "ee_relative",
}


def normalize_action_type(action_type: str) -> str:
    """Map a legacy ``action_type`` string to its canonical form.

    Idempotent — normalizing an already-canonical value (or an unrecognized
    one) returns it unchanged. This is the single normalization point:
    callers should apply it once, as early as possible (right after parsing
    CLI args / reading ``anvil_config.json``), so everything downstream only
    ever compares against canonical values.
    """
    return ACTION_TYPE_ALIASES.get(action_type, action_type)
