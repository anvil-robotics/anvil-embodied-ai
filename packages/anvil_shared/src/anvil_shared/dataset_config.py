"""Lenient reader for a converted dataset's ``conversion_config.yaml``.

Every dataset ``mcap-convert`` produces writes a ``conversion_config.yaml`` recording its
own ``data_space`` / ``action_encoding`` / ``observation_encoding`` verbatim. This module
reads that file directly as ground truth — no guessing an EE dataset's shape from
feature-name suffixes — for any caller that just needs to know what's on disk, not the
full migration-aware ``mcap_converter.config.loader.ConfigLoader`` (which mcap_converter
itself still owns, since it also needs strict validation and legacy-schema migration when
actually converting; every dataset targeted here is already current-schema).

Deliberately lives in ``anvil_shared``, not ``mcap_converter``: the ROS2 inference image
does not ship ``mcap_converter``, but does ship ``anvil_shared`` (via each node's
``sys.path`` shim), so this is the one reader every consumer — anvil_trainer, anvil_eval,
anvil_eval_ros, ROS2 inference/GT-replay — can share without re-implementing the same
``yaml.safe_load``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_conversion_config(dataset_root: str | Path, logger: Any = None) -> Dict[str, Any]:
    """Read ``<dataset_root>/conversion_config.yaml``. Empty dict if missing.

    ``logger`` may be any object with a ``.warning``/``.warn`` method (stdlib
    ``logging.Logger`` or a ROS node logger) — defaults to a module-level
    ``logging.Logger`` when not given.
    """
    cfg_path = Path(dataset_root) / "conversion_config.yaml"
    if not cfg_path.exists():
        msg = (
            f"[dataset_config] {cfg_path} not found — using schema defaults "
            "(data_space=joint, action_encoding=absolute, observation_encoding=quaternion)"
        )
        if logger is not None:
            (logger.warning if hasattr(logger, "warning") else logger.warn)(msg)
        else:
            import logging

            logging.getLogger(__name__).warning(msg)
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def resolve_data_space(cfg: Dict[str, Any]) -> str:
    """``cfg["data_space"]``, defaulting to ``"joint"`` (schema default)."""
    return cfg.get("data_space", "joint")


def resolve_action_encoding(cfg: Dict[str, Any]) -> str:
    """``cfg["action_encoding"]``, defaulting to ``"absolute"`` (schema default)."""
    return cfg.get("action_encoding", "absolute")


def resolve_observation_encoding(cfg: Dict[str, Any]) -> str:
    """``cfg["observation_encoding"]``, defaulting to ``"quaternion"`` (schema default)."""
    return cfg.get("observation_encoding", "quaternion")


def resolve_action_type(cfg: Dict[str, Any]) -> str:
    """Resolve an ``action_type`` string from a conversion_config's schema-v1.1 fields.

    joint -> ``joint_abs``; ee+absolute -> ``ee_abs``; ee+delta -> ``ee_delta``.
    ee+relative is reserved/unimplemented in mcap_converter, so unreachable here.
    """
    if resolve_data_space(cfg) != "ee":
        return "joint_abs"
    return "ee_delta" if resolve_action_encoding(cfg) == "delta" else "ee_abs"
