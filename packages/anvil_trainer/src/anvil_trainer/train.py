#!/usr/bin/env python3
"""
LeRobot Training with Pluggable Customizations

This module is the thin entry point for the ``anvil-trainer`` CLI.  The
training machinery is split across sibling modules so each concern lives in
one place:

    anvil_trainer.config      — ``TrainingConfig`` + argv/env parsing + note resolution
    anvil_trainer.transforms  — ``Transform`` ABC and the three concrete transforms
    anvil_trainer.patches     — ``TransformRunner`` (monkey-patches for lerobot)
    anvil_trainer.train       — this file: ``train()`` entry + CLI help

Usage:
    # CLI
    anvil-trainer [lerobot args] [--use-delta-actions] [--task-description="..."] [--exclude-observation=images.chest,velocity]

    # Python
    from anvil_trainer import train, TrainingConfig
    config = TrainingConfig(exclude_observation=["images.chest"])
    train(config)

Environment variables:
    LEROBOT_EXCLUDE_OBSERVATION: Comma-separated observation suffixes to drop
    LEROBOT_TASK_OVERRIDE: Override task string for all samples
"""
from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from anvil_trainer.config import TrainingConfig, _resolve_note
from anvil_trainer.patches import TransformRunner, patched_lerobot

# Backward-compat re-exports: symbols that previously lived in this module.
# Existing tests and user code may import them from `anvil_trainer.train`.
from anvil_trainer.transforms import (  # noqa: E402, F401
    DeltaActionTransform,
    ExcludeObservationTransform,
    TaskOverrideTransform,
    Transform,
)

log = logging.getLogger(__name__)


# =============================================================================
# Note resolution helper
# =============================================================================


def _resolve_note(config: TrainingConfig) -> str | None:
    """
    Resolve the final note string for this run.

    During a new run (no --resume-job):
      - --note=TEXT        → use TEXT
      - --note-append=TEXT → treat as plain note (no old note to append to)
      - neither            → None

    During --resume-job:
      - neither            → auto-preserve: read old note from last checkpoint
      - --note=TEXT        → replace: discard old note, use TEXT
      - --note-append=TEXT → append: old note + "\\n[YYYY-MM-DD] TEXT"
    """
    if not config.resume_job_path:
        if config.note_append and not config.note:
            return config.note_append
        return config.note

    # Resume: read old note from last checkpoint's anvil_config.json
    old_note: str | None = None
    last_anvil = (
        Path(config.resume_job_path) / "checkpoints" / "last"
        / "pretrained_model" / "anvil_config.json"
    )
    if last_anvil.exists():
        try:
            data = json.loads(last_anvil.read_text())
            old_note = data.get("note") or None
        except Exception:
            pass

    if config.note is not None:
        return config.note  # explicit replace
    if config.note_append is not None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        if old_note:
            return f"{old_note}\n[{date_str}] {config.note_append}"
        return f"[{date_str}] {config.note_append}"
    return old_note  # auto-preserve


# =============================================================================
# Training Functions
# =============================================================================


def train(config: TrainingConfig | None = None) -> None:
    """
    Run LeRobot training with custom transforms.

    All monkey-patches are installed by :func:`patched_lerobot` and removed on
    exit (including on exception), so repeated calls do not compound patches
    and test runs don't pollute lerobot module state.

    Args:
        config: Training configuration. If None, parsed from env/args.
    """
    # Parse configuration if not provided
    if config is None:
        config = TrainingConfig.from_env_and_args()

    # Warn about unknown --exclude-observation keys
    config.warn_unknown_exclude_keys()

    # Resolve final note (auto-preserve / replace / append during resume)
    resolved_note = _resolve_note(config)
    config.note = resolved_note  # propagate to anvil_config.json writer
    if resolved_note:
        os.environ["WANDB_NOTES"] = resolved_note
        log.info("[anvil_trainer] Note: %s", resolved_note)

    # Resume path injection (requires patches-free access to sys.argv)
    if config.resume_job_path:
        # LeRobot 0.5.1 saves train_config.json inside each checkpoint
        last_cfg_path = Path(config.resume_job_path) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if last_cfg_path.exists() and not any(a.startswith("--config_path=") for a in sys.argv):
            sys.argv.append(f"--config_path={last_cfg_path}")
            log.info("[anvil_trainer] Resuming with config from last checkpoint: %s", last_cfg_path)

    # Install all lerobot patches; they are torn down on block exit even if
    # lerobot_train() raises.
    with patched_lerobot(config):
        from lerobot.scripts.lerobot_train import train as lerobot_train
        lerobot_train()


# =============================================================================
# CLI help text
# =============================================================================


_ANVIL_HELP = """\
anvil-trainer — LeRobot training with Anvil customizations
===========================================================

Examples:

  # Train ACT (basic)
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
    --policy.type=act --job_name=grabbing-w1

  # Train SmolVLA with task description
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
    --policy.type=smolvla --job_name=grabbing-w1 \\
    --task-description="Grab the gray doll and put it in the bucket"

  # Train with delta actions and camera subset
  anvil-trainer --dataset.root=data/datasets/my-dataset \\
    --policy.type=act --job_name=grabbing-w1 \\
    --camera-filter=chest,waist --use-delta-actions

  # Resume a stopped run
  anvil-trainer --resume-job=model_zoo/my-dataset/grabbing-w1

===============================================================================

Anvil-specific flags (stripped before passing to LeRobot):

  --use-delta-actions
      Convert actions to delta form (action - observation.state).
      Persisted to anvil_config.json in each checkpoint so inference
      can read it automatically.

  --resume-job=PATH
      Resume a previously stopped training job.
      Shortcut for --resume=true --output_dir=PATH.

  --task-description=TEXT
      Task prompt for SmolVLA. Overrides LEROBOT_TASK_OVERRIDE env var.
      Example: --task-description="Grab the gray doll and put it in the bucket"

  --camera-filter=CAM1,CAM2,...
      Train using only the specified cameras. Overrides LEROBOT_CAMERA_FILTER.
      Example: --camera-filter=chest,waist

  --note=TEXT
      Free-text note attached to this run.
      Stored in anvil_config.json in each checkpoint and sent to wandb as run notes.
      During --resume-job: replaces the previous note.
      Example: --note="lr=1e-4, wider backbone, retrain from scratch"

  --note-append=TEXT
      Append TEXT to the existing note when using --resume-job.
      Prefixes TEXT with the current date: "[YYYY-MM-DD] TEXT".
      On a new run (no --resume-job), treated as plain --note.
      Example: --note-append="switched to resnet34, bumped lr to 3e-4"

  --split-ratio=TRAIN,VAL,TEST
      Episode split ratios for train/val/test (default: 8,1,1).
      Two values also accepted: --split-ratio=8,2 means [8,2,0] (no test set).
      Val loss logged every log_freq*5 steps (eval/val_loss).
      Test loss logged every save_freq steps (eval/test_loss).
      Set to --split-ratio=1,0,0 to disable held-out sets.

  --job_name=NAME
      Human-readable run name. Checkpoints saved to model_zoo/<name>/.
      Auto-generated from policy type + timestamp if omitted.

  --output_dir=PATH
      Override the default checkpoint directory (default: model_zoo/).
      For resuming, use --resume-job instead.

Output:
  Checkpoints  →  model_zoo/<job_name>/checkpoints/<step>/pretrained_model/
  anvil_config →  model_zoo/<job_name>/checkpoints/<step>/pretrained_model/anvil_config.json

===============================================================================
LeRobot flags (passed through):
===============================================================================
"""


def _capture_lerobot_help() -> str:
    """Capture lerobot's help output without exiting."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    buf = io.StringIO()
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "--help"]
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            from lerobot.scripts.lerobot_train import train as lerobot_train
            lerobot_train()
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv
    return buf.getvalue()


def main() -> None:
    """CLI entry point for anvil-trainer."""
    import pydoc

    if "-h" in sys.argv or "--help" in sys.argv:
        lerobot_help = _capture_lerobot_help()
        pydoc.pager(_ANVIL_HELP + lerobot_help)
        sys.exit(0)
    train()


if __name__ == "__main__":
    main()
