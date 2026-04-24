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


# _resolve_note is defined in anvil_trainer.config and imported above.


# =============================================================================
# W&B fork helpers
# =============================================================================


def _find_wandb_run_id(job_path: Path) -> tuple[str | None, str | None, str | None]:
    """Return (run_id, entity, project) from the local wandb directory, or (None, None, None)."""
    latest = job_path / "wandb" / "latest-run"
    if not latest.exists():
        return None, None, None
    try:
        resolved = latest.resolve()
        # Directory name format: run-YYYYMMDD_HHMMSS-<run_id>
        run_id = resolved.name.rsplit("-", 1)[-1]
        entity: str | None = None
        project: str | None = None
        meta = resolved / "files" / "wandb-metadata.json"
        if meta.exists():
            data = json.loads(meta.read_text())
            entity = data.get("entity")
            project = data.get("project")
        return run_id, entity, project
    except Exception:
        return None, None, None


def _delete_wandb_run(old_run_id: str, new_run: object) -> None:
    """Delete the old W&B run via the API after a successful forked resume."""
    try:
        import wandb
        entity = getattr(new_run, "entity", None)
        project = getattr(new_run, "project", None)
        if not entity or not project:
            log.warning("[anvil_trainer] W&B fork: cannot delete old run — entity/project unknown")
            return
        api = wandb.Api()
        api.run(f"{entity}/{project}/{old_run_id}").delete()
        log.info("[anvil_trainer] W&B fork: deleted old run %s/%s/%s", entity, project, old_run_id)
    except Exception as exc:
        log.warning("[anvil_trainer] W&B fork: failed to delete old run '%s': %s", old_run_id, exc)


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
        # LeRobot 0.5.1 saves train_config.json inside each checkpoint.
        # Use the specific checkpoint (e.g. "020000") if given, otherwise "last".
        ckpt_cfg_path = (
            Path(config.resume_job_path) / "checkpoints"
            / config.resume_checkpoint / "pretrained_model" / "train_config.json"
        )
        if ckpt_cfg_path.exists() and not any(a.startswith("--config_path=") for a in sys.argv):
            sys.argv.append(f"--config_path={ckpt_cfg_path}")
            log.info("[anvil_trainer] Resuming with config from checkpoint '%s': %s", config.resume_checkpoint, ckpt_cfg_path)

    # Install all lerobot patches; they are torn down on block exit even if
    # lerobot_train() raises.
    with patched_lerobot(config):
        from lerobot.scripts.lerobot_train import train as lerobot_train

        # Patch init_logging (in lerobot_train's namespace) so our resume summary
        # is emitted right after lerobot sets up its log format — and therefore
        # appears in the same format, just before the "Output dir:" line.
        if config.resume_job_path:
            import lerobot.scripts.lerobot_train as _lt
            _orig_init_logging = _lt.init_logging

            def _resume_init_logging(*args, **kwargs):
                import lerobot.scripts.lerobot_train as _lt_inner
                _lt_inner.init_logging = _orig_init_logging  # self-restore
                _orig_init_logging(*args, **kwargs)
                ckpt = config.resume_checkpoint
                if ckpt == "last":
                    last_link = Path(config.resume_job_path) / "checkpoints" / "last"
                    if last_link.is_symlink():
                        ckpt = f"last → {last_link.resolve().name}"
                import logging as _logging
                _logging.info(
                    "[anvil_trainer] Resuming: %s  (checkpoint: %s)",
                    config.resume_job_path,
                    ckpt,
                )

            _lt.init_logging = _resume_init_logging

        # W&B fork: when resuming from a specific (non-last) checkpoint with W&B enabled,
        # fork the old run at the target step instead of resuming it inline.
        # This creates a clean new run whose history starts at fork_step, with no
        # stale data from the original run's later steps.
        # After lerobot_train() completes successfully, the old run is deleted.
        _old_wandb_run_id: str | None = None
        _fork_new_run: object = None
        _wandb_fork_active = (
            config.resume_job_path is not None
            and config.resume_checkpoint != "last"
            and any(a == "--wandb.enable=true" for a in sys.argv)
        )

        if _wandb_fork_active:
            fork_step = int(config.resume_checkpoint)
            _old_wandb_run_id, _ent, _proj = _find_wandb_run_id(Path(config.resume_job_path))
            if _old_wandb_run_id:
                fork_from = (
                    f"{_ent}/{_proj}/{_old_wandb_run_id}?_step={fork_step}"
                    if _ent and _proj
                    else f"{_old_wandb_run_id}?_step={fork_step}"
                )
                import wandb as _wandb
                _orig_wandb_init = _wandb.init

                def _wandb_fork_init(*args, **kwargs):
                    nonlocal _fork_new_run
                    import wandb as _w
                    _w.init = _orig_wandb_init  # self-restore before calling
                    kwargs.pop("resume", None)
                    kwargs.pop("id", None)
                    kwargs["fork_from"] = fork_from
                    _fork_new_run = _orig_wandb_init(*args, **kwargs)
                    return _fork_new_run

                _wandb.init = _wandb_fork_init
                log.info(
                    "[anvil_trainer] W&B fork: will fork run '%s' at step %d",
                    _old_wandb_run_id, fork_step,
                )
            else:
                log.warning(
                    "[anvil_trainer] W&B fork: no previous run ID found in %s/wandb/ — "
                    "falling back to normal resume",
                    config.resume_job_path,
                )

        lerobot_train()

        # Reached only on successful completion — delete the now-superseded old run
        if _old_wandb_run_id and _fork_new_run is not None:
            _delete_wandb_run(_old_wandb_run_id, _fork_new_run)


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

  # Resume a stopped run (from latest checkpoint)
  anvil-trainer --resume=model_zoo/my-dataset/grabbing-w1

  # Resume from a specific checkpoint (e.g. step 20000)
  anvil-trainer --resume=model_zoo/my-dataset/grabbing-w1/checkpoints/020000

===============================================================================

Anvil-specific flags (stripped before passing to LeRobot):

  --use-delta-actions
      Convert actions to delta form (action - observation.state).
      Persisted to anvil_config.json in each checkpoint so inference
      can read it automatically.

  --resume=PATH
      Resume a previously stopped training job.
      PATH can be the job root dir or a specific checkpoint dir:
        --resume=model_zoo/my-dataset/grabbing-w1              (resume from last)
        --resume=model_zoo/my-dataset/grabbing-w1/checkpoints/020000  (resume from step 20000)
      When a specific checkpoint is given, the 'last' symlink is updated to point
      to it so lerobot loads weights and the step counter from that checkpoint.

  --task-description=TEXT
      Task prompt for SmolVLA. Overrides LEROBOT_TASK_OVERRIDE env var.
      Example: --task-description="Grab the gray doll and put it in the bucket"

  --camera-filter=CAM1,CAM2,...
      Train using only the specified cameras. Overrides LEROBOT_CAMERA_FILTER.
      Example: --camera-filter=chest,waist

  --note=TEXT
      Free-text note attached to this run.
      Stored in anvil_config.json in each checkpoint and sent to wandb as run notes.
      During --resume: replaces the previous note.
      Example: --note="lr=1e-4, wider backbone, retrain from scratch"

  --note-append=TEXT
      Append TEXT to the existing note when using --resume.
      Prefixes TEXT with the current date: "[YYYY-MM-DD] TEXT".
      On a new run (no --resume), treated as plain --note.
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
      For resuming, use --resume instead.

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
