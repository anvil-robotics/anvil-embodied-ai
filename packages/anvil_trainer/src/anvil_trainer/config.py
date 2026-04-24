"""Training configuration + argv/env parsing + note resolution.

``TrainingConfig`` carries every anvil-specific training flag and hosts the
``from_env_and_args`` classmethod that pulls values out of ``sys.argv`` and
environment variables while removing them so lerobot's own CLI parser doesn't
reject unknown flags.

``_resolve_note`` implements the --note / --note-append semantics, including
the resume-time auto-preserve behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


# =============================================================================
# TrainingConfig
# =============================================================================


def _parse_resume_path(raw: str) -> tuple[str, str]:
    """Split a --resume path into (job_root, checkpoint_name).

    Examples::

        "model_zoo/foo/bar"                    -> ("model_zoo/foo/bar", "last")
        "model_zoo/foo/bar/checkpoints/020000" -> ("model_zoo/foo/bar", "020000")
    """
    p = Path(raw)
    parts = p.parts
    if "checkpoints" in parts:
        idx = list(parts).index("checkpoints")
        job_root = str(Path(*parts[:idx]))
        ckpt = parts[idx + 1] if idx + 1 < len(parts) else "last"
        return job_root, ckpt
    return raw, "last"


@dataclass
class TrainingConfig:
    """
    Configuration for custom training transformations.

    Attributes:
        exclude_observation: Observation suffixes to DROP (None = keep all).
            Use the key suffix after "observation." — supports both image and non-image keys:
            e.g. ["images.chest", "images.wrist_l", "velocity", "effort"]
        task_override: Override task string for all samples (for SmolVLA)
        use_delta_actions: Convert actions to delta (action - observation.state)
        dataset_root: Path to local dataset (for validation)
        note: Free-text note attached to this run (stored in anvil_config.json and wandb)
        note_append: Text to append to the existing note when resuming a run
    """

    exclude_observation: list[str] | None = None
    task_override: str | None = None
    use_delta_actions: bool = False
    delta_exclude_joints: list[str] | None = None  # Joint names to keep in absolute space when use_delta_actions=True
    delta_stats_n_steps: int = 8  # Number of look-ahead steps for delta stats (1 = single-frame, N = include k=0..N multi-step deltas)
    dataset_root: str | None = None
    output_dir: str | None = None
    resume_job_path: str | None = None   # Job root dir (before checkpoints/)
    resume_checkpoint: str = "last"       # Checkpoint to resume from ("last" or e.g. "020000")
    split_ratio: list[float] = field(default_factory=lambda: [8.0, 1.0, 1.0])  # train/val/test episode split ratios
    # Vision backbone for ACT/Diffusion: resnet18 | resnet34 | resnet50 (VLA models ignore this)
    backbone: str = "resnet18"
    note: str | None = None         # Free-text note for this run (also sent to wandb as run notes)
    note_append: str | None = None  # Append to existing note during --resume

    @classmethod
    def from_env_and_args(cls) -> TrainingConfig:
        """
        Parse configuration from environment variables and command line args.

        Environment variables:
            LEROBOT_EXCLUDE_OBSERVATION: Comma-separated observation suffixes to exclude
            LEROBOT_TASK_OVERRIDE: Task string override

        Command line args:
            --use-delta-actions: Enable delta action transform
        """
        # --exclude-observation=images.chest,images.wrist_l,velocity,effort
        excl_str = None
        for arg in sys.argv:
            if arg.startswith("--exclude-observation="):
                excl_str = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break
        if excl_str is None:
            excl_str = os.environ.get("LEROBOT_EXCLUDE_OBSERVATION", "")
        exclude_observation = [k.strip() for k in excl_str.split(",") if k.strip()] or None

        # Task description from --task-description arg (takes precedence over env var)
        task_override = None
        for arg in sys.argv:
            if arg.startswith("--task-description="):
                task_override = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break
        # Fall back to environment variable
        if task_override is None:
            task_override = os.environ.get("LEROBOT_TASK_OVERRIDE", "") or None

        # Delta actions from command line (remove to avoid lerobot arg parsing error)
        use_delta_actions = "--use-delta-actions" in sys.argv
        if use_delta_actions:
            sys.argv.remove("--use-delta-actions")

        # Joints to exclude from delta transform (kept in absolute space)
        delta_exclude_joints: list[str] | None = None
        for arg in sys.argv:
            if arg.startswith("--delta-exclude-joints="):
                raw = arg.split("=", 1)[1]
                delta_exclude_joints = [j.strip() for j in raw.split(",") if j.strip()]
                sys.argv.remove(arg)
                break

        # Number of look-ahead steps for delta stats computation
        delta_stats_n_steps = 8
        for arg in sys.argv:
            if arg.startswith("--delta-stats-n-steps="):
                delta_stats_n_steps = int(arg.split("=", 1)[1])
                sys.argv.remove(arg)
                break

        # Episode split ratios from --split-ratio arg (remove to avoid lerobot arg parsing error)
        split_ratio = [8.0, 1.0, 1.0]  # default: train/val/test
        for arg in sys.argv:
            if arg.startswith("--split-ratio="):
                parts = [float(x) for x in arg.split("=", 1)[1].split(",")]
                if len(parts) == 2:
                    parts.append(0.0)  # no test set
                split_ratio = parts
                sys.argv.remove(arg)
                break

        # Extract dataset_root and policy_type early (needed for naming and backbone injection)
        dataset_root = None
        for arg in sys.argv:
            if arg.startswith("--dataset.root="):
                dataset_root = arg.split("=", 1)[1]
                break
        dataset_name = Path(dataset_root).name if dataset_root else "dataset"

        policy_type = "run"
        for arg in sys.argv:
            if arg.startswith("--policy.type="):
                policy_type = arg.split("=", 1)[1]
                break

        # When --policy.path is given without --policy.type, read the type from
        # the checkpoint's config.json so the auto-generated job_name is meaningful.
        if policy_type == "run":
            for arg in sys.argv:
                if arg.startswith("--policy.path="):
                    _pp = Path(arg.split("=", 1)[1])
                    try:
                        _t = json.loads((_pp / "config.json").read_text()).get("type", "")
                        if not _t:  # fallback: parent train_config.json
                            _t = json.loads((_pp.parent / "train_config.json").read_text()).get("policy", {}).get("type", "")
                        if _t:
                            policy_type = _t
                    except Exception:
                        pass
                    break

        # --resume=PATH  (anvil flag — value is a path, not a boolean)
        # lerobot's own --resume=true/false is left in sys.argv untouched.
        resume_raw: str | None = None
        for arg in sys.argv:
            if arg.startswith("--resume="):
                val = arg.split("=", 1)[1]
                if val.lower() not in ("true", "false", "1", "0"):
                    resume_raw = val
                    sys.argv.remove(arg)
                    break

        resume_job_path: str | None = None
        resume_checkpoint: str = "last"
        if resume_raw is not None:
            resume_job_path, resume_checkpoint = _parse_resume_path(resume_raw)

        is_resume = resume_job_path is not None

        if is_resume:
            # If a specific checkpoint was requested, redirect 'last' symlink to it so
            # lerobot loads weights and step count from the correct checkpoint.
            if resume_checkpoint != "last":
                target_dir = Path(resume_job_path) / "checkpoints" / resume_checkpoint
                if not target_dir.exists():
                    raise FileNotFoundError(
                        f"[anvil_trainer] Checkpoint not found: {target_dir}"
                    )
                last_link = Path(resume_job_path) / "checkpoints" / "last"
                if last_link.is_symlink() or not last_link.exists():
                    last_link.unlink(missing_ok=True)
                    last_link.symlink_to(resume_checkpoint)
                    log.info("[anvil_trainer] Updated 'last' → '%s' for resume", resume_checkpoint)
                else:
                    log.warning(
                        "[anvil_trainer] 'last' is a real directory; cannot redirect to '%s'. "
                        "Lerobot will resume from the existing 'last' checkpoint.",
                        resume_checkpoint,
                    )

            # Inject lerobot resume flags
            if not any(a.startswith("--resume=") for a in sys.argv) and "--resume" not in sys.argv:
                sys.argv.append("--resume=true")
            if not any(a.startswith("--output_dir=") for a in sys.argv):
                sys.argv.append(f"--output_dir={resume_job_path}")

            output_dir = resume_job_path

            # Auto-inherit delta action settings from the target checkpoint if not explicitly set
            if not use_delta_actions:
                ckpt_anvil = (
                    Path(resume_job_path) / "checkpoints" / resume_checkpoint
                    / "pretrained_model" / "anvil_config.json"
                )
                if ckpt_anvil.exists():
                    try:
                        prev = json.loads(ckpt_anvil.read_text())
                        if prev.get("use_delta_actions"):
                            use_delta_actions = True
                            log.info("[anvil_trainer] --resume: inherited use_delta_actions=True from checkpoint")
                        if delta_exclude_joints is None and prev.get("delta_exclude_joints"):
                            delta_exclude_joints = prev["delta_exclude_joints"]
                            log.info(
                                "[anvil_trainer] --resume: inherited delta_exclude_joints=%s from checkpoint",
                                delta_exclude_joints,
                            )
                    except Exception:
                        pass
        else:
            # Resolve output_dir for NEW job: model_zoo/{dataset_name}/{run_name}
            # Extract job_name if provided (passed through to lerobot as-is)
            job_name = None
            for arg in sys.argv:
                if arg.startswith("--job_name="):
                    job_name = arg.split("=", 1)[1]
                    break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = job_name if job_name else f"{policy_type}_{timestamp}"

            output_dir = None
            for arg in sys.argv:
                if arg.startswith("--output_dir="):
                    output_dir = arg.split("=", 1)[1]
                    break
            if output_dir is None:
                output_dir = f"model_zoo/{dataset_name}/{run_name}"
                sys.argv.append(f"--output_dir={output_dir}")

            # Auto-inject job_name if not provided (used as wandb run name)
            if not job_name:
                sys.argv.append(f"--job_name={run_name}")

            # Set wandb project = dataset_name (not hardcoded "anvil")
            if not any(a.startswith("--wandb.project=") for a in sys.argv):
                sys.argv.append(f"--wandb.project={dataset_name}")

        # Backbone selection from --backbone= arg (for ACT/Diffusion; VLAs use their own vision)
        backbone = "resnet18"
        for arg in sys.argv:
            if arg.startswith("--backbone="):
                backbone = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # Free-text note for this run (stored in anvil_config.json + wandb)
        note: str | None = None
        for arg in sys.argv:
            if arg.startswith("--note="):
                note = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # Append text to existing note when resuming (--note-append=TEXT)
        note_append: str | None = None
        for arg in sys.argv:
            if arg.startswith("--note-append="):
                note_append = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break

        # Defaults injection — skip if resuming to avoid draccus decoding errors
        if not is_resume:
            # Default push_to_hub=false unless explicitly set
            if not any(arg.startswith("--policy.push_to_hub") for arg in sys.argv):
                sys.argv.append("--policy.push_to_hub=false")

            # Default dataset.repo_id=local for local dataset training
            if not any(arg.startswith("--dataset.repo_id") for arg in sys.argv):
                sys.argv.append("--dataset.repo_id=local")

            # Disable eval by default — no gym env available for Anvil datasets
            if not any(arg.startswith("--eval_freq") for arg in sys.argv):
                sys.argv.append("--eval_freq=0")

            # Default total training steps
            if not any(arg.startswith("--steps") for arg in sys.argv):
                sys.argv.append("--steps=100000")

            # Default checkpoint save frequency
            if not any(arg.startswith("--save_freq") for arg in sys.argv):
                sys.argv.append("--save_freq=10000")

            # If --policy.path is given (loading from checkpoint), lerobot rejects --policy.type.
            # Strip --policy.type from sys.argv; we've already captured the value for naming purposes.
            # Also skip backbone injection — the checkpoint already contains backbone config.
            has_policy_path = any(a.startswith("--policy.path=") for a in sys.argv)
            if has_policy_path:
                sys.argv = [a for a in sys.argv if not a.startswith("--policy.type=")]

            # Inject backbone settings for non-VLA policies (ACT, Diffusion).
            # Pi0.5 / SmolVLA use their own vision encoders and ignore these flags.
            _VLA_POLICIES = {"pi05", "smolvla", "pi0"}
            if policy_type not in _VLA_POLICIES and not has_policy_path:
                _BACKBONE_MAP = {
                    "resnet18": ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
                    "resnet34": ("resnet34", "ResNet34_Weights.IMAGENET1K_V1"),
                    "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V1"),
                }
                _vb, _pw = _BACKBONE_MAP.get(backbone, ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"))
                if not any(a.startswith("--policy.vision_backbone=") for a in sys.argv):
                    sys.argv.append(f"--policy.vision_backbone={_vb}")
                if not any(a.startswith("--policy.pretrained_backbone_weights=") for a in sys.argv):
                    sys.argv.append(f"--policy.pretrained_backbone_weights={_pw}")
                if policy_type == "diffusion":
                    if not any(a.startswith("--policy.use_group_norm=") for a in sys.argv):
                        sys.argv.append("--policy.use_group_norm=false")

        return cls(
            exclude_observation=exclude_observation,
            task_override=task_override,
            use_delta_actions=use_delta_actions,
            delta_exclude_joints=delta_exclude_joints,
            delta_stats_n_steps=delta_stats_n_steps,
            dataset_root=dataset_root,
            output_dir=output_dir,
            resume_job_path=resume_job_path,
            resume_checkpoint=resume_checkpoint,
            split_ratio=split_ratio,
            backbone=backbone,
            note=note,
            note_append=note_append,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TrainingConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(
            exclude_observation=data.get("exclude_observation"),
            task_override=data.get("task_override"),
            use_delta_actions=data.get("use_delta_actions", False),
            dataset_root=data.get("dataset_root"),
            split_ratio=data.get("split_ratio", [8.0, 1.0, 1.0]),
            backbone=data.get("backbone", "resnet18"),
        )

    def warn_unknown_exclude_keys(self) -> None:
        """Warn about --exclude-observation keys not present in the dataset features."""
        if not self.exclude_observation or not self.dataset_root:
            return

        info_path = Path(self.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[anvil_trainer] Cannot validate --exclude-observation: %s not found", info_path)
            return

        with open(info_path) as f:
            info = json.load(f)

        available = set(info.get("features", {}).keys())
        for suffix in self.exclude_observation:
            full_key = f"observation.{suffix}"
            if full_key not in available:
                log.warning("[anvil_trainer] --exclude-observation key not in dataset: %s", full_key)


# =============================================================================
# Note resolution
# =============================================================================


def _resolve_note(config: TrainingConfig) -> str | None:
    """
    Resolve the final note string for this run.

    During a new run (no --resume):
      - --note=TEXT        → use TEXT
      - --note-append=TEXT → treat as plain note (no old note to append to)
      - neither            → None

    During --resume:
      - neither            → auto-preserve: read old note from the target checkpoint
      - --note=TEXT        → replace: discard old note, use TEXT
      - --note-append=TEXT → append: old note + "\\n[YYYY-MM-DD] TEXT"
    """
    if not config.resume_job_path:
        if config.note_append and not config.note:
            return config.note_append
        return config.note

    # Resume: read old note from the target checkpoint's anvil_config.json
    old_note: str | None = None
    last_anvil = (
        Path(config.resume_job_path) / "checkpoints" / config.resume_checkpoint
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
