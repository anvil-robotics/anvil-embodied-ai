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


def _pop_argv(flag: str, *, remove: bool = True) -> str | None:
    """Extract --flag=VALUE from sys.argv, return VALUE or None."""
    prefix = f"--{flag}="
    for arg in sys.argv:
        if arg.startswith(prefix):
            val = arg.split("=", 1)[1]
            if remove:
                sys.argv.remove(arg)
            return val
    return None


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


def _load_config_path_args() -> dict:
    """Read a LeRobot ``--config_path`` for Anvil's default resolution.

    LeRobot parses this file later.  Anvil still needs a few values early for
    output naming and for deciding which defaults to inject.  Previously those
    values were visible only when repeated on the command line, so a valid YAML
    recipe could be silently overridden with ``model_zoo/dataset/...`` defaults.
    """
    raw = _pop_argv("config_path", remove=False)
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        return {}
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            import yaml

            data = yaml.safe_load(path.read_text())
    except Exception as exc:
        log.warning("[anvil_trainer] Could not inspect --config_path=%s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _config_value(data: dict, *keys: str):
    """Return a nested config value, or ``None`` when any key is absent."""
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


@dataclass
class TrainingConfig:
    """
    Configuration for custom training transformations.

    Attributes:
        exclude_observs: Observation suffixes to DROP (None = keep all).
            Use the key suffix after "observation." — supports both image and non-image keys:
            e.g. ["images.chest", "images.wrist_l", "velocity", "effort"]
        task_override: Override task string for all samples (language-conditioned policies)
        use_delta_actions: Convert actions to delta (action - observation.state)
        dataset_root: Path to local dataset (for validation)
        note: Free-text note attached to this run (stored in anvil_config.json and wandb)
        note_append: Text to append to the existing note when resuming a run
    """

    exclude_observs: list[str] | None = None
    # Backward-compatible alias for the older --exclude-observation spelling.
    exclude_observation: list[str] | None = None
    task_override: str | None = None
    # action_type: "absolute" | "delta_obs_t" | "delta_sequential"
    # delta_obs_t:     delta[k] = action[t+k] - obs[t]  (all k share the same obs reference)
    # delta_sequential: delta[0] = action[t] - obs[t], delta[k] = action[t+k] - action[t+k-1]
    action_type: str = "absolute"
    delta_exclude_joints: list[str] | None = None  # Joint names to keep in absolute space when using delta actions
    delta_stats_n_steps: int = 1  # Look-ahead steps for delta stats (1 = single-frame, N = k=0..N-1 multi-step)
    dataset_root: str | None = None
    output_dir: str | None = None
    resume_job_path: str | None = None   # Job root dir (before checkpoints/)
    resume_checkpoint: str = "last"       # Checkpoint to resume from ("last" or e.g. "020000")
    split_ratio: list[float] = field(default_factory=lambda: [8.0, 1.0, 1.0])  # train/val/test episode split ratios
    max_episodes: int | None = None  # Randomly subsample N episodes before train/val/test split (None = use all)
    # Vision backbone for ACT/Diffusion: resnet18 | resnet34 | resnet50
    backbone: str = "resnet18"
    note: str | None = None         # Free-text note for this run (also sent to wandb as run notes)
    note_append: str | None = None  # Append to existing note during --resume

    def __post_init__(self) -> None:
        if self.exclude_observs is None and self.exclude_observation is not None:
            self.exclude_observs = self.exclude_observation
        self.exclude_observation = self.exclude_observs

    @property
    def use_delta_actions(self) -> bool:
        return self.action_type in ("delta_obs_t", "delta_sequential")

    @property
    def delta_sequential(self) -> bool:
        return self.action_type == "delta_sequential"

    @classmethod
    def from_env_and_args(cls) -> TrainingConfig:
        """
        Parse configuration from environment variables and command line args.

        Environment variables:
            LEROBOT_EXCLUDE_OBSERVS: Comma-separated observation suffixes to exclude
            LEROBOT_TASK_OVERRIDE: Task string override

        Command line args:
            --action-type=absolute|delta_obs_t|delta_sequential
            --use-delta-actions: Legacy flag, maps to --action-type=delta_obs_t
            --exclude-observs=SUFFIX1,SUFFIX2: Drop observations by suffix
        """
        file_config = _load_config_path_args()

        # Pop both spellings unconditionally so neither leaks through to
        # lerobot's parser (draccus rejects unknown args) if both are passed.
        excl_primary = _pop_argv("exclude-observs")
        excl_legacy = _pop_argv("exclude-observation")
        excl_str = (
            excl_primary
            or excl_legacy
            or os.environ.get("LEROBOT_EXCLUDE_OBSERVS")
            or os.environ.get("LEROBOT_EXCLUDE_OBSERVATION", "")
        )
        exclude_observs = [k.strip() for k in excl_str.split(",") if k.strip()] or None

        task_override = _pop_argv("task-description") or os.environ.get("LEROBOT_TASK_OVERRIDE", "") or None

        action_type = _pop_argv("action-type") or "absolute"
        # Backward compat: --use-delta-actions maps to delta_obs_t
        if "--use-delta-actions" in sys.argv:
            if action_type == "absolute":
                action_type = "delta_obs_t"
            sys.argv.remove("--use-delta-actions")
        _VALID_ACTION_TYPES = {"absolute", "delta_obs_t", "delta_sequential"}
        if action_type not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"--action-type={action_type!r} is not valid. "
                f"Choose from: {sorted(_VALID_ACTION_TYPES)}"
            )

        _dej_raw = _pop_argv("delta-exclude-joints")
        delta_exclude_joints: list[str] | None = (
            [j.strip() for j in _dej_raw.split(",") if j.strip()] if _dej_raw else None
        )

        _dsns_raw = _pop_argv("delta-stats-n-steps") or "1"
        try:
            delta_stats_n_steps = int(_dsns_raw)
        except ValueError:
            raise ValueError(
                f"--delta-stats-n-steps={_dsns_raw!r} is not a valid integer."
            ) from None

        _sr_raw = _pop_argv("split-ratio")
        if _sr_raw:
            parts = [float(x) for x in _sr_raw.split(",")]
            split_ratio = parts + [0.0] if len(parts) == 2 else parts
        else:
            split_ratio = [8.0, 1.0, 1.0]

        _me_raw = _pop_argv("max-episodes")
        max_episodes: int | None = int(_me_raw) if _me_raw else None

        # peek (no remove) — needed for naming and backbone injection
        dataset_root = (
            _pop_argv("dataset.root", remove=False)
            or _config_value(file_config, "dataset", "root")
        )
        dataset_name = Path(dataset_root).name if dataset_root else "dataset"

        policy_type = (
            _pop_argv("policy.type", remove=False)
            or _config_value(file_config, "policy", "type")
            or "run"
        )

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

            # Auto-inherit action_type and delta_exclude_joints from checkpoint if not set on CLI
            if action_type == "absolute":
                ckpt_anvil = (
                    Path(resume_job_path) / "checkpoints" / resume_checkpoint
                    / "pretrained_model" / "anvil_config.json"
                )
                if ckpt_anvil.exists():
                    try:
                        prev = json.loads(ckpt_anvil.read_text())
                        _inherited = prev.get("action_type", "absolute")
                        if _inherited == "absolute" and prev.get("use_delta_actions", False):
                            _inherited = "delta_obs_t"
                        action_type = _inherited
                        if action_type != "absolute":
                            log.info(
                                "[anvil_trainer] --resume: inherited action_type=%s from checkpoint", action_type,
                            )
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
            job_name = _config_value(file_config, "job_name")
            for arg in sys.argv:
                if arg.startswith("--job_name="):
                    job_name = arg.split("=", 1)[1]
                    break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = job_name if job_name else f"{policy_type}_{timestamp}"

            output_dir = _config_value(file_config, "output_dir")
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
            if (
                not any(a.startswith("--wandb.project=") for a in sys.argv)
                and _config_value(file_config, "wandb", "project") is None
            ):
                sys.argv.append(f"--wandb.project={dataset_name}")

        backbone = _pop_argv("backbone") or "resnet18"
        note: str | None = _pop_argv("note")
        note_append: str | None = _pop_argv("note-append")

        # Back-compat: LeRobot 0.6 renamed --eval_freq to --env_eval_freq.
        # Rewrite in-place (even on resume) — draccus rejects the old flag as unknown.
        for i, arg in enumerate(sys.argv):
            if arg == "--eval_freq" or arg.startswith("--eval_freq="):
                sys.argv[i] = "--env_eval_freq" + arg[len("--eval_freq"):]
                log.info("[anvil_trainer] Rewrote legacy %s to %s", arg, sys.argv[i])

        # Defaults injection — skip if resuming to avoid draccus decoding errors
        if not is_resume:
            # Default push_to_hub=false unless explicitly set
            if (
                not any(arg.startswith("--policy.push_to_hub") for arg in sys.argv)
                and _config_value(file_config, "policy", "push_to_hub") is None
            ):
                sys.argv.append("--policy.push_to_hub=false")

            # Default dataset.repo_id=local for local dataset training
            if (
                not any(arg.startswith("--dataset.repo_id") for arg in sys.argv)
                and _config_value(file_config, "dataset", "repo_id") is None
            ):
                sys.argv.append("--dataset.repo_id=local")

            # TorchCodec's CUDA 13 wheels require extra runtime library path setup on Jetson.
            # PyAV is already part of LeRobot's dataset extra and works for local Anvil datasets.
            if (
                not any(arg.startswith("--dataset.video_backend") for arg in sys.argv)
                and _config_value(file_config, "dataset", "video_backend") is None
            ):
                sys.argv.append("--dataset.video_backend=pyav")

            # Disable env eval by default — no gym env available for Anvil datasets.
            # (Legacy --eval_freq has already been rewritten to --env_eval_freq above.)
            if (
                not any(arg.startswith("--env_eval_freq") for arg in sys.argv)
                and _config_value(file_config, "env_eval_freq") is None
            ):
                sys.argv.append("--env_eval_freq=0")

            # Default total training steps
            if (
                not any(arg.startswith("--steps") for arg in sys.argv)
                and _config_value(file_config, "steps") is None
            ):
                sys.argv.append("--steps=100000")

            # Default checkpoint save frequency
            if (
                not any(arg.startswith("--save_freq") for arg in sys.argv)
                and _config_value(file_config, "save_freq") is None
            ):
                sys.argv.append("--save_freq=10000")

            # If --policy.path is given (loading from checkpoint), lerobot rejects --policy.type.
            # Strip --policy.type from sys.argv; we've already captured the value for naming purposes.
            # Also skip backbone injection — the checkpoint already contains backbone config.
            has_policy_path = (
                any(a.startswith("--policy.path=") for a in sys.argv)
                or _config_value(file_config, "policy", "path") is not None
            )
            if has_policy_path:
                sys.argv = [a for a in sys.argv if not a.startswith("--policy.type=")]

            # Inject backbone settings only for policies that use LeRobot's ResNet-style
            # vision backbone fields. Foundation policies define their own encoders and
            # should not receive --policy.vision_backbone overrides.
            _NO_BACKBONE_POLICIES = {
                "pi05",
                "smolvla",
                "pi0",
                "molmoact2",
                "groot",
                "multi_task_dit",
                "evo1",
                "fastwam",
                "vla_jepa",
            }
            if policy_type not in _NO_BACKBONE_POLICIES and not has_policy_path:
                _BACKBONE_MAP = {
                    "resnet18": ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
                    "resnet34": ("resnet34", "ResNet34_Weights.IMAGENET1K_V1"),
                    "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V1"),
                }
                if backbone not in _BACKBONE_MAP:
                    log.warning(
                        "[anvil_trainer] Unknown --backbone=%r; falling back to resnet18. "
                        "Valid choices: %s",
                        backbone,
                        sorted(_BACKBONE_MAP),
                    )
                _vb, _pw = _BACKBONE_MAP.get(backbone, ("resnet18", "ResNet18_Weights.IMAGENET1K_V1"))
                if not any(a.startswith("--policy.vision_backbone=") for a in sys.argv):
                    sys.argv.append(f"--policy.vision_backbone={_vb}")
                if not any(a.startswith("--policy.pretrained_backbone_weights=") for a in sys.argv):
                    sys.argv.append(f"--policy.pretrained_backbone_weights={_pw}")
                if policy_type == "diffusion":
                    if not any(a.startswith("--policy.use_group_norm=") for a in sys.argv):
                        sys.argv.append("--policy.use_group_norm=false")

        # Disable wandb artifact upload by default for all runs (new + resume)
        if (
            not any(arg.startswith("--wandb.disable_artifact") for arg in sys.argv)
            and _config_value(file_config, "wandb", "disable_artifact") is None
        ):
            sys.argv.append("--wandb.disable_artifact=true")

        return cls(
            exclude_observs=exclude_observs,
            task_override=task_override,
            action_type=action_type,
            delta_exclude_joints=delta_exclude_joints,
            delta_stats_n_steps=delta_stats_n_steps,
            dataset_root=dataset_root,
            output_dir=output_dir,
            resume_job_path=resume_job_path,
            resume_checkpoint=resume_checkpoint,
            split_ratio=split_ratio,
            max_episodes=max_episodes,
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

        _action_type = data.get("action_type", "absolute")
        if _action_type == "absolute" and data.get("use_delta_actions", False):
            _action_type = "delta_obs_t"
        return cls(
            exclude_observs=data.get("exclude_observs") or data.get("exclude_observation"),
            task_override=data.get("task_override"),
            action_type=_action_type,
            delta_exclude_joints=data.get("delta_exclude_joints"),
            dataset_root=data.get("dataset_root"),
            split_ratio=data.get("split_ratio", [8.0, 1.0, 1.0]),
            backbone=data.get("backbone", "resnet18"),
        )

    def warn_unknown_exclude_keys(self) -> None:
        """Warn about --exclude-observs keys not present in the dataset features."""
        if not self.exclude_observs or not self.dataset_root:
            return

        info_path = Path(self.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[anvil_trainer] Cannot validate --exclude-observs: %s not found", info_path)
            return

        with open(info_path) as f:
            info = json.load(f)

        available = set(info.get("features", {}).keys())
        for suffix in self.exclude_observs:
            full_key = f"observation.{suffix}"
            if full_key not in available:
                log.warning("[anvil_trainer] --exclude-observs key not in dataset: %s", full_key)


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
