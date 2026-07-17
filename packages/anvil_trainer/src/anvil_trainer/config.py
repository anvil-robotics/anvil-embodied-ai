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

from anvil_shared.action_types import normalize_action_type

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


def _pop_float(flag: str, default: float) -> float:
    """Extract --flag=VALUE from sys.argv and return as float, or return *default*."""
    raw = _pop_argv(flag)
    return float(raw) if raw is not None else default


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


def _parse_names(info: dict, feat_key: str) -> list[str]:
    """Extract feature names from info.json for the given feature key.

    Handles both flat string lists and grouped dicts with ``motor_names``.
    """
    names = info.get("features", {}).get(feat_key, {}).get("names", [])
    if names and isinstance(names[0], dict):
        names = [n for group in names for n in group.get("motor_names", [])]
    return names


# "ee_rel" is a permanent legacy alias for "ee_relative" (normalized via
# normalize_action_type() — see anvil_shared.action_types). Both are accepted
# as valid input; only "ee_relative" is ever stored on TrainingConfig after
# construction (TrainingConfig.__post_init__ normalizes it).
#
# "ee_delta" is the Delta(n-(n-1)) mechanism: per-frame anchor, baked into the
# on-disk action column at mcap_converter time (action_encoding="delta") —
# NOT computed live like "ee_relative". No legacy alias; it's new.
_VALID_ACTION_TYPES = {"joint_abs", "ee_abs", "ee_relative", "ee_rel", "ee_delta"}

# Suffix components that appear in EE datasets but not joint datasets.
# Feature names may be bare ("qx") or arm-prefixed ("right_qx", "left_r0").
_EE_STATE_MARKER_SUFFIXES = {"qx", "qy", "qz", "qw"}
_EE_ACTION_MARKER_SUFFIXES = {"r0", "r1", "r2", "r3", "r4", "r5"}


def _has_ee_markers(names: list[str], markers: set[str]) -> bool:
    """Return True if any name is or ends with an EE marker suffix.

    Handles both bare names (``qx``) and arm-prefixed names (``right_qx``).
    """
    for name in names:
        # bare match: name == "qx"
        if name in markers:
            return True
        # suffix match: name == "right_qx" → last segment after "_" is "qx"
        _, _, suffix = name.rpartition("_")
        if suffix in markers:
            return True
    return False


@dataclass
class TrainingConfig:
    """
    Configuration for custom training transformations.

    Attributes:
        exclude_observs: Observation suffixes to DROP (None = keep all).
            Use the key suffix after "observation." — supports both image and non-image keys:
            e.g. ["images.chest", "images.wrist_l", "velocity", "effort"]
        task_override: Override task string for all samples (for SmolVLA)
        action_type: One of "joint_abs", "ee_abs", "ee_relative", "ee_delta"
            ("ee_rel" is a permanent legacy alias for "ee_relative", normalized
            in __post_init__ — see anvil_shared.action_types). "ee_delta" is
            the Delta(n-(n-1)) mechanism — its action is baked on disk by
            mcap_converter (action_encoding="delta"), not computed live.
        dataset_root: Path to local dataset (for validation)
        note: Free-text note attached to this run (stored in anvil_config.json and wandb)
        note_append: Text to append to the existing note when resuming a run
    """

    exclude_observs: list[str] | None = None
    task_override: str | None = None
    # action_type values:
    #   "joint_abs"    — joint absolute positions (default)
    #   "ee_abs"       — EE Cartesian rot6d, absolute
    #   "ee_relative"  — EE Cartesian rot6d, SE(3) relative to the chunk anchor
    #                    (body-frame xyz + relative rotation). Legacy alias:
    #                    "ee_rel" (normalized to "ee_relative" in __post_init__).
    #   "ee_delta"     — EE Cartesian rot6d, per-frame Delta(n-(n-1)) baked on
    #                    disk by mcap_converter (world-frame xyz + rotation,
    #                    anchored to the immediately-preceding real state, not
    #                    a fixed chunk anchor). No live transform of action.
    action_type: str = "joint_abs"
    dataset_root: str | None = None
    output_dir: str | None = None
    resume_job_path: str | None = None   # Job root dir (before checkpoints/)
    resume_checkpoint: str = "last"       # Checkpoint to resume from ("last" or e.g. "020000")
    split_ratio: list[float] = field(default_factory=lambda: [8.0, 1.0, 1.0])  # train/val/test episode split ratios
    max_episodes: int | None = None  # Randomly subsample N episodes before train/val/test split (None = use all)
    # Vision backbone for ACT/Diffusion: resnet18 | resnet34 | resnet50 (VLA models ignore this)
    backbone: str = "resnet18"
    note: str | None = None         # Free-text note for this run (also sent to wandb as run notes)
    note_append: str | None = None  # Append to existing note during --resume
    # EMA (Exponential Moving Average) — enabled by default for diffusion training.
    # Disable with --no-ema.  Hyperparameters match UMI production defaults.
    # pretrained_model/ will contain EMA weights (inference-ready, zero change to
    # inference_node); training_state/ will contain raw weights + counter for resume.
    use_ema: bool = True             # --no-ema disables
    ema_power: float = 0.75          # --ema-power=<float>   (UMI: 0.75)
    ema_max_value: float = 0.9999    # --ema-max-value=<float> (UMI: 0.9999)
    ema_inv_gamma: float = 1.0       # --ema-inv-gamma=<float> (UMI: 1.0)
    # DDPM-IP (Input Perturbation) — add small noise perturbation to the noise
    # during training to reduce exposure bias in closed-loop inference (UMI: 0.1).
    # Disable with --no-ddpm-ip.  Override alpha with --ddpm-ip-alpha=<float>.
    use_ddpm_ip: bool = True         # --no-ddpm-ip disables
    ddpm_ip_alpha: float = 0.1       # --ddpm-ip-alpha=<float> (UMI: 0.1)

    def __post_init__(self) -> None:
        # Normalize legacy action_type aliases (e.g. "ee_rel" → "ee_relative")
        # exactly once, here, regardless of how this config was constructed
        # (from_env_and_args, from_yaml, or direct instantiation). Everything
        # downstream (is_ee / is_ee_relative / is_ee_abs, transform selection,
        # anvil_config.json persistence) only ever sees the canonical value.
        self.action_type = normalize_action_type(self.action_type)

    @property
    def is_ee(self) -> bool:
        return self.action_type in ("ee_abs", "ee_relative", "ee_delta")

    @property
    def is_ee_relative(self) -> bool:
        return self.action_type == "ee_relative"

    @property
    def is_ee_abs(self) -> bool:
        return self.action_type == "ee_abs"

    @property
    def is_ee_delta(self) -> bool:
        return self.action_type == "ee_delta"

    @classmethod
    def from_env_and_args(cls) -> TrainingConfig:
        """
        Parse configuration from environment variables and command line args.

        Environment variables:
            LEROBOT_EXCLUDE_OBSERVS: Comma-separated observation suffixes to exclude
            LEROBOT_TASK_OVERRIDE: Task string override

        Command line args:
            --action-type=joint_abs|ee_abs|ee_relative|ee_delta  ("ee_rel" legacy alias also accepted)
            --exclude-observs=SUFFIX1,SUFFIX2: Drop observations by suffix
        """
        excl_str = _pop_argv("exclude-observs") or os.environ.get("LEROBOT_EXCLUDE_OBSERVS", "")
        exclude_observs = [k.strip() for k in excl_str.split(",") if k.strip()] or None

        task_override = _pop_argv("task-description") or os.environ.get("LEROBOT_TASK_OVERRIDE", "") or None

        action_type = _pop_argv("action-type") or "joint_abs"

        # Legacy flag: --use-delta-actions is no longer supported.
        if "--use-delta-actions" in sys.argv:
            sys.argv.remove("--use-delta-actions")
            raise ValueError(
                "--use-delta-actions is no longer supported. "
                "Use --action-type=joint_abs (joint absolute, the default) instead. "
                "For EE space training: --action-type=ee_abs, --action-type=ee_relative, "
                "or --action-type=ee_delta."
            )

        # Strip removed flags silently (they may appear in old scripts)
        for _legacy in ("delta-exclude-joints", "delta-stats-n-steps"):
            _pop_argv(_legacy)

        if action_type not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"--action-type={action_type!r} is not valid. "
                f"Choose from: {sorted(_VALID_ACTION_TYPES)}"
            )
        # Normalize once, early — everything below only ever sees "ee_relative".
        action_type = normalize_action_type(action_type)

        _sr_raw = _pop_argv("split-ratio")
        if _sr_raw:
            parts = [float(x) for x in _sr_raw.split(",")]
            split_ratio = parts + [0.0] if len(parts) == 2 else parts
        else:
            split_ratio = [8.0, 1.0, 1.0]

        _me_raw = _pop_argv("max-episodes")
        max_episodes: int | None = int(_me_raw) if _me_raw else None

        # peek (no remove) — needed for naming and backbone injection
        dataset_root = _pop_argv("dataset.root", remove=False)
        dataset_name = Path(dataset_root).name if dataset_root else "dataset"

        policy_type = _pop_argv("policy.type", remove=False) or "run"

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
        # Also handles --resume PATH (space-separated) for shell convenience.
        # lerobot's own --resume=true/false is left in sys.argv untouched.
        resume_raw: str | None = None
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--resume="):
                val = arg.split("=", 1)[1]
                if val.lower() not in ("true", "false", "1", "0"):
                    resume_raw = val
                    sys.argv.remove(arg)
                    break
            elif arg == "--resume" and i + 1 < len(sys.argv):
                val = sys.argv[i + 1]
                if val.lower() not in ("true", "false", "1", "0"):
                    resume_raw = val
                    sys.argv.pop(i + 1)
                    sys.argv.pop(i)
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

            # Auto-inherit action_type from checkpoint if not set on CLI.
            # Resume mechanism is preserved as-is; only the joint-delta legacy
            # mappings are removed here (use_delta_actions → delta_obs_t and
            # delta_exclude_joints inheritance no longer exist).
            if action_type == "joint_abs":
                ckpt_anvil = (
                    Path(resume_job_path) / "checkpoints" / resume_checkpoint
                    / "pretrained_model" / "anvil_config.json"
                )
                if ckpt_anvil.exists():
                    try:
                        prev = json.loads(ckpt_anvil.read_text())
                        _inherited = prev.get("action_type", "joint_abs")
                        if _inherited in _VALID_ACTION_TYPES and _inherited != "joint_abs":
                            # Normalize here too (checkpoint may persist the
                            # legacy "ee_rel" alias) — same single helper, no
                            # separate branch to drift out of sync.
                            action_type = normalize_action_type(_inherited)
                            log.info(
                                "[anvil_trainer] --resume: inherited action_type=%s from checkpoint",
                                action_type,
                            )
                    except Exception:
                        pass
        else:
            # Resolve output_dir for NEW job:
            #   model_zoo/{data_space}-space/{dataset_name}/{run_name}
            # ee_abs/ee_relative → ee-space/; joint_abs → joint-space/
            data_space = "ee" if action_type in ("ee_abs", "ee_relative") else "joint"

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
                output_dir = f"model_zoo/{data_space}-space/{dataset_name}/{run_name}"
                sys.argv.append(f"--output_dir={output_dir}")

            # Auto-inject job_name if not provided (used as wandb run name)
            if not job_name:
                sys.argv.append(f"--job_name={run_name}")

            # Set wandb project = dataset_name (not hardcoded "anvil")
            if not any(a.startswith("--wandb.project=") for a in sys.argv):
                sys.argv.append(f"--wandb.project={dataset_name}")

        backbone = _pop_argv("backbone") or "resnet18"

        # EMA flags (applicable to diffusion training; harmless for other policy types)
        use_ema = "--no-ema" not in sys.argv
        if not use_ema:
            sys.argv.remove("--no-ema")
        ema_power = _pop_float("ema-power", 0.75)
        ema_max_value = _pop_float("ema-max-value", 0.9999)
        ema_inv_gamma = _pop_float("ema-inv-gamma", 1.0)

        # DDPM-IP flags (input perturbation during training; UMI default alpha=0.1)
        use_ddpm_ip = "--no-ddpm-ip" not in sys.argv
        if not use_ddpm_ip:
            sys.argv.remove("--no-ddpm-ip")
        ddpm_ip_alpha = _pop_float("ddpm-ip-alpha", 0.1)

        note: str | None = _pop_argv("note")
        note_append: str | None = _pop_argv("note-append")

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
                    # UMI-aligned default scheduler: DDIM is deterministic and safe to skip
                    # denoise steps (train 50 → infer 16).  DDPM cannot safely skip steps
                    # due to its stochastic term.  Pass --policy.noise_scheduler_type=DDPM
                    # or --policy.num_train_timesteps=100 to opt out.
                    if not any(a.startswith("--policy.noise_scheduler_type=") for a in sys.argv):
                        sys.argv.append("--policy.noise_scheduler_type=DDIM")
                    if not any(a.startswith("--policy.num_train_timesteps=") for a in sys.argv):
                        sys.argv.append("--policy.num_train_timesteps=50")

        # Disable wandb artifact upload by default for all runs (new + resume)
        if not any(arg.startswith("--wandb.disable_artifact") for arg in sys.argv):
            sys.argv.append("--wandb.disable_artifact=true")

        return cls(
            exclude_observs=exclude_observs,
            task_override=task_override,
            action_type=action_type,
            dataset_root=dataset_root,
            output_dir=output_dir,
            resume_job_path=resume_job_path,
            resume_checkpoint=resume_checkpoint,
            split_ratio=split_ratio,
            max_episodes=max_episodes,
            backbone=backbone,
            note=note,
            note_append=note_append,
            use_ema=use_ema,
            ema_power=ema_power,
            ema_max_value=ema_max_value,
            ema_inv_gamma=ema_inv_gamma,
            use_ddpm_ip=use_ddpm_ip,
            ddpm_ip_alpha=ddpm_ip_alpha,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TrainingConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        _action_type = data.get("action_type", "joint_abs")
        return cls(
            exclude_observs=data.get("exclude_observs"),
            task_override=data.get("task_override"),
            action_type=_action_type,
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

    def validate_action_space(self) -> None:
        """Validate that the dataset's action space matches the chosen action_type.

        Reads ``meta/info.json`` from ``dataset_root`` (skipped when unset or
        info.json missing).  Raises ``DataIntegrityError`` on mismatch with a
        message suggesting the correct type.
        """
        if not self.dataset_root:
            return

        info_path = Path(self.dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            log.warning("[anvil_trainer] Cannot validate action_type: %s not found", info_path)
            return

        with open(info_path) as f:
            info = json.load(f)

        from anvil_trainer.transforms import DataIntegrityError

        state_names = _parse_names(info, "observation.state")
        action_names = _parse_names(info, "action")

        # Determine whether this is an EE dataset from feature names.
        has_ee_state  = _has_ee_markers(state_names,  _EE_STATE_MARKER_SUFFIXES)
        has_ee_action = _has_ee_markers(action_names, _EE_ACTION_MARKER_SUFFIXES)
        is_ee_dataset = has_ee_state and has_ee_action

        if self.action_type in ("ee_abs", "ee_relative"):
            if not is_ee_dataset:
                raise DataIntegrityError(
                    f"[validate_action_space] --action-type={self.action_type!r} requires an EE-space "
                    f"dataset, but the dataset at {self.dataset_root!r} appears to be joint-space.\n"
                    f"  observation.state names: {state_names}\n"
                    f"  action names:            {action_names}\n"
                    f"  Expected EE markers in state: {sorted(_EE_STATE_MARKER_SUFFIXES)}\n"
                    f"  Expected EE markers in action: {sorted(_EE_ACTION_MARKER_SUFFIXES)}\n"
                    "Hint: use --action-type=joint_abs for joint-space datasets."
                )
            # Also validate EE dimensions
            feat = info.get("features", {})
            state_shape = feat.get("observation.state", {}).get("shape", [])
            action_shape = feat.get("action", {}).get("shape", [])
            state_dim = state_shape[0] if state_shape else len(state_names)
            action_dim = action_shape[0] if action_shape else len(action_names)
            if state_dim % 8 != 0 or state_dim == 0:
                raise DataIntegrityError(
                    f"[validate_action_space] EE dataset has unexpected observation.state dim {state_dim} "
                    "(expected positive multiple of 8)."
                )
            n_arms = state_dim // 8
            if action_dim != 10 * n_arms:
                raise DataIntegrityError(
                    f"[validate_action_space] EE dataset action dim {action_dim} != "
                    f"10 * {n_arms} arms = {10 * n_arms}."
                )
            log.info(
                "[anvil_trainer] Validated action_type=%s with EE dataset (%d arm(s))",
                self.action_type, n_arms,
            )

        elif self.action_type == "joint_abs":
            if is_ee_dataset:
                raise DataIntegrityError(
                    f"[validate_action_space] --action-type=joint_abs but the dataset at "
                    f"{self.dataset_root!r} appears to be EE-space.\n"
                    f"  observation.state names: {state_names}\n"
                    f"  action names:            {action_names}\n"
                    "Hint: use --action-type=ee_abs or --action-type=ee_relative for EE datasets."
                )
            log.info("[anvil_trainer] Validated action_type=joint_abs with joint-space dataset.")


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
