"""Declarative experiment spec for the LIBERO validation harness.

One treatment = one YAML file under ``packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/`` (following the
repo's ``configs/<tool>/`` convention). The bench runner
(``anvil-sim-bench``, see ``bench_runner.py``) executes a spec through a
gated pipeline where every cheap check runs BEFORE any expensive training —
the codified lesson of experiments 1-8, where eval-path bugs invisible to
unit tests and training loss burned multiple full training sweeps.

Example (minimal — most fields are derivable and can be omitted, see below)::

    # packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_goal_abs_act.yaml
    name: task10-goal-abs-act
    task_index: 10
    dataset_group: goalabs
    train:
      action_type: ee_abs
      policy_type: act
    eval:
      action_type: zerocal_goal_abs
      n_episodes: 10

Several fields are derived when omitted, so a spec need only state what makes
it distinct:

- ``train.trainer``: ``"anvil-trainer"`` if ``train.action_type`` is set,
  else ``"lerobot-train"`` (:class:`TrainSpec.__post_init__`).
- ``train.steps``: 50000 for ``policy_type: act``, 30000 for ``diffusion``
  (:class:`TrainSpec.__post_init__`).
- ``train.batch_size``: 16 (the :class:`TrainSpec` field default).
- ``env_suite`` / ``env_task_id``: filled from ``task_index`` by the study's
  ``Study.fill_defaults`` (e.g. libero_ee's task_index -> LIBERO's own
  benchmark task_id mapping).
- ``eval.control_mode``: filled from ``eval.action_type`` by the same hook
  (the deliver<->control_mode mapping that Experiment 7 got wrong).
- ``gates.gt_replay_margin``: 15.0 (the :class:`GateSpec` field default).

Explicit values always override the derived ones. Validation encodes the
treatment-legality lessons directly (e.g. the ``deliver``/``control_mode``
pairing above is checked against the eval action-type registry at load
time, not discovered after a training sweep).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from anvil_sim.study import Study

log = logging.getLogger(__name__)

_VALID_TRAINERS = ("anvil-trainer", "lerobot-train")
_VALID_POLICY_TYPES = ("act", "diffusion")
_VALID_CONTROL_MODES = ("relative", "absolute")

# Every artifact for one research topic lives under ``research/<study>/`` (raw
# per-experiment results + the write-ups), keyed by the study/topic name so the
# layout generalizes to any study. Model checkpoints (large binaries) mirror the
# scheme under ``model_zoo/research/<study>/``.
RESEARCH_ROOT = Path("research")


def topic_root(study_name: str) -> Path:
    """Root dir for one research topic's results: ``research/<study_name>/``."""
    return RESEARCH_ROOT / study_name


@dataclass
class TrainSpec:
    # Both left None by default and derived in __post_init__ — the presence
    # of action_type determines the trainer, and policy_type determines the
    # step count, so a spec need not repeat either explicitly:
    #   trainer:  "anvil-trainer" if action_type else "lerobot-train"
    #   steps:    50000 (act) / 30000 (diffusion)
    trainer: str | None = None
    action_type: str | None = None  # anvil-trainer --action-type; None for lerobot-train
    policy_type: str = "act"
    steps: int | None = None
    batch_size: int = 16
    # Reuse an existing checkpoint instead of training. Path to
    # .../pretrained_model.
    reuse_checkpoint: str | None = None
    # Override the output dir (historical specs point at existing artifacts);
    # default: model_zoo/research/<study>/{spec.name}
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if self.trainer is None:
            self.trainer = "anvil-trainer" if self.action_type else "lerobot-train"
        if self.steps is None:
            self.steps = 50000 if self.policy_type == "act" else 30000


@dataclass
class EvalSpec:
    action_type: str = "ee_abs"
    # Left None by default and derived by the study's Study.fill_defaults
    # (from the deliver<->control_mode mapping load_spec already validates
    # explicit values against) — see study.py's _legality/_fill_defaults.
    control_mode: str | None = None
    n_episodes: int = 10


@dataclass
class GateSpec:
    # Treatment GT-replay success may be at most this many pc points below
    # the native replay baseline on the same episodes/seeds.
    gt_replay_margin: float = 15.0
    # Stage names to skip (use sparingly; recorded in the ledger).
    skip: list[str] = field(default_factory=list)


@dataclass
class BenchSpec:
    name: str
    task_index: int
    dataset_group: str
    train: TrainSpec
    eval: EvalSpec
    # Left None by default and derived by the study's Study.fill_defaults
    # (e.g. libero_ee fills env_suite="libero_goal" and env_task_id from
    # task_index) — see study.py's _fill_defaults. Guaranteed non-None by
    # the time validate() runs.
    env_suite: str | None = None
    env_task_id: int | None = None
    gates: GateSpec = field(default_factory=GateSpec)
    source_path: str | None = None  # where the YAML was loaded from
    # Which registered study this spec belongs to (YAML key ``study``). Selects
    # the plugin that owns dataset groups, action types, commands, math
    # identities, and the GT-replay wiring — see :mod:`anvil_sim.study`.
    study_name: str = "libero_ee"

    # ------------------------------------------------------------------ #
    # Derived paths (single source of truth for the runner)               #
    # ------------------------------------------------------------------ #
    @cached_property
    def study(self) -> Study:
        """The resolved study plugin (cached per spec)."""
        from anvil_sim.study import get_study

        return get_study(self.study_name)

    @property
    def dataset_root(self) -> Path:
        return self.study.dataset_root(self.dataset_group, self.task_index)

    @property
    def output_dir(self) -> Path:
        return Path(self.train.output_dir or f"model_zoo/research/{self.study_name}/{self.name}")

    @property
    def checkpoint(self) -> Path:
        if self.train.reuse_checkpoint:
            return Path(self.train.reuse_checkpoint)
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"

    @property
    def run_dir(self) -> Path:
        return topic_root(self.study_name) / "experiments" / self.name

    @property
    def eval_output_dir(self) -> Path:
        return self.run_dir / "eval"

    def validate(self) -> None:
        """Raise ValueError on any illegal combination. Encodes the
        treatment-legality lessons from experiments 1-8."""
        errors: list[str] = []

        if not self.name or "/" in self.name:
            errors.append(f"name must be a non-empty path-safe string, got {self.name!r}")
        if self.train.trainer not in _VALID_TRAINERS:
            errors.append(f"train.trainer must be one of {_VALID_TRAINERS}")
        if self.train.policy_type not in _VALID_POLICY_TYPES:
            errors.append(f"train.policy_type must be one of {_VALID_POLICY_TYPES}")
        if self.eval.control_mode not in _VALID_CONTROL_MODES:
            errors.append(f"eval.control_mode must be one of {_VALID_CONTROL_MODES}")

        if self.train.trainer == "anvil-trainer" and not self.train.action_type:
            errors.append("train.action_type is required for anvil-trainer")
        if self.train.trainer == "lerobot-train" and self.train.action_type:
            errors.append("train.action_type must be omitted for lerobot-train")

        # Study-specific legality: dataset groups, eval action-type registry,
        # and the deliver<->control_mode pairing (the Experiment 7 lesson).
        errors.extend(self.study.legality(self))

        if errors:
            raise ValueError(
                f"Invalid bench spec {self.source_path or self.name!r}:\n  - "
                + "\n  - ".join(errors)
            )


def load_spec(path: Path | str) -> BenchSpec:
    """Load and validate a spec YAML. Raises on unknown keys so typos fail
    loudly instead of silently falling back to defaults."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")

    def _take(d: dict, cls, section: str):
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)} in '{section}'")
        return cls(**d)

    top_allowed = {"name", "task_index", "env_suite", "env_task_id", "dataset_group",
                   "train", "eval", "gates", "study"}
    unknown = set(raw) - top_allowed
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")

    spec = BenchSpec(
        name=raw["name"],
        task_index=raw["task_index"],
        env_suite=raw.get("env_suite"),
        env_task_id=raw.get("env_task_id"),
        dataset_group=raw["dataset_group"],
        train=_take(raw.get("train", {}), TrainSpec, "train"),
        eval=_take(raw.get("eval", {}), EvalSpec, "eval"),
        gates=_take(raw.get("gates", {}), GateSpec, "gates"),
        source_path=str(path),
        study_name=raw.get("study", "libero_ee"),
    )
    spec.study.fill_defaults(spec)
    spec.validate()
    return spec
