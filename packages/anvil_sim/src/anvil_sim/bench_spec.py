"""Declarative experiment spec for the LIBERO validation harness.

One treatment = one YAML file under ``configs/libero_bench/`` (following the
repo's ``configs/<tool>/`` convention). The bench runner
(``anvil-sim-bench``, see ``bench_runner.py``) executes a spec through a
gated pipeline where every cheap check runs BEFORE any expensive training —
the codified lesson of experiments 1-8, where eval-path bugs invisible to
unit tests and training loss burned multiple full training sweeps.

Example::

    # configs/libero_bench/task10_goal_abs_act.yaml
    name: task10-goal-abs-act
    task_index: 10
    env_suite: libero_goal
    env_task_id: 8
    dataset_group: goalabs
    train:
      trainer: anvil-trainer
      action_type: ee_abs
      policy_type: act
      steps: 50000
      batch_size: 16
    eval:
      action_type: zerocal_goal_abs
      control_mode: relative
      n_episodes: 10
    gates:
      gt_replay_margin: 15.0

Validation encodes the treatment-legality lessons directly (e.g. the
``deliver``/``control_mode`` pairing that Experiment 7 got wrong is checked
against the eval action-type registry at load time, not discovered after a
training sweep).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anvil_sim.eval_libero_ee import _ALL_ACTION_TYPES, _ZERO_CAL_ACTION_TYPES
from anvil_sim.libero_convert import ALL_DATASET_GROUPS

log = logging.getLogger(__name__)

_VALID_TRAINERS = ("anvil-trainer", "lerobot-train")
_VALID_POLICY_TYPES = ("act", "diffusion")
_VALID_CONTROL_MODES = ("relative", "absolute")
# Eval action types replayable/evaluable, incl. the native family handled by
# lerobot's stock eval path rather than anvil-eval-libero.
_NATIVE_EVAL_TYPES = ("native", "native_rot6d")

# deliver -> the env.control_mode it REQUIRES. Feeding a treatment through
# the wrong mode is exactly the class of mistake that produced Experiment
# 7's first 0% sweep; specs violating this fail at load time.
_REQUIRED_CONTROL_MODE = {"relative": "relative", "absolute": "absolute"}


@dataclass
class TrainSpec:
    trainer: str = "anvil-trainer"
    action_type: str | None = None  # anvil-trainer --action-type; None for lerobot-train
    policy_type: str = "act"
    steps: int = 50000
    batch_size: int = 16
    # Reuse an existing checkpoint instead of training (e.g. world-seq reuses
    # the ee_delta checkpoint). Path to .../pretrained_model.
    reuse_checkpoint: str | None = None
    # Override the output dir (historical specs point at existing artifacts);
    # default: model_zoo/bench/{spec.name}
    output_dir: str | None = None


@dataclass
class EvalSpec:
    action_type: str = "ee_abs"
    control_mode: str = "relative"
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
    env_suite: str
    env_task_id: int
    dataset_group: str
    train: TrainSpec
    eval: EvalSpec
    gates: GateSpec = field(default_factory=GateSpec)
    source_path: str | None = None  # where the YAML was loaded from

    # ------------------------------------------------------------------ #
    # Derived paths (single source of truth for the runner)               #
    # ------------------------------------------------------------------ #
    @property
    def dataset_root(self) -> Path:
        return Path(f"data/datasets/ee-space/libero-task{self.task_index}-{self.dataset_group}")

    @property
    def output_dir(self) -> Path:
        return Path(self.train.output_dir or f"model_zoo/bench/{self.name}")

    @property
    def checkpoint(self) -> Path:
        if self.train.reuse_checkpoint:
            return Path(self.train.reuse_checkpoint)
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"

    @property
    def run_dir(self) -> Path:
        return Path(f"outputs/bench/runs/{self.name}")

    @property
    def eval_output_dir(self) -> Path:
        return Path(f"outputs/eval/bench-{self.name}")

    def validate(self) -> None:
        """Raise ValueError on any illegal combination. Encodes the
        treatment-legality lessons from experiments 1-8."""
        errors: list[str] = []

        if not self.name or "/" in self.name:
            errors.append(f"name must be a non-empty path-safe string, got {self.name!r}")
        if self.dataset_group not in ALL_DATASET_GROUPS:
            errors.append(
                f"dataset_group {self.dataset_group!r} not in {sorted(ALL_DATASET_GROUPS)}"
            )
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

        valid_eval = tuple(_ALL_ACTION_TYPES) + _NATIVE_EVAL_TYPES
        if self.eval.action_type not in valid_eval:
            errors.append(f"eval.action_type {self.eval.action_type!r} not in {sorted(valid_eval)}")

        # deliver <-> control_mode pairing (the Experiment 7 lesson).
        if self.eval.action_type in _ZERO_CAL_ACTION_TYPES:
            deliver = _ZERO_CAL_ACTION_TYPES[self.eval.action_type][2]
            required = _REQUIRED_CONTROL_MODE[deliver]
            if self.eval.control_mode != required:
                errors.append(
                    f"eval.action_type {self.eval.action_type!r} delivers via "
                    f"{deliver!r} and REQUIRES env.control_mode={required!r} "
                    f"(got {self.eval.control_mode!r}) — see Experiment 7's "
                    "post-mortem for what happens otherwise"
                )
        elif self.eval.action_type in _NATIVE_EVAL_TYPES and self.eval.control_mode != "relative":
            errors.append("native/native_rot6d eval requires env.control_mode=relative")

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
                   "train", "eval", "gates"}
    unknown = set(raw) - top_allowed
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")

    spec = BenchSpec(
        name=raw["name"],
        task_index=raw["task_index"],
        env_suite=raw["env_suite"],
        env_task_id=raw["env_task_id"],
        dataset_group=raw["dataset_group"],
        train=_take(raw.get("train", {}), TrainSpec, "train"),
        eval=_take(raw.get("eval", {}), EvalSpec, "eval"),
        gates=_take(raw.get("gates", {}), GateSpec, "gates"),
        source_path=str(path),
    )
    spec.validate()
    return spec
