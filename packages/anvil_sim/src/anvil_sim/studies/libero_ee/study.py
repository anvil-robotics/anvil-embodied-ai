"""The libero_ee study plugin: the action-representation study's dataset
groups, action-type registry, command builders, legality rules, and GT-replay
wiring, assembled into a :class:`~anvil_sim.study.Study` for the harness.

This is the single source of truth for everything study-specific the harness
delegates to; :func:`build_libero_ee_study` is registered under the name
``"libero_ee"`` in :mod:`anvil_sim.study`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anvil_sim.studies.libero_ee.eval_libero_ee import (
    _ALL_ACTION_TYPES,
    _ZERO_CAL_ACTION_TYPES,
)
from anvil_sim.studies.libero_ee.libero_convert import ALL_DATASET_GROUPS, LIBERO_ENV_SUITE
from anvil_sim.studies.libero_ee.math_validators import MATH_VALIDATORS
from anvil_sim.studies.libero_ee.replay_adapter import build_replay_adapter
from anvil_sim.study import GtReplayConfig, Study

if TYPE_CHECKING:
    from anvil_sim.bench_spec import BenchSpec

# The GT-replay baseline group and the native eval types it can take.
BASELINE_GROUP = "native"
NATIVE_EVAL_TYPES = ("native", "native_rot6d")

# deliver -> the env.control_mode it REQUIRES. Feeding a treatment through the
# wrong mode is exactly the class of mistake that produced Experiment 7's
# first 0% sweep; specs violating this fail at load time.
_REQUIRED_CONTROL_MODE = {"relative": "relative", "absolute": "absolute", "relative_converted": "relative"}

# lerobot/libero's global task_index -> libero.libero's own benchmark task_id
# (see libero_convert.py:LIBERO_ENV_TASK_ID for why these don't line up by a
# simple offset). Extend this as new tasks are added to the study.
_ENV_TASK_ID_BY_TASK_INDEX = {10: 8, 11: 9, 14: 2}

# Dataset groups whose action column is the Anvil-native (not Anvil-EE-writer)
# schema, so the generic mcap dataset-validate stage does not apply.
_DATASET_VALIDATE_SKIP = frozenset(
    {
        "native", "native_hand", "native_rot6d", "native_abs", "native_n0", "native_ctrlgoal",
        "afo_abs_h1", "afo_abs_h5", "afo_abs_h10",
    }
)


def _dataset_root(group: str, task_index: int) -> Path:
    # Directory suffixes are hyphenated (libero_convert.py convention:
    # delta_hand -> libero-taskN-delta-hand, native_rot6d -> ...-native-rot6d).
    suffix = group.replace("_", "-")
    return Path(f"data/datasets/ee-space/libero-task{task_index}-{suffix}")


def _convert_command(task_index: int, groups: list[str]) -> list[str]:
    return [
        "uv", "run", "--package", "anvil-sim", "anvil-libero-convert",
        f"--task-index={task_index}",
        f"--only={','.join(groups)}",
    ]


def _train_command(spec: BenchSpec, *, steps: int, batch_size: int, output_dir: Path) -> list[str]:
    if spec.train.trainer == "anvil-trainer":
        return [
            "uv", "run", "--package", "anvil-trainer", "anvil-trainer",
            f"--dataset.root={spec.dataset_root}",
            f"--policy.type={spec.train.policy_type}",
            f"--action-type={spec.train.action_type}",
            f"--output_dir={output_dir}",
            f"--job_name={spec.name}",
            f"--batch_size={batch_size}",
            f"--steps={steps}",
            "--policy.device=cuda",
            "--wandb.enable=false",
        ]
    return [  # lerobot-train (native family)
        "uv", "run", "--package", "anvil-sim", "lerobot-train",
        "--dataset.repo_id=local",
        f"--dataset.root={spec.dataset_root}",
        f"--policy.type={spec.train.policy_type}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={spec.name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--policy.device=cuda",
        "--wandb.enable=false",
    ]


def _eval_command(spec: BenchSpec, checkpoint: Path, output_dir: Path, n_episodes: int) -> list[str]:
    if spec.eval.action_type == "native":
        entry = ["uv", "run", "--package", "anvil-sim", "lerobot-eval"]
        extra: list[str] = []
    else:
        entry = ["uv", "run", "--package", "anvil-sim", "anvil-eval-libero"]
        extra = [f"--action-type={spec.eval.action_type}"]
    return [
        *entry, *extra,
        f"--policy.path={checkpoint}",
        "--env.type=libero",
        f"--env.task={spec.env_suite}",
        f"--env.task_ids=[{spec.env_task_id}]",
        f"--env.control_mode={spec.eval.control_mode}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={output_dir}",
        "--policy.device=cuda",
    ]


def _legality(spec: BenchSpec) -> list[str]:
    """Study-specific spec legality — dataset-group / eval-action-type
    membership and the deliver<->control_mode pairing (the Experiment 7
    lesson as a load-time error). Empty list == legal."""
    errors: list[str] = []

    if spec.dataset_group not in ALL_DATASET_GROUPS:
        errors.append(
            f"dataset_group {spec.dataset_group!r} not in {sorted(ALL_DATASET_GROUPS)}"
        )

    valid_eval = tuple(_ALL_ACTION_TYPES) + NATIVE_EVAL_TYPES
    if spec.eval.action_type not in valid_eval:
        errors.append(f"eval.action_type {spec.eval.action_type!r} not in {sorted(valid_eval)}")

    # INVARIANT: deliver mode and env.control_mode must be paired correctly —
    # feeding a treatment through the wrong control_mode is exactly the
    # mistake that produced Experiment 7's first 0% sweep (silent, no crash,
    # only visible as a training/eval result). See stage1-closeout.md and
    # research/libero_ee/ARCHITECTURE.md.
    if spec.eval.action_type in _ZERO_CAL_ACTION_TYPES:
        deliver = _ZERO_CAL_ACTION_TYPES[spec.eval.action_type][2]
        required = _REQUIRED_CONTROL_MODE[deliver]
        if spec.eval.control_mode != required:
            errors.append(
                f"eval.action_type {spec.eval.action_type!r} delivers via "
                f"{deliver!r} and REQUIRES env.control_mode={required!r} "
                f"(got {spec.eval.control_mode!r}) — see Experiment 7's "
                "post-mortem for what happens otherwise"
            )
    elif spec.eval.action_type in NATIVE_EVAL_TYPES and spec.eval.control_mode != "relative":
        errors.append("native/native_rot6d eval requires env.control_mode=relative")

    return errors


def _is_baseline(spec: BenchSpec) -> bool:
    return spec.eval.action_type in NATIVE_EVAL_TYPES and spec.dataset_group == "native"


# Ledger `status` classification — see research/libero_ee/stage1-closeout.md
# for the doubts each flag points at. Deliberately conservative: anything not
# listed here (native, native_hand, native_rot6d, and any future validated
# addition) surfaces as "—", i.e. no caveat attached.
_CONDITION_STATUS: dict[str, str] = {
    # Structurally locked to per_frame_anchor=True at convert time; a true
    # chunk-start (n-0) anchor was never actually tested. Not "provisional" —
    # definitively invalid as an n-0 test. See report.md §3.4 / diary.md.
    "native_n0": "⛔ invalid",
    # Absolute-vs-relative + goal-family + AFO/ctrlgoal results carry open
    # doubts: does absolute delivery generalize past this one reconstruction
    # formula, and is `relative_converted`'s reliance on robosuite's
    # `output_max` a LIBERO-only workaround? Both unresolved pending Stage 2.
    "native_abs": "⚠ provisional",
    "zerocal_goal_abs": "⚠ provisional",
    "zerocal_goal_world_n0": "⚠ provisional",
    "zerocal_goal_hand_n0": "⚠ provisional",
    "native_ctrlgoal": "⚠ provisional",
    "native_ctrlgoal_relconv": "⚠ provisional",
    "afo_abs_h1": "⚠ provisional",
    "afo_abs_h5": "⚠ provisional",
    "afo_abs_h10": "⚠ provisional",
    "afo_abs_rel_h1": "⚠ provisional",
    "afo_abs_rel_h5": "⚠ provisional",
    "afo_abs_rel_h10": "⚠ provisional",
    "afo_relative": "⚠ provisional",
}


def _condition_status(spec: BenchSpec) -> str:
    return _CONDITION_STATUS.get(spec.eval.action_type, "—")


def _dataset_validate_skip(spec: BenchSpec) -> bool:
    return spec.dataset_group in _DATASET_VALIDATE_SKIP


def _default_control_mode(action_type: str) -> str:
    """The env.control_mode a treatment REQUIRES, mirroring _legality()'s own
    mapping — zero-cal goal types derive it from their deliver mode; every
    other action type (native/native_rot6d/native_hand) always uses
    "relative"."""
    if action_type in _ZERO_CAL_ACTION_TYPES:
        deliver = _ZERO_CAL_ACTION_TYPES[action_type][2]
        return _REQUIRED_CONTROL_MODE[deliver]
    return "relative"


def _fill_defaults(spec: BenchSpec) -> None:
    """Fill study-specific fields a spec left unset, so specs need not repeat
    the same env_suite/env_task_id/control_mode on every file. Mutates
    ``spec`` in place; called by load_spec before validate()."""
    if spec.env_suite is None:
        spec.env_suite = LIBERO_ENV_SUITE
    if spec.env_task_id is None:
        if spec.task_index not in _ENV_TASK_ID_BY_TASK_INDEX:
            raise ValueError(
                f"env_task_id must be given explicitly for task_index={spec.task_index} "
                f"(no entry in _ENV_TASK_ID_BY_TASK_INDEX={sorted(_ENV_TASK_ID_BY_TASK_INDEX)})"
            )
        spec.env_task_id = _ENV_TASK_ID_BY_TASK_INDEX[spec.task_index]
    if spec.eval.control_mode is None:
        spec.eval.control_mode = _default_control_mode(spec.eval.action_type)


def build_libero_ee_study() -> Study:
    return Study(
        baseline_group=BASELINE_GROUP,
        eval_action_types=tuple(_ALL_ACTION_TYPES) + NATIVE_EVAL_TYPES,
        dataset_root=_dataset_root,
        math_validators=MATH_VALIDATORS,
        legality=_legality,
        convert_command=_convert_command,
        train_command=_train_command,
        eval_command=_eval_command,
        dataset_validate_skip=_dataset_validate_skip,
        fill_defaults=_fill_defaults,
        condition_status=_condition_status,
        gt_replay=GtReplayConfig(
            baseline_action_type="native",
            baseline_control_mode="relative",
            n_action_steps=1,
            is_baseline=_is_baseline,
        ),
        replay_adapter=build_replay_adapter(),
    )
