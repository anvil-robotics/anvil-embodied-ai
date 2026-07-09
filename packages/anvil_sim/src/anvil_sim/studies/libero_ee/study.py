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
from anvil_sim.studies.libero_ee.libero_convert import ALL_DATASET_GROUPS
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
_REQUIRED_CONTROL_MODE = {"relative": "relative", "absolute": "absolute"}

# Dataset groups whose action column is the Anvil-native (not Anvil-EE-writer)
# schema, so the generic mcap dataset-validate stage does not apply.
_DATASET_VALIDATE_SKIP = frozenset(
    {"native", "native_hand", "native_rot6d", "native_abs", "native_n0", "goalabs_aa"}
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
    elif spec.eval.action_type == "native_rot6d":
        entry = ["uv", "run", "--package", "anvil-sim", "anvil-eval-native-rot6d"]
        extra = []
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


def _dataset_validate_skip(spec: BenchSpec) -> bool:
    return spec.dataset_group in _DATASET_VALIDATE_SKIP


def build_libero_ee_study() -> Study:
    return Study(
        name="libero_ee",
        dataset_groups=ALL_DATASET_GROUPS,
        baseline_group=BASELINE_GROUP,
        eval_action_types=tuple(_ALL_ACTION_TYPES) + NATIVE_EVAL_TYPES,
        dataset_root=_dataset_root,
        math_validators=MATH_VALIDATORS,
        legality=_legality,
        convert_command=_convert_command,
        train_command=_train_command,
        eval_command=_eval_command,
        dataset_validate_skip=_dataset_validate_skip,
        gt_replay=GtReplayConfig(
            baseline_action_type="native",
            baseline_control_mode="relative",
            n_action_steps=1,
            is_baseline=_is_baseline,
        ),
        replay_adapter=build_replay_adapter(),
    )
