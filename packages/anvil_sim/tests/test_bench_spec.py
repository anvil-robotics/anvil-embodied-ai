"""Tests for the bench spec loader/validator — the treatment-legality rules
that turn experiments 1-8's hard-won lessons into load-time errors."""

from __future__ import annotations

import textwrap

import pytest

from anvil_sim.bench_spec import BenchSpec, EvalSpec, GateSpec, TrainSpec, load_spec


def _write_spec(tmp_path, body: str):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body))
    return p


_VALID_YAML = """
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
    eval:
      action_type: zerocal_goal_abs
      control_mode: relative
      n_episodes: 10
"""


def test_load_valid_spec(tmp_path):
    spec = load_spec(_write_spec(tmp_path, _VALID_YAML))
    assert spec.name == "task10-goal-abs-act"
    assert spec.dataset_root.name == "libero-task10-goalabs"
    assert spec.output_dir.as_posix() == "model_zoo/bench/task10-goal-abs-act"
    assert spec.checkpoint.as_posix().endswith("checkpoints/last/pretrained_model")
    assert spec.gates.gt_replay_margin == 15.0  # default


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_spec(_write_spec(tmp_path, _VALID_YAML + "\n    bogus_key: 1\n"))


def test_unknown_nested_key_rejected(tmp_path):
    bad = _VALID_YAML.replace("steps: 50000", "steps: 50000\n      bogus: true")
    with pytest.raises(ValueError, match="unknown key"):
        load_spec(_write_spec(tmp_path, bad))


def test_deliver_control_mode_pairing_enforced(tmp_path):
    """The Experiment 7 lesson as a load-time error: zerocal_goal_abs
    delivers via 'relative' and must NOT be run with control_mode=absolute."""
    bad = _VALID_YAML.replace("control_mode: relative", "control_mode: absolute")
    with pytest.raises(ValueError, match="REQUIRES env.control_mode='relative'"):
        load_spec(_write_spec(tmp_path, bad))


def test_absolute_deliver_types_require_absolute_mode(tmp_path):
    bad = _VALID_YAML.replace("zerocal_goal_abs", "zerocal_goal_world_seq").replace(
        "dataset_group: goalabs", "dataset_group: delta"
    )
    with pytest.raises(ValueError, match="REQUIRES env.control_mode='absolute'"):
        load_spec(_write_spec(tmp_path, bad))


def test_lerobot_train_must_not_set_action_type(tmp_path):
    bad = _VALID_YAML.replace("trainer: anvil-trainer", "trainer: lerobot-train")
    with pytest.raises(ValueError, match="must be omitted for lerobot-train"):
        load_spec(_write_spec(tmp_path, bad))


def test_anvil_trainer_requires_action_type(tmp_path):
    bad = _VALID_YAML.replace("      action_type: ee_abs\n", "")
    with pytest.raises(ValueError, match="action_type is required"):
        load_spec(_write_spec(tmp_path, bad))


def test_unknown_dataset_group_rejected(tmp_path):
    bad = _VALID_YAML.replace("dataset_group: goalabs", "dataset_group: bogus")
    with pytest.raises(ValueError, match="dataset_group"):
        load_spec(_write_spec(tmp_path, bad))


@pytest.mark.parametrize(
    ("group", "expected_dir"),
    [
        ("goalabs", "libero-task10-goalabs"),
        ("delta_hand", "libero-task10-delta-hand"),   # underscore group -> hyphen dir
        ("native_rot6d", "libero-task10-native-rot6d"),
    ],
)
def test_dataset_root_uses_hyphenated_dir_suffix(group, expected_dir):
    """Regression: group names use underscores but libero_convert writes
    hyphenated directory names — the mismatch made the convert stage try to
    re-create an existing dataset (and fail) for delta_hand/native_rot6d."""
    spec = BenchSpec(
        name="x", task_index=10, env_suite="libero_goal", env_task_id=8,
        dataset_group=group,
        train=TrainSpec(action_type="ee_abs"),
        eval=EvalSpec(action_type="ee_abs", control_mode="relative"),
    )
    assert spec.dataset_root.name == expected_dir


def test_reuse_checkpoint_overrides_checkpoint_path():
    spec = BenchSpec(
        name="x", task_index=10, env_suite="libero_goal", env_task_id=8,
        dataset_group="delta",
        train=TrainSpec(reuse_checkpoint="model_zoo/ee-space/foo/pretrained_model"),
        eval=EvalSpec(action_type="zerocal_goal_world_seq", control_mode="absolute"),
        gates=GateSpec(),
    )
    assert spec.checkpoint.as_posix() == "model_zoo/ee-space/foo/pretrained_model"


def test_output_dir_override():
    spec = BenchSpec(
        name="x", task_index=10, env_suite="libero_goal", env_task_id=8,
        dataset_group="abs",
        train=TrainSpec(action_type="ee_abs", output_dir="model_zoo/ee-space/libero-task10-abs/act"),
        eval=EvalSpec(action_type="ee_abs", control_mode="relative"),
    )
    assert spec.checkpoint.as_posix() == (
        "model_zoo/ee-space/libero-task10-abs/act/checkpoints/last/pretrained_model"
    )
