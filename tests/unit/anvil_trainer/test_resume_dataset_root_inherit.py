"""--resume auto-inherits dataset.root from the checkpoint's train_config.json.

Regression test for a bug where `--resume=<path>` without an explicit
`--dataset.root=` left TrainingConfig.dataset_root unset at the point
validate_action_space() runs, silently defaulting observation_encoding to
"quaternion" even for rot6d/axis_angle-encoded EE datasets.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from anvil_trainer.config import TrainingConfig


@contextlib.contextmanager
def _resume_argv(*extra_args: str):
    """sys.argv with only --resume set (no --dataset.root) — restores on exit."""
    saved_argv = sys.argv[:]
    saved_env = os.environ.copy()
    try:
        sys.argv = ["anvil-trainer"] + list(extra_args)
        for key in ("LEROBOT_EXCLUDE_OBSERVS", "LEROBOT_TASK_OVERRIDE"):
            os.environ.pop(key, None)
        yield
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)


def _make_checkpoint(job_dir: Path, checkpoint: str, dataset_root: str | None) -> None:
    pretrained = job_dir / "checkpoints" / checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    train_cfg: dict = {"policy": {"type": "diffusion"}}
    if dataset_root is not None:
        train_cfg["dataset"] = {"root": dataset_root}
    (pretrained / "train_config.json").write_text(json.dumps(train_cfg))


class TestResumeDatasetRootInherit:
    def test_inherits_dataset_root_from_checkpoint(self, tmp_path):
        job_dir = tmp_path / "my_job"
        _make_checkpoint(job_dir, "last", dataset_root="data/datasets/ee-delta/pbib-standard-env-merged")

        with _resume_argv(f"--resume={job_dir}"):
            cfg = TrainingConfig.from_env_and_args()
            assert any(
                a == "--dataset.root=data/datasets/ee-delta/pbib-standard-env-merged"
                for a in sys.argv
            )

        assert cfg.dataset_root == "data/datasets/ee-delta/pbib-standard-env-merged"

    def test_explicit_cli_dataset_root_wins(self, tmp_path):
        job_dir = tmp_path / "my_job"
        _make_checkpoint(job_dir, "last", dataset_root="data/datasets/ee-delta/pbib-standard-env-merged")

        with _resume_argv(f"--resume={job_dir}", "--dataset.root=data/datasets/other"):
            cfg = TrainingConfig.from_env_and_args()

        assert cfg.dataset_root == "data/datasets/other"

    def test_specific_checkpoint_step_resolved(self, tmp_path):
        job_dir = tmp_path / "my_job"
        _make_checkpoint(job_dir, "020000", dataset_root="data/datasets/ee-delta/foo")

        with _resume_argv(f"--resume={job_dir}/checkpoints/020000"):
            cfg = TrainingConfig.from_env_and_args()

        assert cfg.dataset_root == "data/datasets/ee-delta/foo"

    def test_missing_train_config_does_not_crash(self, tmp_path):
        job_dir = tmp_path / "my_job"
        (job_dir / "checkpoints" / "last").mkdir(parents=True)

        with _resume_argv(f"--resume={job_dir}"):
            cfg = TrainingConfig.from_env_and_args()

        assert cfg.dataset_root is None

    def test_train_config_without_dataset_key_does_not_crash(self, tmp_path):
        job_dir = tmp_path / "my_job"
        _make_checkpoint(job_dir, "last", dataset_root=None)

        with _resume_argv(f"--resume={job_dir}"):
            cfg = TrainingConfig.from_env_and_args()

        assert cfg.dataset_root is None

    def test_non_resume_run_unaffected(self, tmp_path):
        with _resume_argv("--dataset.root=data/datasets/plain"):
            cfg = TrainingConfig.from_env_and_args()

        assert cfg.dataset_root == "data/datasets/plain"
