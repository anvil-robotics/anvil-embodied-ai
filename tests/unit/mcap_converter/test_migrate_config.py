"""Tests for mcap_converter.cli.migrate_config — the dataset-config-migrate CLI.

Covers the four behaviors called out precisely in
claude_docs/mcap-converter-encoding-refactor-plan.md:
  - Gap 2: interactive yes/no confirmation before any file operation
  - Gap 3: reject-by-default on an existing version-tagged backup; --force overrides
  - Gap 4: already-current-version is a true no-op (zero file operations, not even a
    re-serialize with identical content — checked via mtime)
  - --force does NOT skip the confirmation prompt (only affects the backup-collision check)
"""
from __future__ import annotations

import time

import yaml

from mcap_converter.cli.migrate_config import main
from mcap_converter.config.versioning import CURRENT_SCHEMA_VERSION

_OLD_CONFIG = {
    "data_space": "ee",
    "observation_topics": {"right": "/ee_pose_right"},
    "action_topics": {},
    "ee_action_encoding": "delta",
    "camera_topics": ["/cam_waist/image_raw/compressed"],
    "camera_topic_mapping": {"/cam_waist/image_raw/compressed": "waist"},
    "image_resolution": [640, 480],
}


def _write_config(tmp_path, content: dict):
    cfg_path = tmp_path / "conversion_config.yaml"
    cfg_path.write_text(yaml.dump(content))
    return cfg_path


def test_confirm_no_aborts_with_no_file_operations(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, _OLD_CONFIG)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    exit_code = main(["--dataset", str(tmp_path)])

    assert exit_code == 1
    assert cfg_path.exists()
    assert not (tmp_path / "conversion_config_v1.0.yaml").exists()
    assert yaml.safe_load(cfg_path.read_text()) == _OLD_CONFIG  # byte-for-byte untouched


def test_confirm_yes_migrates_and_backs_up(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, _OLD_CONFIG)
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    exit_code = main(["--dataset", str(tmp_path)])

    assert exit_code == 0
    backup_path = tmp_path / "conversion_config_v1.0.yaml"
    assert backup_path.exists()
    assert yaml.safe_load(backup_path.read_text()) == _OLD_CONFIG  # original preserved verbatim

    upgraded = yaml.safe_load(cfg_path.read_text())
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    assert upgraded["action_encoding"] == "delta"
    assert "ee_action_encoding" not in upgraded


def test_already_current_is_true_noop(tmp_path, monkeypatch):
    current_config = dict(_OLD_CONFIG)
    current_config.pop("ee_action_encoding")
    current_config["schema_version"] = CURRENT_SCHEMA_VERSION
    current_config["action_encoding"] = "delta"
    current_config["observation_encoding"] = "quaternion"
    cfg_path = _write_config(tmp_path, current_config)

    before_mtime = cfg_path.stat().st_mtime_ns
    time.sleep(0.05)

    # Confirmation must never even be requested for a true no-op — fail loudly if it is.
    def _unexpected_input(_):
        raise AssertionError("input() must not be called for an already-current config")
    monkeypatch.setattr("builtins.input", _unexpected_input)

    exit_code = main(["--dataset", str(tmp_path)])

    assert exit_code == 0
    after_mtime = cfg_path.stat().st_mtime_ns
    assert after_mtime == before_mtime  # zero file operations — not even a re-serialize
    assert not (tmp_path / f"conversion_config_v{CURRENT_SCHEMA_VERSION}.yaml").exists()


def test_existing_backup_rejected_without_force(tmp_path, monkeypatch):
    _write_config(tmp_path, _OLD_CONFIG)
    (tmp_path / "conversion_config_v1.0.yaml").write_text("stale backup contents\n")

    def _unexpected_input(_):
        raise AssertionError("input() must not be called when refusing on backup collision")
    monkeypatch.setattr("builtins.input", _unexpected_input)

    exit_code = main(["--dataset", str(tmp_path)])

    assert exit_code == 1
    # Neither file touched.
    assert (tmp_path / "conversion_config_v1.0.yaml").read_text() == "stale backup contents\n"
    assert yaml.safe_load((tmp_path / "conversion_config.yaml").read_text()) == _OLD_CONFIG


def test_force_overwrites_existing_backup_but_still_prompts(tmp_path, monkeypatch):
    _write_config(tmp_path, _OLD_CONFIG)
    (tmp_path / "conversion_config_v1.0.yaml").write_text("stale backup contents\n")

    prompted = {"called": False}

    def _confirm_yes(_):
        prompted["called"] = True
        return "yes"
    monkeypatch.setattr("builtins.input", _confirm_yes)

    exit_code = main(["--dataset", str(tmp_path), "--force"])

    assert exit_code == 0
    assert prompted["called"] is True  # --force does NOT skip the confirmation prompt
    backup_content = yaml.safe_load((tmp_path / "conversion_config_v1.0.yaml").read_text())
    assert backup_content == _OLD_CONFIG  # old backup overwritten with the real old config


def test_force_without_confirmation_still_aborts(tmp_path, monkeypatch):
    """The clearest possible proof that --force does not bypass the yes/no prompt."""
    _write_config(tmp_path, _OLD_CONFIG)
    (tmp_path / "conversion_config_v1.0.yaml").write_text("stale backup contents\n")
    monkeypatch.setattr("builtins.input", lambda _: "no")

    exit_code = main(["--dataset", str(tmp_path), "--force"])

    assert exit_code == 1
    assert (tmp_path / "conversion_config_v1.0.yaml").read_text() == "stale backup contents\n"


def test_missing_dataset_config_errors(tmp_path):
    exit_code = main(["--dataset", str(tmp_path)])
    assert exit_code == 1
