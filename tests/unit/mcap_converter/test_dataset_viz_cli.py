"""CLI-level tests for dataset-viz (Task 4): verify main()'s control flow --
which functions get called, in what order, and how early exits behave on
failure -- using mocks for anything that would touch Docker, git, or the
network.

Pure-function tests for the underlying viz.{dataset_check,nginx_config,
config,orchestrator} helpers live in test_dataset_viz.py; this file is
specifically about mcap_converter.cli.dataset_viz.main().
"""

import json
import socket
from pathlib import Path

import pytest

from mcap_converter.cli.dataset_viz import main


def _make_dataset(tmp_path: Path) -> Path:
    """Build a minimal valid synthetic LeRobot v3.0 dataset root under tmp_path."""
    root = tmp_path / "my-dataset"
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    (root / "data").mkdir()
    return root


class TestHelp:
    def test_help_exits_zero_and_mentions_key_flags(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

        out = capsys.readouterr().out
        for flag in (
            "--repo-id",
            "--episode",
            "--frontend-port",
            "--static-port",
            "--no-open",
            "--detach",
            "--stop",
            "--rebuild",
            "--refresh-source",
            "--cache-dir",
        ):
            assert flag in out


class TestRootRequiredness:
    def test_root_missing_without_stop_returns_1(self, capsys):
        rc = main([])
        assert rc == 1
        out = capsys.readouterr().out
        assert "ROOT" in out

    def test_stop_without_root_parses_fine(self, tmp_path):
        # No run.env in this empty cache dir -> should exit 0 without needing ROOT.
        rc = main(["--stop", "--cache-dir", str(tmp_path)])
        assert rc == 0


class TestStop:
    def test_stop_with_no_prior_run_is_graceful(self, tmp_path, capsys):
        rc = main(["--stop", "--cache-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "nothing to stop" in out.lower()

    def test_stop_with_existing_run_env_tears_down(self, tmp_path, monkeypatch):
        (tmp_path / "run.env").write_text("MCAP_VIZ_DATASET_ROOT=/tmp/x\n")

        calls = []

        class _FakeCompletedProcess:
            returncode = 0

        def fake_run(argv, *a, **kw):
            calls.append(argv)
            return _FakeCompletedProcess()

        monkeypatch.setattr("mcap_converter.cli.dataset_viz.subprocess.run", fake_run)

        rc = main(["--stop", "--cache-dir", str(tmp_path)])
        assert rc == 0
        assert len(calls) == 1
        assert calls[0][:2] == ["docker", "compose"]
        assert "down" in calls[0]


class TestDockerPreflight:
    def test_docker_unavailable_returns_1_and_short_circuits(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.check_docker_available",
            lambda: (False, "Docker is not installed or not on PATH."),
        )

        validate_calls = []
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.validate_dataset_root",
            lambda root: validate_calls.append(root),
        )

        dataset_root = _make_dataset(tmp_path)
        rc = main([str(dataset_root)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "Docker is not installed" in out
        assert validate_calls == []


class TestDatasetValidation:
    def test_invalid_dataset_root_returns_1_and_short_circuits(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.check_docker_available", lambda: (True, "")
        )

        ensure_calls = []
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.ensure_visualizer_source",
            lambda *a, **kw: ensure_calls.append((a, kw)),
        )

        bad_root = tmp_path / "not-a-dataset"
        bad_root.mkdir()
        rc = main([str(bad_root)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "info.json" in out or "does not exist" in out
        assert ensure_calls == []


class TestPortPreflight:
    def test_port_in_use_returns_1_and_names_port_and_flag(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.check_docker_available", lambda: (True, "")
        )

        dataset_root = _make_dataset(tmp_path)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        occupied_port = sock.getsockname()[1]

        try:
            ensure_calls = []
            monkeypatch.setattr(
                "mcap_converter.cli.dataset_viz.ensure_visualizer_source",
                lambda *a, **kw: ensure_calls.append((a, kw)),
            )

            rc = main([str(dataset_root), "--frontend-port", str(occupied_port)])

            assert rc == 1
            out = capsys.readouterr().out
            assert str(occupied_port) in out
            assert "--frontend-port" in out
            assert ensure_calls == []
        finally:
            sock.close()
