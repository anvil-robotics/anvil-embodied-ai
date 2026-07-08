"""Tests for mcap-convert's --quality-report / --include-flagged integration."""

import json
import os
from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "smoke" / "fixtures" / "test-session"


def _write_report(tmp_path, episodes):
    """episodes: list of (path_str, severity) tuples."""
    payload = {
        "episodes": [
            {"path": path, "duration_s": 1.0, "severity": severity, "passed": severity != "critical", "topics": []}
            for path, severity in episodes
        ]
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload))
    return report_path


class TestResolveQualitySkipSet:
    def test_include_flagged_pass_skips_warning_and_critical(self, tmp_path):
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning"), ("/c.mcap", "pass")])

        skip_set = resolve_quality_skip_paths(str(report), include_flagged="pass")

        assert set(skip_set.keys()) == {"/a.mcap", "/b.mcap"}
        assert skip_set["/a.mcap"] == "critical"
        assert skip_set["/b.mcap"] == "warning"

    def test_include_flagged_warning_skips_only_critical(self, tmp_path):
        """'warning' is the new CLI default — only critical episodes are skipped."""
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning"), ("/c.mcap", "pass")])

        skip_set = resolve_quality_skip_paths(str(report), include_flagged="warning")

        assert set(skip_set.keys()) == {"/a.mcap"}
        assert skip_set["/a.mcap"] == "critical"

    def test_include_flagged_critical_skips_nothing(self, tmp_path):
        """--include-flagged critical is the explicit escape hatch: even with
        critical and warning episodes present in the report, nothing should be
        skipped."""
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning"), ("/c.mcap", "pass")])

        skip_set = resolve_quality_skip_paths(str(report), include_flagged="critical")

        assert skip_set == {}


def _single_camera_config():
    """A minimal DataConfig with exactly one camera.

    Using a single camera avoids LeRobotDataset's multi-camera parallel video
    encoding path (ProcessPoolExecutor), keeping this test fast and
    deterministic while still exercising a real conversion end-to-end.
    """
    from mcap_converter.config.schema import (
        ActionTopicConfig,
        DataConfig,
        FeatureMapping,
        JointNamePattern,
    )

    return DataConfig(
        robot_state_topic="/joint_states",
        joint_name_pattern=JointNamePattern(separator="_", source={"follower": "observation"}, arms={"r": "right"}),
        action_topics={
            "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                arm="right",
                joint_order=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"],
            )
        },
        action_from_observation=False,
        camera_topics=["/cam_waist/image_raw/compressed"],
        camera_topic_mapping={"/cam_waist/image_raw/compressed": "waist"},
        image_resolution=[640, 480],
        observation_feature_mapping=FeatureMapping(state="position", others=[]),
        action_feature_mapping=FeatureMapping(state="position", others=[]),
    )


def _episode_row_index(output: str, mcap_filename: str) -> str:
    """Extract the '#' column value of the per-episode table row for a given MCAP filename.

    The Rich table renders rows as cells separated by box-drawing "│"
    characters (and the whole table is itself nested inside an outer panel
    border), e.g.: "│    │ 3 │ 0003_0.mcap │    119 │ ...". Splitting on "│"
    and taking the cell immediately before the filename cell gives the row's
    "#" column value.
    """
    for line in output.splitlines():
        if mcap_filename not in line:
            continue
        cells = [cell.strip() for cell in line.split("│")]
        for i, cell in enumerate(cells):
            if cell == mcap_filename and i > 0:
                return cells[i - 1]
    raise AssertionError(f"No table row found for {mcap_filename!r} in captured output:\n{output}")


class TestQualitySkipMiddleEpisode:
    """Regression test for episode_original_indices: quality-skip on a
    non-prefix, non-trailing episode must not misattribute frame counts /
    table rows to the wrong original episode index (see commit that
    introduced episode_original_indices)."""

    def test_skip_middle_episode_preserves_original_indices(self, capsys):
        from mcap_converter.cli.convert import collect_mcap_files, convert_session

        mcap_files = collect_mcap_files(str(FIXTURES_ROOT))[:3]
        assert len(mcap_files) == 3, "expected at least 3 fixture episodes under tests/smoke/fixtures/test-session"

        middle_episode = mcap_files[1]
        quality_skip_paths = {str(middle_episode.resolve()): "critical"}

        def run(tmp_path):
            return convert_session(
                input_dir=str(FIXTURES_ROOT),
                output_dir=str(tmp_path / "out"),
                repo_id="testuser/quality-skip-middle",
                robot_type="anvil_openarm",
                fps=30,
                config=_single_camera_config(),
                mcap_files=mcap_files,
                quality_skip_paths=quality_skip_paths,
                debug_plot_episodes=0,
            )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = run(Path(tmp))

        captured = capsys.readouterr()

        # (a) no crash — the skip of a non-prefix episode must not raise IndexError
        # when zipping episode_frame_counts/episode_times against mcap_files.
        assert dataset.meta.total_episodes == 2, "middle episode should be skipped, leaving 2 converted episodes"

        # (b) the per-episode table must attribute frames to the TRUE original
        # index (1-based), not a positionally-shifted index from the filtered
        # (post-skip) sequence.
        first_row_index = _episode_row_index(captured.out, mcap_files[0].name)
        third_row_index = _episode_row_index(captured.out, mcap_files[2].name)

        assert first_row_index == "1", (
            f"expected {mcap_files[0].name} to be labeled episode 1, got {first_row_index}"
        )
        assert third_row_index == "3", (
            f"expected {mcap_files[2].name} to be labeled episode 3 (its true original index), "
            f"got {third_row_index} — this indicates the misattribution bug "
            "episode_original_indices was introduced to fix"
        )


class TestParseEpisodeIndexSpec:
    def test_single_list(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec("1,2,5,6", total_episodes=10) == {1, 2, 5, 6}

    def test_range_end_is_exclusive(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec("1:4", total_episodes=10) == {1, 2, 3}

    def test_open_ended_range_to_end(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec("2:", total_episodes=6) == {2, 3, 4, 5, 6}

    def test_open_ended_range_from_start(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec(":4", total_episodes=10) == {1, 2, 3}

    def test_mixed_list_and_ranges(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec("1,3:5,8", total_episodes=10) == {1, 3, 4, 8}

    def test_whitespace_tolerated(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        assert parse_episode_index_spec("1, 3:5, 8", total_episodes=10) == {1, 3, 4, 8}

    def test_out_of_range_raises(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        with pytest.raises(ValueError):
            parse_episode_index_spec("11", total_episodes=10)

    def test_start_equal_to_end_raises(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        with pytest.raises(ValueError):
            parse_episode_index_spec("3:3", total_episodes=10)

    def test_start_greater_than_end_raises(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        with pytest.raises(ValueError):
            parse_episode_index_spec("5:2", total_episodes=10)

    def test_non_integer_raises(self):
        from mcap_converter.cli.convert import parse_episode_index_spec

        with pytest.raises(ValueError):
            parse_episode_index_spec("abc", total_episodes=10)


class TestMandatoryQualityGate:
    """mcap-convert must refuse to run without a mcap-valid quality report,
    exiting before any output-directory mutation. This gate only checks that
    a report FILE exists — it does not validate contents (that's the
    orthogonal --include-flagged mechanism, covered by TestResolveQualitySkipSet)."""

    def _base_argv(self, tmp_path, output_dir):
        return [
            "-i", str(tmp_path / "fake-input-dir"),
            "-o", str(output_dir),
            "--hf-user", "testuser",
        ]

    def test_explicit_quality_report_path_missing_blocks_and_exits_1(self, tmp_path, capsys):
        from mcap_converter.cli.convert import main

        output_dir = tmp_path / "output"
        missing_report = tmp_path / "nonexistent" / "report.json"
        argv = self._base_argv(tmp_path, output_dir) + [
            "--quality-report", str(missing_report),
        ]

        with pytest.raises(SystemExit) as exc_info:
            main(argv)

        assert exc_info.value.code == 1
        assert not os.path.exists(output_dir), "output dir must not be created when the gate rejects the run"
        captured = capsys.readouterr()
        assert "No mcap-valid quality report found" in captured.out

    def test_no_quality_report_and_no_default_blocks_and_exits_1(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.convert import main

        # input_dir ("fake-input-dir") never exists/gets created, so
        # <input_dir>/mcap_valid_reports/ can never exist either — auto-discovery
        # is guaranteed to find nothing regardless of cwd.
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        argv = self._base_argv(tmp_path, output_dir)

        with pytest.raises(SystemExit) as exc_info:
            main(argv)

        assert exc_info.value.code == 1
        assert not os.path.exists(output_dir), "output dir must not be created when the gate rejects the run"
        captured = capsys.readouterr()
        assert "No mcap-valid quality report found" in captured.out

    def test_directory_as_quality_report_blocks_and_exits_1(self, tmp_path, capsys):
        """A directory satisfies Path.exists() but is not a valid report file.
        The gate must reject it cleanly instead of letting it through to a raw
        IsADirectoryError downstream in resolve_quality_skip_paths()."""
        from mcap_converter.cli.convert import main

        output_dir = tmp_path / "output"
        report_dir = tmp_path / "fake-report-dir"
        report_dir.mkdir()
        argv = self._base_argv(tmp_path, output_dir) + [
            "--quality-report", str(report_dir),
        ]

        with pytest.raises(SystemExit) as exc_info:
            main(argv)

        assert exc_info.value.code == 1
        assert not os.path.exists(output_dir), "output dir must not be created when the gate rejects the run"
        captured = capsys.readouterr()
        assert "No mcap-valid quality report found" in captured.out


class TestIncludeFlaggedDefaultsToWarning:
    """--include-flagged now defaults to 'warning', so a critical episode is
    skipped automatically even without passing the flag at all. 'critical' is
    the explicit escape hatch to convert everything."""

    def _write_single_camera_yaml_config(self, tmp_path) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
robot_state_topic: "/joint_states"
joint_names:
  separator: "_"
  source:
    follower: observation
  arms:
    r: right
action_topics:
  "/follower_r_forward_position_controller/commands":
    arm: "right"
    joint_order: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"]
action_from_observation: false
camera_topics:
  - "/cam_waist/image_raw/compressed"
camera_topic_mapping:
  "/cam_waist/image_raw/compressed": "waist"
image_resolution: [640, 480]
observation_feature_mapping:
  state: "position"
  others: []
action_feature_mapping:
  state: "position"
  others: []
"""
        )
        return config_path

    def _write_report_marking_one_critical(self, tmp_path, mcap_files, critical_index) -> Path:
        payload = {
            "episodes": [
                {
                    "path": str(p.resolve()),
                    "duration_s": 1.0,
                    "severity": "critical" if i == critical_index else "pass",
                    "passed": i != critical_index,
                    "topics": [],
                }
                for i, p in enumerate(mcap_files)
            ]
        }
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(payload))
        return report_path

    def _base_argv(self, tmp_path, config_path, report_path):
        return [
            "-i", str(FIXTURES_ROOT),
            "-o", str(tmp_path / "out"),
            "--config", str(config_path),
            "--robot-type", "anvil_openarm",
            "--hf-user", "testuser",
            "--quality-report", str(report_path),
            "--max-episodes", "3",
            "--debug-plot-episodes", "0",
        ]

    def test_no_include_flagged_arg_skips_critical_episode_by_default(self, tmp_path, capsys):
        from mcap_converter.cli.convert import collect_mcap_files, main

        mcap_files = collect_mcap_files(str(FIXTURES_ROOT))[:3]
        config_path = self._write_single_camera_yaml_config(tmp_path)
        report_path = self._write_report_marking_one_critical(tmp_path, mcap_files, critical_index=1)

        argv = self._base_argv(tmp_path, config_path, report_path)

        main(argv)

        captured = capsys.readouterr()
        assert f"{mcap_files[1].name}  skipped (quality: critical)" in captured.out, (
            "the critical episode should be skipped automatically even though "
            "--include-flagged was never passed on the command line"
        )

    def test_include_flagged_critical_converts_the_critical_episode_anyway(self, tmp_path, capsys):
        from mcap_converter.cli.convert import collect_mcap_files, main

        mcap_files = collect_mcap_files(str(FIXTURES_ROOT))[:3]
        config_path = self._write_single_camera_yaml_config(tmp_path)
        report_path = self._write_report_marking_one_critical(tmp_path, mcap_files, critical_index=1)

        argv = self._base_argv(tmp_path, config_path, report_path) + ["--include-flagged", "critical"]

        main(argv)

        captured = capsys.readouterr()
        assert "skipped (quality:" not in captured.out, (
            "--include-flagged critical must override the new default and "
            "convert every episode, including the one marked critical"
        )
        assert f"{mcap_files[1].name}" in captured.out
