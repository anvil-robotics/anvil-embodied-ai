"""Tests for mcap-convert's --quality-report / --skip-flagged integration."""

import json

import pytest


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
    def test_no_skip_flagged_skips_nothing(self, tmp_path):
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning")])

        skip_set = resolve_quality_skip_paths(str(report), skip_flagged=None)

        assert skip_set == {}

    def test_bare_skip_flagged_skips_only_critical(self, tmp_path):
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning"), ("/c.mcap", "ok")])

        skip_set = resolve_quality_skip_paths(str(report), skip_flagged="critical")

        assert set(skip_set.keys()) == {"/a.mcap"}
        assert skip_set["/a.mcap"] == "critical"

    def test_skip_flagged_warning_skips_both(self, tmp_path):
        from mcap_converter.cli.convert import resolve_quality_skip_paths

        report = _write_report(tmp_path, [("/a.mcap", "critical"), ("/b.mcap", "warning"), ("/c.mcap", "ok")])

        skip_set = resolve_quality_skip_paths(str(report), skip_flagged="warning")

        assert set(skip_set.keys()) == {"/a.mcap", "/b.mcap"}
