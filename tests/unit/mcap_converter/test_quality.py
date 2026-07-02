"""Tests for the mcap quality validator's coverage/gap analysis.

Verifies:
1. Per-topic coverage analysis (exact/idle/dropframe/leading/trailing gaps).
2. Cross-episode fps degradation detection.
3. Topic resolution from DataConfig (which topics to monitor, quest vs
   leader-follower mode).
4. The I/O adapter that reads a real MCAP file and produces a report.
"""

import json
from pathlib import Path

import pytest

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.schema import ActionTopicConfig, DataConfig
from mcap_converter.core.quality import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    EpisodeQualityReport,
    QualityThresholds,
    TopicQualityReport,
    analyze_topic_coverage,
    apply_batch_fps_check,
    detect_fps_degradation,
    resolve_monitored_topics,
    scan_episode,
    worst_severity,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STUB_MCAP = _REPO_ROOT / "tests/smoke/fixtures/test-session/0001/0001_0.mcap"
_STUB_CMD_CONFIG = _REPO_ROOT / "tests/smoke/fixtures/configs/mcap-converter-smoke-test-cmd.yaml"
_BIMANUAL_CONFIG = _REPO_ROOT / "configs/mcap_converter/openarm_bimanual_quest.yaml"


def make_quest_config() -> DataConfig:
    """Bimanual quest-teleop config, matching the real bug scenario."""
    return DataConfig(
        action_topics={
            "/follower_l_forward_position_controller/commands": ActionTopicConfig(
                arm="left", joint_order=["joint1", "joint2"]
            ),
            "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                arm="right", joint_order=["joint1", "joint2"]
            ),
        },
        camera_topics=["/cam_chest/image_raw/compressed"],
        camera_topic_mapping={"/cam_chest/image_raw/compressed": "chest"},
    )


def make_leader_follower_config() -> DataConfig:
    """Default DataConfig has empty action_topics -> leader-follower mode."""
    return DataConfig(
        camera_topics=["/cam_chest/image_raw/compressed"],
        camera_topic_mapping={"/cam_chest/image_raw/compressed": "chest"},
    )


class TestWorstSeverity:
    def test_critical_beats_warning_and_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING, SEVERITY_CRITICAL]) == SEVERITY_CRITICAL

    def test_warning_beats_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING]) == SEVERITY_WARNING

    def test_all_ok_is_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_OK]) == SEVERITY_OK

    def test_empty_defaults_to_ok(self):
        assert worst_severity([]) == SEVERITY_OK


class TestResolveMonitoredTopics:
    def test_picks_present_camera_variant(self):
        config = make_quest_config()
        available = {"/cam_chest/image_raw/compressed", "/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        camera = next(m for m in monitored if m.label == "chest")
        assert camera.topic == "/cam_chest/image_raw/compressed"
        assert camera.role == "stream"

    def test_camera_missing_all_variants_selects_base(self):
        config = make_quest_config()
        available = {"/joint_states"}  # camera topic entirely absent

        monitored = resolve_monitored_topics(config, available)

        camera = next(m for m in monitored if m.label == "chest")
        assert camera.topic == "/cam_chest/image_raw/compressed"  # falls back to configured name

    def test_quest_mode_produces_action_items_with_arm_label(self):
        config = make_quest_config()
        available = {
            "/follower_l_forward_position_controller/commands",
            "/follower_r_forward_position_controller/commands",
            "/cam_chest/image_raw/compressed",
            "/joint_states",
        }

        monitored = resolve_monitored_topics(config, available)

        labels = {m.label for m in monitored if m.role == "action"}
        assert labels == {"action[left]", "action[right]"}

    def test_leader_follower_mode_has_no_action_items(self):
        config = make_leader_follower_config()
        available = {"/cam_chest/image_raw/compressed", "/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        assert not [m for m in monitored if m.role == "action"]

    def test_robot_state_topic_is_a_stream(self):
        config = make_quest_config()
        available = {"/joint_states"}

        monitored = resolve_monitored_topics(config, available)

        joint_states = next(m for m in monitored if m.label == "joint_states")
        assert joint_states.topic == "/joint_states"
        assert joint_states.role == "stream"


def _thresholds(**overrides) -> QualityThresholds:
    return QualityThresholds(**overrides)


class TestAnalyzeTopicCoverageStream:
    def test_dense_stream_no_gaps_is_ok(self):
        # 30fps for 1 second: 30 evenly spaced timestamps
        timestamps = [i / 30.0 for i in range(30)]

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=timestamps[-1],
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_OK
        assert report.gaps == []
        assert report.message_count == 30
        assert report.avg_fps == pytest.approx(30.0, rel=0.05)

    def test_mid_stream_dropframe_is_critical(self):
        # dense up to t=1.0, then a 2s gap, then dense again
        timestamps = [i / 30.0 for i in range(30)] + [3.0 + i / 30.0 for i in range(30)]

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=timestamps[-1],
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL
        assert any(g.kind == "dropframe" for g in report.gaps)

    def test_leading_gap_is_critical(self):
        timestamps = [i / 30.0 for i in range(30)]
        session_start = timestamps[0] - 5.0  # session began 5s before first message

        report = analyze_topic_coverage(
            timestamps, session_start=session_start, session_end=timestamps[-1],
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL
        assert any(g.kind == "leading" for g in report.gaps)

    def test_trailing_gap_is_critical(self):
        timestamps = [i / 30.0 for i in range(30)]
        session_end = timestamps[-1] + 5.0  # session continued 5s after last message

        report = analyze_topic_coverage(
            timestamps, session_start=timestamps[0], session_end=session_end,
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL
        assert any(g.kind == "trailing" for g in report.gaps)

    def test_zero_messages_is_critical(self):
        report = analyze_topic_coverage(
            [], session_start=0.0, session_end=10.0,
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL
        assert report.message_count == 0
        assert report.avg_fps is None

    def test_single_message_is_critical(self):
        report = analyze_topic_coverage(
            [5.0], session_start=0.0, session_end=10.0,
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL

    def test_high_fps_jitter_is_not_falsely_flagged(self):
        # 60fps with occasional jitter up to 0.2s — below the 0.5s floor, should not flag
        timestamps = [0.0]
        for _ in range(59):
            timestamps.append(timestamps[-1] + 1 / 60.0)
        timestamps[30] = timestamps[29] + 0.2  # one jittery interval, still < floor

        report = analyze_topic_coverage(
            sorted(timestamps), session_start=timestamps[0], session_end=max(timestamps),
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(stream_min_gap_s=0.5),
        )

        assert report.severity == SEVERITY_OK

    def test_unsorted_timestamps_are_sorted_before_analysis(self):
        timestamps = [i / 30.0 for i in range(30)]
        shuffled = list(reversed(timestamps))

        report = analyze_topic_coverage(
            shuffled, session_start=0.0, session_end=timestamps[-1],
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_OK  # would be nonsense/negative intervals if not sorted

    def test_avg_fps_is_none_when_all_timestamps_identical(self):
        report = analyze_topic_coverage(
            [1.0, 1.0, 1.0], session_start=0.0, session_end=2.0,
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.avg_fps is None  # ts[-1] == ts[0] would otherwise divide by zero

    def test_median_not_mean_is_used_for_drop_threshold(self):
        # 4 normal 0.1s intervals plus one 5.0s outlier.
        # median of [0.1, 0.1, 0.1, 0.1, 5.0] is 0.1 -> drop_threshold =
        # max(0.5, 5*0.1) = 0.5, so the 5.0s interval (> 0.5) is flagged.
        # mean of the same intervals is 1.08 -> a mean-based drop_threshold
        # would be max(0.5, 5*1.08) = 5.4, under which 5.0 would NOT be
        # flagged, so severity would stay OK. The two implementations
        # disagree on both severity and whether a dropframe gap is reported.
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 5.4]

        report = analyze_topic_coverage(
            timestamps, session_start=timestamps[0], session_end=timestamps[-1],
            topic="/joint_states", label="joint_states", role="stream",
            thresholds=_thresholds(),
        )

        assert report.severity == SEVERITY_CRITICAL
        dropframe_gaps = [g for g in report.gaps if g.kind == "dropframe"]
        assert len(dropframe_gaps) == 1
        assert dropframe_gaps[0].duration_s == pytest.approx(5.0)


class TestAnalyzeTopicCoverageAction:
    def test_zero_messages_without_afo_is_warning_not_critical(self):
        report = analyze_topic_coverage(
            [], session_start=0.0, session_end=10.0,
            topic="/follower_r_.../commands", label="action[right]", role="action",
            thresholds=_thresholds(), action_from_observation=False,
        )

        assert report.severity == SEVERITY_WARNING  # NOT critical — could be a single-arm task

    def test_zero_messages_with_afo_is_ok(self):
        report = analyze_topic_coverage(
            [], session_start=0.0, session_end=10.0,
            topic="/follower_r_.../commands", label="action[right]", role="action",
            thresholds=_thresholds(), action_from_observation=True,
        )

        assert report.severity == SEVERITY_OK

    def test_long_idle_gap_is_warning_not_critical(self):
        # published at t=1.0, then idle for 5s, then again at t=6.0
        timestamps = [1.0, 6.0]

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=7.0,
            topic="/follower_r_.../commands", label="action[right]", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )

        assert report.severity == SEVERITY_WARNING
        assert any(g.kind == "idle" for g in report.gaps)

    def test_dense_action_is_ok(self):
        timestamps = [i * 0.1 for i in range(20)]  # 10Hz, no idle gaps

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=timestamps[-1],
            topic="/follower_r_.../commands", label="action[right]", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )

        assert report.severity == SEVERITY_OK
        assert report.avg_fps is None  # action topics don't get a fixed-rate fps figure

    def test_action_leading_and_trailing_gaps_are_not_flagged(self):
        # action starts late and ends early relative to session — this is normal idle,
        # not a leading/trailing dropframe like a stream would have.
        timestamps = [3.0, 3.1, 3.2]

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=10.0,
            topic="/follower_r_.../commands", label="action[right]", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )

        assert not any(g.kind in ("leading", "trailing") for g in report.gaps)


class TestAnalyzeTopicCoverageMetrics:
    def test_coverage_and_gap_metrics_are_computed(self):
        timestamps = [1.0, 6.0]  # one 5s idle gap out of a 10s session

        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=10.0,
            topic="t", label="t", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )

        assert report.total_gap_s == pytest.approx(5.0)
        assert report.longest_gap_s == pytest.approx(5.0)
        assert report.coverage_ratio == pytest.approx(0.5)


class TestDetectFpsDegradation:
    def test_episode_far_below_median_is_degraded(self):
        episode_fps = {"ep0": 60.0, "ep1": 60.0, "ep2": 50.0}  # ep2 is ~17% below median 60

        result = detect_fps_degradation(episode_fps, _thresholds(fps_degradation_tolerance=0.15))

        assert result["ep2"][0] is True
        assert result["ep0"][0] is False
        assert result["ep1"][0] is False

    def test_all_similar_fps_none_degraded(self):
        episode_fps = {"ep0": 60.0, "ep1": 59.0, "ep2": 61.0}

        result = detect_fps_degradation(episode_fps, _thresholds(fps_degradation_tolerance=0.15))

        assert all(not degraded for degraded, _ in result.values())

    def test_single_episode_is_its_own_median_never_degraded(self):
        episode_fps = {"ep0": 60.0}

        result = detect_fps_degradation(episode_fps, _thresholds(fps_degradation_tolerance=0.15))

        assert result["ep0"][0] is False

    def test_leave_one_out_generalizes_to_four_episodes(self):
        # ep3's leave-one-out reference is median([60.0, 60.0, 60.0]) = 60.0,
        # threshold 60.0 * 0.85 = 51.0, and 48.0 < 51.0 -> degraded.
        # ep0's leave-one-out reference is median([60.0, 60.0, 48.0]) = 60.0
        # (middle of sorted [48.0, 60.0, 60.0]), threshold 51.0, 60.0 is not
        # below that -> not degraded.
        episode_fps = {"ep0": 60.0, "ep1": 60.0, "ep2": 60.0, "ep3": 48.0}

        result = detect_fps_degradation(episode_fps, _thresholds(fps_degradation_tolerance=0.15))

        assert result["ep3"][0] is True
        assert result["ep0"][0] is False
        assert result["ep1"][0] is False
        assert result["ep2"][0] is False


class TestApplyBatchFpsCheck:
    def test_ok_episode_upgraded_to_warning_on_degradation(self):
        ok_topic = TopicQualityReport(
            topic="/cam", label="chest", role="stream", message_count=100, avg_fps=50.0,
            coverage_ratio=1.0, total_gap_s=0.0, longest_gap_s=0.0, severity=SEVERITY_OK, reason="OK",
        )
        healthy_topic = TopicQualityReport(
            topic="/cam", label="chest", role="stream", message_count=100, avg_fps=60.0,
            coverage_ratio=1.0, total_gap_s=0.0, longest_gap_s=0.0, severity=SEVERITY_OK, reason="OK",
        )
        degraded_ep = EpisodeQualityReport(
            path="ep_degraded", duration_s=10.0, severity=SEVERITY_OK, passed=True, topics=[ok_topic],
        )
        healthy_ep = EpisodeQualityReport(
            path="ep_healthy", duration_s=10.0, severity=SEVERITY_OK, passed=True, topics=[healthy_topic],
        )

        updated = apply_batch_fps_check([degraded_ep, healthy_ep], _thresholds(fps_degradation_tolerance=0.15))

        degraded_result = next(r for r in updated if r.path == "ep_degraded")
        assert degraded_result.severity == SEVERITY_WARNING
        assert degraded_result.passed is True  # warning still passes
        assert "fps" in degraded_result.topics[0].reason.lower()

    def test_existing_critical_not_downgraded_by_fps_check(self):
        critical_topic = TopicQualityReport(
            topic="/cam", label="chest", role="stream", message_count=0, avg_fps=None,
            coverage_ratio=0.0, total_gap_s=10.0, longest_gap_s=10.0,
            severity=SEVERITY_CRITICAL, reason="stream topic 零訊息",
        )
        healthy_topic = TopicQualityReport(
            topic="/cam", label="chest", role="stream", message_count=100, avg_fps=60.0,
            coverage_ratio=1.0, total_gap_s=0.0, longest_gap_s=0.0, severity=SEVERITY_OK, reason="OK",
        )
        critical_ep = EpisodeQualityReport(
            path="ep_critical", duration_s=10.0, severity=SEVERITY_CRITICAL, passed=False, topics=[critical_topic],
        )
        healthy_ep = EpisodeQualityReport(
            path="ep_healthy", duration_s=10.0, severity=SEVERITY_OK, passed=True, topics=[healthy_topic],
        )

        updated = apply_batch_fps_check([critical_ep, healthy_ep], _thresholds(fps_degradation_tolerance=0.15))

        critical_result = next(r for r in updated if r.path == "ep_critical")
        assert critical_result.severity == SEVERITY_CRITICAL  # fps check with avg_fps=None must skip this topic
        assert critical_result.passed is False


class TestScanEpisodeIntegration:
    def test_healthy_stub_passes_with_matching_single_arm_config(self):
        config = ConfigLoader.from_yaml(str(_STUB_CMD_CONFIG))

        report = scan_episode(str(_STUB_MCAP), config, QualityThresholds())

        assert report.passed is True
        assert report.severity in (SEVERITY_OK, SEVERITY_WARNING)  # never critical for a healthy stub

    def test_duration_is_a_plausible_positive_value(self):
        # NOTE: this used to be named
        # test_session_bounds_come_from_message_timestamps_not_summary, but a
        # loose `1.0 < duration_s < 30.0` range check can't actually tell
        # timestamps-of-monitored-topics apart from file-wide-summary bounds.
        # Investigated: in this fixture every channel (joint_states, all
        # cameras, the action-command topic) spans the exact same range,
        # 0.0 -> 3.966666627s, which is also exactly what the MCAP summary's
        # file-level message_start_time/message_end_time report. So no
        # config subset of this fixture can discriminate the two approaches.
        # This test is renamed to describe what it actually checks; see
        # test_duration_matches_manually_verified_monitored_topic_timestamps
        # below for a test that pins the exact computation.
        config = ConfigLoader.from_yaml(str(_STUB_CMD_CONFIG))

        report = scan_episode(str(_STUB_MCAP), config, QualityThresholds())

        assert 1.0 < report.duration_s < 30.0

    def test_duration_matches_manually_verified_monitored_topic_timestamps(self):
        # Pins scan_episode's duration computation to a manually-verified
        # exact value: `_collect_timestamps` on this fixture's
        # /joint_states topic returns messages spanning exactly
        # 0.0 -> 3.966666627 seconds (verified directly against the file).
        # This doesn't discriminate "from timestamps" vs. "from file-wide
        # summary" (they coincide in this fixture — see the note on
        # test_duration_is_a_plausible_positive_value above), but it does
        # pin the exact computed value to high precision, which the old
        # loose range check did not.
        config = ConfigLoader.from_yaml(str(_STUB_CMD_CONFIG))

        report = scan_episode(str(_STUB_MCAP), config, QualityThresholds())

        assert report.duration_s == pytest.approx(3.966666627, abs=1e-6)

    def test_missing_arm_is_warning_with_bimanual_config_on_single_arm_stub(self):
        config = ConfigLoader.from_yaml(str(_BIMANUAL_CONFIG))
        # The single-arm stub never recorded a left-wrist camera (no left-arm
        # hardware was mounted for this session), so the bimanual config's
        # full camera list would also surface an unrelated, genuinely-correct
        # CRITICAL for that missing stream. Drop it so this test isolates the
        # actual regression under test: the missing *action* topic alone.
        left_wrist_cam = "/cam_wrist_l/image_raw/compressed"
        config.camera_topics = [c for c in config.camera_topics if c != left_wrist_cam]
        config.camera_topic_mapping.pop(left_wrist_cam, None)

        report = scan_episode(str(_STUB_MCAP), config, QualityThresholds())

        left_action = next(t for t in report.topics if t.label == "action[left]")
        assert left_action.severity == SEVERITY_WARNING  # NOT critical — key regression test for this revision
        assert report.passed is True  # a warning-only episode still passes

    def test_missing_arm_is_ok_when_action_from_observation_true(self):
        config = ConfigLoader.from_yaml(str(_BIMANUAL_CONFIG))
        config.action_from_observation = True

        report = scan_episode(str(_STUB_MCAP), config, QualityThresholds())

        left_action = next(t for t in report.topics if t.label == "action[left]")
        assert left_action.severity == SEVERITY_OK

    def test_nonexistent_file_produces_read_error_not_a_crash(self):
        report = scan_episode("/no/such/file.mcap", ConfigLoader.get_default(), QualityThresholds())

        assert report.passed is False
        assert report.severity == SEVERITY_CRITICAL
        assert report.read_error is not None
        assert report.topics == []

    def test_corrupt_file_produces_read_error_not_a_misleading_report(self, tmp_path):
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"this is not a valid mcap file at all, just garbage bytes")

        report = scan_episode(str(garbage_file), ConfigLoader.get_default(), QualityThresholds())

        assert report.passed is False
        assert report.read_error is not None


class TestMcapValidCli:
    def test_json_output_is_valid_and_exit_code_zero_without_critical(self, capsys):
        from mcap_converter.cli.mcap_valid import main

        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--config", str(_STUB_CMD_CONFIG),
            "--format", "json",
            "--fail-on-critical",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert "episodes" in payload
        assert len(payload["episodes"]) == 1
        assert exit_code == 0

    def test_fail_on_critical_exits_nonzero_when_critical_present(self, tmp_path):
        from mcap_converter.cli.mcap_valid import main

        # Point camera_topics at something that doesn't exist in the stub -> critical.
        bad_config_path = tmp_path / "bad.yaml"
        bad_config_path.write_text(
            "robot_state_topic: \"/joint_states\"\n"
            "camera_topics:\n  - \"/nonexistent_camera/image_raw\"\n"
            "camera_topic_mapping:\n  \"/nonexistent_camera/image_raw\": \"missing_cam\"\n"
        )

        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--config", str(bad_config_path),
            "--format", "json",
            "--fail-on-critical",
        ])

        assert exit_code == 1

    def test_output_file_is_written(self, tmp_path):
        from mcap_converter.cli.mcap_valid import main

        out_file = tmp_path / "report.json"
        main([
            "-i", str(_STUB_MCAP), "--config", str(_STUB_CMD_CONFIG),
            "--format", "json", "--output", str(out_file),
        ])

        payload = json.loads(out_file.read_text())
        assert "episodes" in payload

    def test_unreadable_file_is_reported_not_crashed_and_fails_on_critical(self, tmp_path, capsys):
        from mcap_converter.cli.mcap_valid import main

        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        exit_code = main([
            "-i", str(garbage_file), "--config", str(_STUB_CMD_CONFIG),
            "--format", "json", "--fail-on-critical",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["episodes"][0]["read_error"] is not None
        assert exit_code == 1

    def test_unreadable_file_shown_in_table_output(self, tmp_path, capsys):
        from mcap_converter.cli.mcap_valid import main

        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        main(["-i", str(garbage_file), "--config", str(_STUB_CMD_CONFIG)])

        captured = capsys.readouterr()
        assert "error" in captured.out.lower()
        # The read_error message itself must appear, not just the word "error" —
        # this is the regression the plan revision was written to catch.
        assert "InvalidMagic" in captured.out or "not a valid" in captured.out.lower() or "Errno" in captured.out
