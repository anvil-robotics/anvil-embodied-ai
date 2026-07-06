"""Tests for the mcap quality validator's coverage/gap analysis.

Verifies:
1. Per-topic coverage analysis (exact/idle/dropframe/leading/trailing gaps).
2. Cross-episode fps degradation detection.
3. Config-free topic classification from ROS2 message type (classify_topic /
   classify_topics) — replaces the old DataConfig-based topic resolution.
4. The I/O adapter that reads a real MCAP file and produces a report.
5. Cross-episode topic-presence check (apply_batch_topic_presence_check).
"""

import json
from pathlib import Path

import pytest

from mcap_converter.core.quality import (
    ROLE_ACTION,
    ROLE_STREAM,
    ROLE_UNCLASSIFIED,
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    EpisodeQualityReport,
    GapInterval,
    QualityThresholds,
    TopicQualityReport,
    _TopicSummary,
    analyze_topic_coverage,
    apply_batch_fps_check,
    apply_batch_topic_presence_check,
    classify_topic,
    classify_topics,
    detect_fps_degradation,
    scan_episode,
    worst_severity,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STUB_MCAP = _REPO_ROOT / "tests/smoke/fixtures/test-session/0001/0001_0.mcap"


class TestWorstSeverity:
    def test_critical_beats_warning_and_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING, SEVERITY_CRITICAL]) == SEVERITY_CRITICAL

    def test_warning_beats_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_WARNING]) == SEVERITY_WARNING

    def test_all_ok_is_ok(self):
        assert worst_severity([SEVERITY_OK, SEVERITY_OK]) == SEVERITY_OK

    def test_empty_defaults_to_ok(self):
        assert worst_severity([]) == SEVERITY_OK


class TestClassifyTopic:
    def test_joint_state_is_stream_joint_states(self):
        m = classify_topic("/joint_states", "sensor_msgs/JointState")

        assert m.role == ROLE_STREAM
        assert m.label == "joint_states"
        assert m.message_type == "sensor_msgs/JointState"

    def test_compressed_image_matching_cam_pattern_is_stream_with_camera_label(self):
        m = classify_topic("/cam_waist/image_raw/compressed", "sensor_msgs/CompressedImage")

        assert m.role == ROLE_STREAM
        assert m.label == "waist"

    def test_uncompressed_image_also_classifies_as_stream(self):
        m = classify_topic("/cam_chest/image_raw", "sensor_msgs/Image")

        assert m.role == ROLE_STREAM
        assert m.label == "chest"

    def test_camera_type_not_matching_cam_pattern_falls_back_without_crash(self):
        m = classify_topic("/some/other/camera/topic", "sensor_msgs/CompressedImage")

        assert m.role == ROLE_STREAM
        assert m.label == "some_other_camera_topic"

    def test_float64_multiarray_left_arm_is_action_left(self):
        m = classify_topic(
            "/follower_l_forward_position_controller/commands", "std_msgs/Float64MultiArray"
        )

        assert m.role == ROLE_ACTION
        assert m.label == "action[left]"

    def test_float64_multiarray_right_arm_is_action_right(self):
        m = classify_topic(
            "/follower_r_forward_position_controller/commands", "std_msgs/Float64MultiArray"
        )

        assert m.role == ROLE_ACTION
        assert m.label == "action[right]"

    def test_float64_multiarray_not_matching_arm_pattern_falls_back_without_crash(self):
        m = classify_topic("/some/command/topic", "std_msgs/Float64MultiArray")

        assert m.role == ROLE_ACTION
        assert m.label == "action[some_command_topic]"

    def test_unrecognized_schema_is_unclassified_with_type_preserved(self):
        m = classify_topic("/tf", "tf2_msgs/TFMessage")

        assert m.role == ROLE_UNCLASSIFIED
        assert m.message_type == "tf2_msgs/TFMessage"

    def test_none_schema_is_unclassified(self):
        m = classify_topic("/rosout", None)

        assert m.role == ROLE_UNCLASSIFIED
        assert m.message_type is None

    def test_schema_name_with_msg_infix_still_classifies_as_stream(self):
        m = classify_topic("/joint_states", "sensor_msgs/msg/JointState")

        assert m.role == ROLE_STREAM
        assert m.label == "joint_states"
        assert m.message_type == "sensor_msgs/JointState"

    def test_classify_topics_returns_one_entry_per_key_sorted_by_topic(self):
        topic_schemas = {
            "/joint_states": "sensor_msgs/JointState",
            "/cam_chest/image_raw/compressed": "sensor_msgs/CompressedImage",
            "/follower_r_forward_position_controller/commands": "std_msgs/Float64MultiArray",
        }

        monitored = classify_topics(topic_schemas)

        assert len(monitored) == 3
        assert [m.topic for m in monitored] == sorted(topic_schemas)


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

    def test_idle_gap_reason_includes_time_range(self):
        # timestamps with a single arm-idle gap from 3.0s to 10.0s within a 15s session
        timestamps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0, 10.5, 11.0, 11.5, 12.0]
        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=15.0,
            topic="/some/action/topic", label="action[left]", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )
        assert report.severity == SEVERITY_WARNING
        assert "3.00s~10.00s" in report.reason
        assert "7.00s" in report.reason  # duration of the gap (10.0 - 3.0)

    def test_idle_gap_reason_picks_earliest_gap_when_durations_tie(self):
        # dense messages (0.5s apart, below the 1.0s threshold) except for two
        # idle gaps of exactly equal duration (2.0s each): 3.0s-5.0s and 8.0s-10.0s.
        # max(..., key=...) must deterministically pick the first-encountered (earliest) one.
        timestamps = (
            [round(i * 0.5, 1) for i in range(7)]  # 0.0..3.0
            + [round(5.0 + i * 0.5, 1) for i in range(7)]  # 5.0..8.0
            + [round(10.0 + i * 0.5, 1) for i in range(5)]  # 10.0..12.0
        )
        report = analyze_topic_coverage(
            timestamps, session_start=0.0, session_end=15.0,
            topic="/some/action/topic", label="action[left]", role="action",
            thresholds=_thresholds(action_warn_gap_s=1.0),
        )
        assert report.severity == SEVERITY_WARNING
        assert sum(1 for g in report.gaps if g.kind == "idle") == 2
        assert "3.00s~5.00s" in report.reason
        assert "8.00s~10.00s" not in report.reason


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


def _make_topic(topic, role, label=None, severity=SEVERITY_OK):
    return TopicQualityReport(
        topic=topic, label=label or topic, role=role, message_count=100, avg_fps=30.0,
        coverage_ratio=1.0, total_gap_s=0.0, longest_gap_s=0.0, severity=severity, reason="OK",
    )


def _make_episode(path, topics):
    return EpisodeQualityReport(
        path=path, duration_s=10.0, severity=SEVERITY_OK, passed=True, topics=topics,
    )


class TestApplyBatchTopicPresenceCheck:
    def test_episode_missing_majority_topic_gets_synthesized_critical_entry(self):
        camera = _make_topic("/cam_chest/image_raw/compressed", ROLE_STREAM, label="chest")
        ep_with_cam = [_make_episode(f"ep{i}", [camera]) for i in range(3)]
        ep_missing_cam = _make_episode("ep_missing", [])

        updated = apply_batch_topic_presence_check([*ep_with_cam, ep_missing_cam])

        missing_result = next(r for r in updated if r.path == "ep_missing")
        synthesized = next(t for t in missing_result.topics if t.topic == "/cam_chest/image_raw/compressed")
        assert synthesized.severity == SEVERITY_CRITICAL
        assert "3/4" in synthesized.reason
        assert missing_result.severity == SEVERITY_CRITICAL
        assert missing_result.passed is False

    def test_episodes_not_missing_anything_are_returned_unchanged(self):
        camera = _make_topic("/cam_chest/image_raw/compressed", ROLE_STREAM, label="chest")
        episodes = [_make_episode(f"ep{i}", [camera]) for i in range(3)]

        updated = apply_batch_topic_presence_check(episodes)

        for original, result in zip(episodes, updated):
            assert result.topics == original.topics
            assert result.severity == original.severity
            assert result.passed == original.passed

    def test_single_episode_batch_is_a_no_op(self):
        episodes = [_make_episode("ep0", [])]

        updated = apply_batch_topic_presence_check(episodes)

        assert updated == episodes

    def test_minority_topic_does_not_trigger_false_positive_on_majority(self):
        camera = _make_topic("/cam_chest/image_raw/compressed", ROLE_STREAM, label="chest")
        rare_topic = _make_topic("/rare/event_log", ROLE_ACTION, label="rare")
        # Only 1 of 3 episodes has the rare topic -> below quorum (2) -> the
        # other 2 episodes must NOT be flagged as "missing" it.
        ep0 = _make_episode("ep0", [camera, rare_topic])
        ep1 = _make_episode("ep1", [camera])
        ep2 = _make_episode("ep2", [camera])

        updated = apply_batch_topic_presence_check([ep0, ep1, ep2])

        for result in updated:
            assert not any(t.topic == "/rare/event_log" for t in result.topics if t.severity == SEVERITY_CRITICAL)
        # And none of them gained any synthesized entries at all.
        assert len(next(r for r in updated if r.path == "ep1").topics) == 1
        assert len(next(r for r in updated if r.path == "ep2").topics) == 1


class TestScanEpisodeIntegration:
    def test_healthy_stub_passes(self):
        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

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
        # subset of this fixture can discriminate the two approaches.
        # This test is renamed to describe what it actually checks; see
        # test_duration_matches_manually_verified_monitored_topic_timestamps
        # below for a test that pins the exact computation.
        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

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
        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        assert report.duration_s == pytest.approx(3.966666627, abs=1e-6)

    def test_every_topic_is_classified_stream_or_action_with_message_type(self):
        # The stub fixture (/joint_states, 3 camera topics, 1 action topic)
        # has no unclassified topics — every declared channel is one of the
        # 3 known robot-pipeline message types.
        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        assert len(report.topics) == 5
        for t in report.topics:
            assert t.role in (ROLE_STREAM, ROLE_ACTION)
            assert t.message_type is not None

    def test_unclassified_topic_is_ok_and_does_not_affect_episode_severity(self, monkeypatch):
        from mcap_converter.core import quality as quality_module

        baseline = scan_episode(str(_STUB_MCAP), QualityThresholds())
        real_info = quality_module._summary_topic_info(str(_STUB_MCAP))

        def fake_summary_topic_info(mcap_path):
            info = dict(real_info)
            info["/rosout"] = _TopicSummary(count=5, schema_name="rcl_interfaces/Log")
            return info

        monkeypatch.setattr(quality_module, "_summary_topic_info", fake_summary_topic_info)

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        rosout = next(t for t in report.topics if t.topic == "/rosout")
        assert rosout.role == ROLE_UNCLASSIFIED
        assert rosout.severity == SEVERITY_OK
        assert rosout.message_count == 5
        assert rosout.message_type == "rcl_interfaces/Log"
        # /rosout has no real channel in the fixture file, so there are no
        # actual messages to compute an fps from despite the fake count=5.
        assert rosout.avg_fps is None
        assert report.severity == baseline.severity

    def test_unclassified_topic_with_real_messages_gets_computed_avg_fps(self, monkeypatch):
        # Relabel an existing, genuinely-recorded topic's schema as unclassified
        # (the underlying MCAP channel is untouched, so decoding its real
        # messages still works) to verify avg_fps is now computed from real
        # timestamps for unclassified topics, using the same
        # (n-1)/(span) formula already used for role="stream".
        from mcap_converter.core import quality as quality_module

        real_info = quality_module._summary_topic_info(str(_STUB_MCAP))

        def fake_summary_topic_info(mcap_path):
            info = dict(real_info)
            joint = info["/joint_states"]
            info["/joint_states"] = _TopicSummary(count=joint.count, schema_name="custom_msgs/Unknown")
            return info

        monkeypatch.setattr(quality_module, "_summary_topic_info", fake_summary_topic_info)

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        joint = next(t for t in report.topics if t.topic == "/joint_states")
        assert joint.role == ROLE_UNCLASSIFIED
        assert joint.severity == SEVERITY_OK
        assert joint.message_count == 120
        # Pinned to the same manually-verified span used by
        # test_duration_matches_manually_verified_monitored_topic_timestamps:
        # 120 messages spanning 0.0 -> 3.966666627s.
        assert joint.avg_fps == pytest.approx(119 / 3.966666627, rel=1e-6)

    def test_unclassified_topic_with_single_message_has_no_avg_fps(self, monkeypatch):
        # A single timestamp can't produce a meaningful average — same rule
        # already applied to role="stream" topics.
        from mcap_converter.core import quality as quality_module

        real_info = quality_module._summary_topic_info(str(_STUB_MCAP))
        real_collect_timestamps = quality_module._collect_timestamps

        def fake_summary_topic_info(mcap_path):
            info = dict(real_info)
            info["/rosout"] = _TopicSummary(count=1, schema_name="rcl_interfaces/Log")
            return info

        def fake_collect_timestamps(mcap_path, topics):
            if topics == ["/rosout"]:
                return {"/rosout": [1.23]}
            return real_collect_timestamps(mcap_path, topics)

        monkeypatch.setattr(quality_module, "_summary_topic_info", fake_summary_topic_info)
        monkeypatch.setattr(quality_module, "_collect_timestamps", fake_collect_timestamps)

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        rosout = next(t for t in report.topics if t.topic == "/rosout")
        assert rosout.role == ROLE_UNCLASSIFIED
        assert rosout.message_count == 1
        assert rosout.avg_fps is None
        assert rosout.severity == SEVERITY_OK

    def test_unclassified_topic_never_perturbs_session_bounds_or_monitored_severity(self, monkeypatch):
        # A fast, wildly-out-of-session-range unclassified topic (like a real
        # high-rate /tf) must never shift session_start/session_end, since
        # that would risk mis-flagging gaps/severity on the real monitored
        # stream/action topics. Its own fps is still computed and shown.
        from mcap_converter.core import quality as quality_module

        baseline = scan_episode(str(_STUB_MCAP), QualityThresholds())
        real_info = quality_module._summary_topic_info(str(_STUB_MCAP))
        real_collect_timestamps = quality_module._collect_timestamps

        def fake_summary_topic_info(mcap_path):
            info = dict(real_info)
            info["/tf"] = _TopicSummary(count=3, schema_name="tf2_msgs/TFMessage")
            return info

        def fake_collect_timestamps(mcap_path, topics):
            if topics == ["/tf"]:
                return {"/tf": [-100.0, 0.0, 100.0]}
            return real_collect_timestamps(mcap_path, topics)

        monkeypatch.setattr(quality_module, "_summary_topic_info", fake_summary_topic_info)
        monkeypatch.setattr(quality_module, "_collect_timestamps", fake_collect_timestamps)

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        assert report.duration_s == baseline.duration_s
        assert report.severity == baseline.severity

        tf = next(t for t in report.topics if t.topic == "/tf")
        assert tf.role == ROLE_UNCLASSIFIED
        assert tf.severity == SEVERITY_OK
        assert tf.avg_fps == pytest.approx(2 / 200.0)

        baseline_by_topic = {t.topic: t for t in baseline.topics}
        for t in report.topics:
            if t.topic == "/tf":
                continue
            assert t.severity == baseline_by_topic[t.topic].severity
            assert t.avg_fps == baseline_by_topic[t.topic].avg_fps

    def test_unclassified_timestamp_decode_failure_degrades_without_crashing(self, monkeypatch):
        # An unclassified topic can in principle be any ROS2 message type (or
        # even a non-ROS2-encoded channel) — unlike the 3 well-tested
        # monitored types. A decode failure while collecting its timestamps
        # must degrade to avg_fps=None, not crash the whole scan, since
        # unclassified topics are informational-only by design.
        from mcap.exceptions import McapError

        from mcap_converter.core import quality as quality_module

        real_info = quality_module._summary_topic_info(str(_STUB_MCAP))
        real_collect_timestamps = quality_module._collect_timestamps

        def fake_summary_topic_info(mcap_path):
            info = dict(real_info)
            info["/rosout"] = _TopicSummary(count=5, schema_name="rcl_interfaces/Log")
            return info

        def fake_collect_timestamps(mcap_path, topics):
            if topics == ["/rosout"]:
                raise McapError("simulated decode failure for an exotic message type")
            return real_collect_timestamps(mcap_path, topics)

        monkeypatch.setattr(quality_module, "_summary_topic_info", fake_summary_topic_info)
        monkeypatch.setattr(quality_module, "_collect_timestamps", fake_collect_timestamps)

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())

        rosout = next(t for t in report.topics if t.topic == "/rosout")
        assert rosout.role == ROLE_UNCLASSIFIED
        assert rosout.message_count == 5
        assert rosout.avg_fps is None
        assert rosout.severity == SEVERITY_OK
        # The monitored stream/action topics' own scan must be unaffected.
        assert report.severity in (SEVERITY_OK, SEVERITY_WARNING)

    def test_nonexistent_file_produces_read_error_not_a_crash(self):
        report = scan_episode("/no/such/file.mcap", QualityThresholds())

        assert report.passed is False
        assert report.severity == SEVERITY_CRITICAL
        assert report.read_error is not None
        assert report.topics == []

    def test_corrupt_file_produces_read_error_not_a_misleading_report(self, tmp_path):
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"this is not a valid mcap file at all, just garbage bytes")

        report = scan_episode(str(garbage_file), QualityThresholds())

        assert report.passed is False
        assert report.read_error is not None


class TestMcapValidCli:
    # NOTE: mcap-valid now *always* writes ./mcap_valid_reports/<name>/report.{json,md}
    # relative to the current working directory (see TestDefaultReportPaths /
    # TestDefaultReportWriting below). Every test in this class chdirs into
    # tmp_path first so that unconditional default-report writing never lands
    # inside the real repo working tree while running the test suite.

    def test_json_output_is_valid_and_exit_code_zero_without_critical(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--format", "json",
            "--fail-on-critical",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert "episodes" in payload
        assert len(payload["episodes"]) == 1
        assert exit_code == 0

    def test_fail_on_critical_exits_nonzero_when_critical_present(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        # Config-free classification has no equivalent of the old "config points
        # at a camera topic that doesn't exist" trick to force a critical episode.
        # An unreadable/corrupt file always produces a CRITICAL, failed episode
        # report regardless of config, so it's the simplest reliable trigger here.
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        exit_code = main([
            "-i", str(garbage_file),
            "--format", "json",
            "--fail-on-critical",
        ])

        assert exit_code == 1

    def test_output_file_is_written(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        out_file = tmp_path / "report.json"
        main([
            "-i", str(_STUB_MCAP),
            "--format", "json", "--output", str(out_file),
        ])

        payload = json.loads(out_file.read_text())
        assert "episodes" in payload

    def test_unreadable_file_is_reported_not_crashed_and_fails_on_critical(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        exit_code = main([
            "-i", str(garbage_file),
            "--format", "json", "--fail-on-critical",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["episodes"][0]["read_error"] is not None
        assert exit_code == 1

    def test_unreadable_file_shown_in_table_output(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        main(["-i", str(garbage_file)])

        captured = capsys.readouterr()
        assert "error" in captured.out.lower()
        # The read_error message itself must appear, not just the word "error" —
        # this is the regression the plan revision was written to catch.
        assert "InvalidMagic" in captured.out or "not a valid" in captured.out.lower() or "Errno" in captured.out

    def test_directory_scan_covers_all_episodes_and_runs_batch_fps_check(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        # tests/smoke/fixtures/test-session/ contains 5 numbered episode
        # subdirectories (0001-0005), each with one .mcap file.
        stub_session_dir = _STUB_MCAP.parent.parent

        exit_code = main([
            "-i", str(stub_session_dir),
            "--format", "json",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert len(payload["episodes"]) == 5  # all 5 stub episodes discovered
        assert exit_code == 0

    def test_table_format_with_output_still_writes_json_file(self, tmp_path, monkeypatch, capsys):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        out_file = tmp_path / "report.json"
        main([
            "-i", str(_STUB_MCAP),
            "--output", str(out_file),  # no --format flag -> defaults to table
        ])

        captured = capsys.readouterr()
        # Table was printed to stdout...
        assert "mcap-valid report" in captured.out
        # ...AND the JSON file was also written correctly.
        payload = json.loads(out_file.read_text())
        assert "episodes" in payload
        assert len(payload["episodes"]) == 1

    def test_default_output_writes_json_and_md_without_any_flags(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        stub_session_dir = _STUB_MCAP.parent.parent  # tests/smoke/fixtures/test-session (5 episodes)

        exit_code = main([
            "-i", str(stub_session_dir),
        ])

        assert exit_code == 0
        default_json = tmp_path / "mcap_valid_reports" / "test-session" / "report.json"
        default_md = tmp_path / "mcap_valid_reports" / "test-session" / "report.md"
        assert default_json.exists()
        assert default_md.exists()

        payload = json.loads(default_json.read_text())
        assert len(payload["episodes"]) == 5

        md_text = default_md.read_text()
        assert md_text.count("### ") >= 5

    def test_explicit_output_flag_still_works_alongside_default_files(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        stub_session_dir = _STUB_MCAP.parent.parent
        custom_output = tmp_path / "custom.json"

        exit_code = main([
            "-i", str(stub_session_dir),
            "--format", "json",
            "--output", str(custom_output),
        ])

        assert exit_code == 0
        assert (tmp_path / "mcap_valid_reports" / "test-session" / "report.json").exists()
        assert (tmp_path / "mcap_valid_reports" / "test-session" / "report.md").exists()
        assert custom_output.exists()

    def test_nonexistent_input_path_errors_without_writing_reports(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)

        exit_code = main([
            "-i", str(tmp_path / "does-not-exist"),
        ])

        assert exit_code != 0
        assert not (tmp_path / "mcap_valid_reports").exists()

    def test_verbose_table_shows_full_action_label_not_swallowed_by_rich_markup(self, tmp_path, monkeypatch, capsys):
        # Regression test for Rich markup swallowing "[left]"/"[right]" out of
        # "action[left]"/"action[right]" labels when embedded unescaped in a
        # markup f-string. Config-free classification can no longer force a
        # synthetic bimanual (both action[left] AND action[right]) scenario the
        # way the old config-based version of this test did — there's no config
        # left to declare a fictional second arm. The real single-arm fixture's
        # own action topic still classifies to "action[right]" (see
        # TestScanEpisodeIntegration.test_every_topic_is_classified_...), which
        # is sufficient to exercise the actual regression this test guards
        # against: that the "[right]" bracket doesn't get silently swallowed as
        # Rich markup. This narrows the original two-arm coverage to a single
        # arm; disclosed here as an accepted narrowing of this regression test.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--verbose",
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "action[right]" in captured.out

    def test_topic_flag_dumps_field_structure_in_table_mode(self, tmp_path, monkeypatch, capsys):
        # Folds in the old standalone `mcap-inspect` tool: --topic triggers a deep
        # per-message field-structure dump for that one topic.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--topic", "/joint_states",
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "position" in captured.out
        assert "name" in captured.out
        # Regression: field types like "List[float]" must not be swallowed by Rich
        # markup (the "[float]" part would otherwise be parsed as a markup tag and
        # silently dropped), same class of bug as the "action[left]" escaping above.
        assert "List[float]" in captured.out

    def test_topic_flag_with_json_format_includes_topic_structure_alongside_episodes(
        self, tmp_path, monkeypatch, capsys
    ):
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--format", "json",
            "--topic", "/joint_states",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert exit_code == 0
        assert "episodes" in payload
        assert "topic_structure" in payload
        assert "/joint_states" in payload["topic_structure"]
        assert "position" in payload["topic_structure"]["/joint_states"]["fields"]

    def test_json_format_without_topic_flag_has_no_topic_structure_key(self, tmp_path, monkeypatch, capsys):
        # The --topic addition must not change existing --format json behavior
        # when --topic isn't given.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        exit_code = main([
            "-i", str(_STUB_MCAP),
            "--format", "json",
        ])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert exit_code == 0
        assert "topic_structure" not in payload

    def test_topics_baseline_table_lists_all_fixture_topics_with_types(self, tmp_path, monkeypatch, capsys):
        # New "what's in this file" table (folded in from the old mcap-inspect
        # topic summary): every topic in a representative episode, regardless
        # of severity, with its message type and role.
        #
        # This table now renders through the same terminal-width-aware `console`
        # as everything else (see test_topics_baseline_table_respects_narrow_terminal_width
        # below for the narrow-width behavior), rather than a fixed 200-column
        # console. Under pytest, stdout isn't a real terminal, so Rich falls back
        # to an 80-column default unless COLUMNS says otherwise; simulate a wide
        # real terminal here so the full topic/type strings are asserted the same
        # way a user on a wide terminal would actually see them.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COLUMNS", "200")
        exit_code = main([
            "-i", str(_STUB_MCAP),
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Topics in" in captured.out
        assert "/joint_states" in captured.out
        assert "/cam_chest/image_raw/compressed" in captured.out
        assert "/cam_waist/image_raw/compressed" in captured.out
        assert "/cam_wrist_r/image_raw/compressed" in captured.out
        assert "/follower_r_forward_position_controller/commands" in captured.out
        assert "sensor_msgs/JointState" in captured.out
        assert "sensor_msgs/CompressedImage" in captured.out
        assert "std_msgs/Float64MultiArray" in captured.out

    def test_topics_baseline_table_respects_narrow_terminal_width(self, tmp_path, monkeypatch, capsys):
        # Regression test for the fixed Console(width=200) that used to back this
        # table: it always rendered at exactly 200 columns regardless of the real
        # terminal width, causing a jarring width mismatch against every other
        # table in the same output (which correctly adapts to the real width).
        # The table must now shrink/truncate consistently with the rest of the
        # output on a narrow terminal, and must not crash while doing so.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COLUMNS", "60")
        exit_code = main([
            "-i", str(_STUB_MCAP),
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Topics in" in captured.out
        # No line in the rendered output should exceed the narrow terminal width
        # (plus a small allowance for Rich's own box-drawing edge characters);
        # this is the actual regression the old fixed-width console produced.
        for line in captured.out.splitlines():
            assert len(line) <= 60 + 2

    def test_topic_flag_on_corrupt_file_errors_cleanly_without_traceback(self, tmp_path, monkeypatch, capsys):
        # inspect_message_structure() deliberately lets I/O/parse errors (e.g.
        # mcap.exceptions.InvalidMagic for a corrupt file) propagate uncaught —
        # main() must catch them itself and report a clean error, the same way
        # scan_episode()'s callers turn a read error into a report instead of a
        # crash, rather than letting a raw traceback reach the user.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        garbage_file = tmp_path / "corrupt.mcap"
        garbage_file.write_bytes(b"not a real mcap file")

        exit_code = main([
            "-i", str(garbage_file),
            "--topic", "/joint_states",
        ])

        captured = capsys.readouterr()
        assert exit_code != 0
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err

    def test_topics_baseline_table_skipped_for_empty_directory(self, tmp_path, monkeypatch, capsys):
        # No .mcap files at all -> nothing to build a baseline table from;
        # must be gracefully skipped, not crash or render misleading content.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        empty_dir = tmp_path / "empty-session"
        empty_dir.mkdir()

        exit_code = main([
            "-i", str(empty_dir),
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Topics in" not in captured.out

    def test_topics_baseline_table_skipped_when_all_reports_are_read_errors(self, tmp_path, monkeypatch, capsys):
        # Every file in the batch is corrupt/unreadable -> no representative
        # episode exists to build the baseline table from; must be gracefully
        # skipped rather than rendering an empty/misleading table.
        from mcap_converter.cli.mcap_valid import main

        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "all-corrupt-session"
        session_dir.mkdir()
        (session_dir / "a.mcap").write_bytes(b"not a real mcap file")
        (session_dir / "b.mcap").write_bytes(b"also not a real mcap file")

        exit_code = main([
            "-i", str(session_dir),
        ])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Topics in" not in captured.out


class TestDefaultReportPaths:
    def test_directory_input_uses_directory_name(self, tmp_path):
        from mcap_converter.cli.mcap_valid import default_report_paths

        session_dir = tmp_path / "my-session"
        session_dir.mkdir()

        json_path, md_path = default_report_paths(session_dir)

        assert json_path.name == "report.json"
        assert md_path.name == "report.md"
        assert json_path.parent.name == "my-session"
        assert md_path.parent.name == "my-session"
        assert json_path.parent.parent.name == "mcap_valid_reports"
        assert md_path.parent.parent.name == "mcap_valid_reports"

    def test_file_input_uses_stem_without_extension(self, tmp_path):
        from mcap_converter.cli.mcap_valid import default_report_paths

        mcap_file = tmp_path / "recording.mcap"
        mcap_file.write_bytes(b"")

        json_path, md_path = default_report_paths(mcap_file)

        assert json_path.name == "report.json"
        assert md_path.name == "report.md"
        assert json_path.parent.name == "recording"
        assert md_path.parent.name == "recording"

    def test_uses_cwd_not_input_parent(self, tmp_path, monkeypatch):
        from mcap_converter.cli.mcap_valid import default_report_paths

        # Input lives under a directory that is NOT the cwd we chdir into.
        input_parent = tmp_path / "elsewhere"
        input_parent.mkdir()
        session_dir = input_parent / "my-session"
        session_dir.mkdir()

        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        monkeypatch.chdir(cwd_dir)

        json_path, md_path = default_report_paths(session_dir)

        # mcap_valid_reports/ is created relative to cwd, not relative to
        # the input path's own parent directory.
        assert json_path.parent.parent.parent == cwd_dir
        assert md_path.parent.parent.parent == cwd_dir


class TestRenderMarkdownReport:
    @staticmethod
    def _make_reports():
        healthy = EpisodeQualityReport(
            path="/data/raw/session/0001/0001_0.mcap",
            duration_s=12.5,
            severity=SEVERITY_OK,
            passed=True,
            topics=[
                TopicQualityReport(
                    topic="/joint_states",
                    label="joint_states",
                    role="stream",
                    message_count=9897,
                    avg_fps=499.9,
                    coverage_ratio=1.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    severity=SEVERITY_OK,
                    reason="OK",
                ),
                TopicQualityReport(
                    topic="/camera/image_raw",
                    label="camera",
                    role="stream",
                    message_count=300,
                    avg_fps=30.0,
                    coverage_ratio=1.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    severity=SEVERITY_OK,
                    reason="OK",
                ),
                TopicQualityReport(
                    topic="/follower_position_controller/commands",
                    label="action",
                    role="action",
                    message_count=300,
                    avg_fps=None,
                    coverage_ratio=1.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    severity=SEVERITY_OK,
                    reason="OK",
                ),
            ],
        )
        warning = EpisodeQualityReport(
            path="/data/raw/session/0002/0002_0.mcap",
            duration_s=8.2,
            severity=SEVERITY_WARNING,
            passed=True,
            topics=[
                TopicQualityReport(
                    topic="/follower_position_controller/commands",
                    label="action",
                    role="action",
                    message_count=250,
                    avg_fps=None,
                    coverage_ratio=0.95,
                    total_gap_s=5.61,
                    longest_gap_s=5.61,
                    gaps=[GapInterval(start_s=1.0, end_s=6.61, duration_s=5.61, kind="idle")],
                    severity=SEVERITY_WARNING,
                    reason="1 個 idle gap，最長 5.61s（正常，手臂未操作）",
                ),
            ],
        )
        corrupt = EpisodeQualityReport(
            path="/data/raw/session/0003/0003_0.mcap",
            duration_s=0.0,
            severity=SEVERITY_CRITICAL,
            passed=False,
            topics=[],
            read_error="InvalidMagic: not a valid mcap file",
        )
        return [healthy, warning, corrupt]

    @staticmethod
    def _make_reports_with_readable_critical():
        """3 readable episodes (no read_error) spanning ok / warning / critical.

        Used for scenarios the 3-episode `_make_reports()` fixture can't cover on
        its own: a critical severity that comes from a genuinely-scanned episode
        (not a read_error stand-in), together with a warning in a sibling
        episode, so the Flagged Topics critical-before-warning sort order has
        two distinct groups to actually sort.
        """
        ok_ep = EpisodeQualityReport(
            path="/data/raw/session/0004/0004_0.mcap",
            duration_s=10.0,
            severity=SEVERITY_OK,
            passed=True,
            topics=[
                TopicQualityReport(
                    topic="/joint_states",
                    label="joint_states",
                    role="stream",
                    message_count=1000,
                    avg_fps=100.0,
                    coverage_ratio=1.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    severity=SEVERITY_OK,
                    reason="OK",
                ),
            ],
        )
        warning_ep = EpisodeQualityReport(
            path="/data/raw/session/0005/0005_0.mcap",
            duration_s=8.0,
            severity=SEVERITY_WARNING,
            passed=True,
            topics=[
                TopicQualityReport(
                    topic="/follower_position_controller/commands",
                    label="action",
                    role="action",
                    message_count=100,
                    avg_fps=None,
                    coverage_ratio=0.9,
                    total_gap_s=2.0,
                    longest_gap_s=2.0,
                    severity=SEVERITY_WARNING,
                    reason="1 個 idle gap，最長 2.00s（正常，手臂未操作）",
                ),
            ],
        )
        critical_ep = EpisodeQualityReport(
            path="/data/raw/session/0006/0006_0.mcap",
            duration_s=5.0,
            severity=SEVERITY_CRITICAL,
            passed=False,
            topics=[
                TopicQualityReport(
                    topic="/joint_states",
                    label="joint_states",
                    role="stream",
                    message_count=1,
                    avg_fps=None,
                    coverage_ratio=0.0,
                    total_gap_s=5.0,
                    longest_gap_s=5.0,
                    severity=SEVERITY_CRITICAL,
                    reason="stream 僅 1 則訊息，幾乎沒錄到",
                ),
            ],
        )
        return [ok_ep, warning_ep, critical_ep]

    def test_every_episode_gets_a_header(self):
        # Per-episode headers are "### `<filename>` — ...", uniquely
        # identifiable by the backtick right after "### " — the new
        # Cross-Episode Comparison / Conclusion sub-headings (### Severity,
        # ### Avg FPS Trend, ### Flagged Topics) are also "### "-prefixed
        # (note: bumping them to "#### " would NOT avoid a naive
        # `output.count("### ")` check, since "#### " contains "### " as a
        # substring), so this test must key off the backtick to stay scoped
        # to episode headers specifically.
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        assert output.count("### `") == 3

    def test_every_topic_appears_in_a_table_row(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        assert "/joint_states" in output
        assert "/camera/image_raw" in output
        assert output.count("/follower_position_controller/commands") == 2  # healthy + warning episodes

    def test_type_column_appears_with_real_message_type_values(self):
        # New "Type" column (between Label and Role), populated from a real
        # scan_episode() report against the smoke-test fixture.
        from mcap_converter.cli.mcap_valid import render_markdown_report

        report = scan_episode(str(_STUB_MCAP), QualityThresholds())
        output = render_markdown_report([report], input_path="fake/input")

        assert "| Topic | Label | Type | Role |" in output
        assert "sensor_msgs/JointState" in output
        assert "sensor_msgs/CompressedImage" in output
        assert "std_msgs/Float64MultiArray" in output

    def test_read_error_episode_shows_error_and_no_topics_table(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        assert "**Read error:**" in output
        assert "InvalidMagic" in output

    def test_summary_line_matches_render_table_wording(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        assert "3 episodes: 1 ok, 1 warning, 0 critical, 1 unreadable" in output

    def test_output_is_plain_markdown_with_no_rich_markup(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        for tag in ("[green]", "[/green]", "[yellow]", "[/yellow]", "[red]", "[/red]"):
            assert tag not in output

    def test_empty_reports_list_produces_valid_document_without_crashing(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        output = render_markdown_report([], input_path="some/input")

        assert isinstance(output, str)
        assert "# mcap-valid Report" in output
        assert "0 episodes: 0 ok, 0 warning, 0 critical" in output
        assert "### " not in output

    def test_summary_line_is_byte_for_byte_identical_to_render_table(self, capsys):
        from mcap_converter.cli.mcap_valid import _render_table, render_markdown_report

        reports = self._make_reports()

        _render_table(reports, verbose=False)
        table_output = capsys.readouterr().out
        table_summary = [line for line in table_output.splitlines() if line.strip()][-1]

        md_output = render_markdown_report(reports, input_path="fake/input")
        md_summary = next(line for line in md_output.splitlines() if line.startswith(f"{len(reports)} episodes:"))

        assert table_summary == md_summary

    def test_cross_episode_comparison_present_with_two_readable_episodes(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()  # healthy + warning readable, corrupt unreadable
        output = render_markdown_report(reports, input_path="fake/input")

        assert "## Cross-Episode Comparison" in output
        assert "### Severity" in output
        assert "### Avg FPS Trend" in output
        # Must sit between Summary and Episodes.
        assert output.index("## Summary") < output.index("## Cross-Episode Comparison")
        assert output.index("## Cross-Episode Comparison") < output.index("## Episodes")

    def test_severity_matrix_has_correct_cells(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()[:2]  # healthy + warning, both readable
        output = render_markdown_report(reports, input_path="fake/input")

        severity_section = output.split("### Severity")[1].split("### Avg FPS Trend")[0]
        table_rows = {
            line.split("|")[1].strip(): line
            for line in severity_section.splitlines()
            if line.startswith("|")
        }
        assert table_rows["Topic"] == "| Topic | 0001_0.mcap | 0002_0.mcap |"
        # camera/joint_states only exist in the healthy episode -> "-" for warning.
        assert table_rows["camera"] == "| camera | 🟢 | - |"
        assert table_rows["joint_states"] == "| joint_states | 🟢 | - |"
        # action exists (ok) in healthy and (warning) in the warning episode.
        assert table_rows["action"] == "| action | 🟢 | 🟡 |"

    def test_fps_trend_excludes_label_with_no_numeric_fps_in_any_episode(self):
        # "action" never has an avg_fps (role="action" is event-driven, never
        # gets a computed rate) -> it must be excluded from the trend table
        # even though it's a normal row in the severity matrix.
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()[:2]
        output = render_markdown_report(reports, input_path="fake/input")

        fps_section = output.split("### Avg FPS Trend")[1].split("## Episodes")[0]
        assert "action" not in fps_section
        assert "joint_states" in fps_section
        assert "camera" in fps_section

    def test_comparison_section_omitted_with_a_single_readable_episode(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()[:1]  # just the healthy episode
        output = render_markdown_report(reports, input_path="fake/input")

        assert "## Cross-Episode Comparison" not in output

    def test_unreadable_episode_excluded_from_comparison_but_shown_in_conclusion(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()
        output = render_markdown_report(reports, input_path="fake/input")

        comparison_section = output.split("## Cross-Episode Comparison")[1].split("## Episodes")[0]
        assert "0003_0.mcap" not in comparison_section

        conclusion_section = output.split("## Conclusion")[1]
        assert "- ⚠ 0003_0.mcap: InvalidMagic: not a valid mcap file" in conclusion_section

    def test_conclusion_not_ready_verdict_for_a_genuinely_readable_critical_episode(self):
        # Distinct from a read_error-driven critical: this critical severity
        # comes from an actually-scanned episode's topic.
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports_with_readable_critical()
        output = render_markdown_report(reports, input_path="fake/input")

        conclusion_section = output.split("## Conclusion")[1]
        assert "❌ Not ready to convert" in conclusion_section
        assert "1/3 episode(s) flagged critical" in conclusion_section

    def test_conclusion_warning_verdict_when_worst_severity_is_warning(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports()[:2]  # healthy (ok) + warning, no criticals
        output = render_markdown_report(reports, input_path="fake/input")

        conclusion_section = output.split("## Conclusion")[1]
        assert "⚠️ Ready to convert with warnings" in conclusion_section
        assert "1/2 episode(s) have non-blocking issues" in conclusion_section

    def test_flagged_topics_groups_by_label_and_severity_sorted_critical_first(self):
        from mcap_converter.cli.mcap_valid import render_markdown_report

        reports = self._make_reports_with_readable_critical()
        output = render_markdown_report(reports, input_path="fake/input")

        flagged_section = output.split("### Flagged Topics")[1]
        assert flagged_section.index("joint_states") < flagged_section.index("action")
        assert (
            "🔴 **joint_states**: critical in 1/3 episode(s) (0006_0.mcap) — "
            'e.g. "stream 僅 1 則訊息，幾乎沒錄到"' in flagged_section
        )
        assert (
            "🟡 **action**: warning in 1/3 episode(s) (0005_0.mcap) — "
            'e.g. "1 個 idle gap，最長 2.00s（正常，手臂未操作）"' in flagged_section
        )
