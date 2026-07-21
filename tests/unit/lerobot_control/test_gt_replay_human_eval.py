"""Unit tests for scripts/gt_replay_human_eval.py's pure logic.

Covers episode-spec-independent logic only (that's tested in test_dataset_reader.py):
timeout resolution, sentinel classification, the operator prompt (stdin-monkeypatched,
mirroring mcap_converter's test_migrate_config.py precedent), and report building. No
docker/rclpy dependency — none of this touches the orchestration functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import gt_replay_human_eval as human_eval  # noqa: E402


# --------------------------------------------------------------------------- #
# resolve_timeout_sec
# --------------------------------------------------------------------------- #


def test_resolve_timeout_sec_explicit_override_wins():
    assert human_eval.resolve_timeout_sec(
        explicit=42.0, target="real", n_frames=1000, control_frequency=30.0,
        homing_enabled=True, homing_timeout_sec=30.0,
    ) == 42.0 + 30.0  # homing is still additive even with an explicit override


def test_resolve_timeout_sec_scales_with_nominal_duration():
    # nominal = 300 / 30 = 10s; fake profile is (1.5, 25.0) -> 10*1.5 + 25 = 40
    got = human_eval.resolve_timeout_sec(
        explicit=None, target="fake", n_frames=300, control_frequency=30.0,
        homing_enabled=False, homing_timeout_sec=30.0,
    )
    assert got == 40.0


def test_resolve_timeout_sec_real_gets_larger_allowance_than_fake():
    kwargs = dict(explicit=None, n_frames=300, control_frequency=30.0, homing_enabled=False, homing_timeout_sec=30.0)
    fake_timeout = human_eval.resolve_timeout_sec(target="fake", **kwargs)
    real_timeout = human_eval.resolve_timeout_sec(target="real", **kwargs)
    assert real_timeout > fake_timeout


def test_resolve_timeout_sec_homing_is_additive():
    kwargs = dict(explicit=None, target="real", n_frames=300, control_frequency=30.0, homing_timeout_sec=15.0)
    without_homing = human_eval.resolve_timeout_sec(homing_enabled=False, **kwargs)
    with_homing = human_eval.resolve_timeout_sec(homing_enabled=True, **kwargs)
    assert with_homing == without_homing + 15.0


# --------------------------------------------------------------------------- #
# classify_signal — the three-way orthogonal (homing_status, replay_status)
# --------------------------------------------------------------------------- #


def test_classify_signal_complete():
    homing, replay = human_eval.classify_signal(
        {"status": "complete", "homing_status": "confirmed"}, container_exited=False, timed_out=False,
    )
    assert (homing, replay) == ("confirmed", "completed")


def test_classify_signal_complete_with_skipped_homing():
    homing, replay = human_eval.classify_signal(
        {"status": "complete", "homing_status": "skipped"}, container_exited=False, timed_out=False,
    )
    assert (homing, replay) == ("skipped", "completed")


def test_classify_signal_homing_failed_never_attempts_replay():
    """The core distinction the user asked not to conflate."""
    homing, replay = human_eval.classify_signal(
        {"status": "homing_failed", "homing_status": "failed"}, container_exited=False, timed_out=False,
    )
    assert homing == "failed"
    assert replay == "not_attempted"


def test_classify_signal_interrupted_is_crashed():
    homing, replay = human_eval.classify_signal(
        {"status": "interrupted", "homing_status": "confirmed"}, container_exited=False, timed_out=False,
    )
    assert (homing, replay) == ("confirmed", "crashed")


def test_classify_signal_no_signal_container_exited_is_crashed():
    homing, replay = human_eval.classify_signal(None, container_exited=True, timed_out=False)
    assert (homing, replay) == (None, "crashed")


def test_classify_signal_no_signal_timeout_elapsed():
    homing, replay = human_eval.classify_signal(None, container_exited=False, timed_out=True)
    assert (homing, replay) == (None, "timed_out")


# --------------------------------------------------------------------------- #
# episode_needs_saved_logs — anything short of a clean pass gets its
# container logs kept for post-mortem debugging
# --------------------------------------------------------------------------- #


def test_episode_needs_saved_logs_clean_pass_does_not_save():
    assert human_eval.episode_needs_saved_logs(
        "completed", "pass", {"all_passed": True, "max_pos_err_m": 0.0, "max_rot_err_deg": 0.0},
    ) is False


def test_episode_needs_saved_logs_no_auto_verify_clean_pass_does_not_save():
    """--target real never has auto_verify (always None) -- must not be
    mistaken for a failure on its own."""
    assert human_eval.episode_needs_saved_logs("completed", "pass", None) is False


def test_episode_needs_saved_logs_homing_failure_saves():
    """The exact scenario that motivated this: replay never even attempted."""
    assert human_eval.episode_needs_saved_logs("not_attempted", None, None) is True


def test_episode_needs_saved_logs_timed_out_saves():
    assert human_eval.episode_needs_saved_logs("timed_out", None, None) is True


def test_episode_needs_saved_logs_crashed_saves():
    assert human_eval.episode_needs_saved_logs("crashed", None, None) is True


def test_episode_needs_saved_logs_operator_fail_saves():
    assert human_eval.episode_needs_saved_logs(
        "completed", "fail", {"all_passed": True, "max_pos_err_m": 0.0, "max_rot_err_deg": 0.0},
    ) is True


def test_episode_needs_saved_logs_auto_verify_fail_saves():
    assert human_eval.episode_needs_saved_logs(
        "completed", "pass", {"all_passed": False, "max_pos_err_m": 0.05, "max_rot_err_deg": 3.0},
    ) is True


# --------------------------------------------------------------------------- #
# summarize_auto_verify — fake-target-only extra signal from
# gt_replay_verifier_node's own numeric-tolerance report
# --------------------------------------------------------------------------- #


def test_summarize_auto_verify_none_when_no_report():
    """--target real (or a crashed/timed-out verifier) never has a report."""
    assert human_eval.summarize_auto_verify(None) is None


def test_summarize_auto_verify_extracts_max_error_across_arms():
    report = {
        "all_passed": False,
        "arms": {
            "left": {"max_pos_err_m": 0.002, "max_rot_err_deg": 0.3},
            "right": {"max_pos_err_m": 0.0001, "max_rot_err_deg": 0.9},
        },
    }
    got = human_eval.summarize_auto_verify(report)
    assert got == {"all_passed": False, "max_pos_err_m": 0.002, "max_rot_err_deg": 0.9}


def test_summarize_auto_verify_all_passed_true():
    report = {
        "all_passed": True,
        "arms": {"left": {"max_pos_err_m": 0.0, "max_rot_err_deg": 0.0}},
    }
    got = human_eval.summarize_auto_verify(report)
    assert got["all_passed"] is True


# --------------------------------------------------------------------------- #
# prompt_operator_verdict — foreground input(), monkeypatched (mirrors
# mcap_converter's migrate_config.py test precedent)
# --------------------------------------------------------------------------- #


def test_prompt_operator_verdict_pass(monkeypatch):
    answers = iter(["y", "all good"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    verdict, comment = human_eval.prompt_operator_verdict(0, rows_replayed=100, elapsed_sec=5.0)
    assert verdict == "pass"
    assert comment == "all good"


def test_prompt_operator_verdict_fail_with_empty_comment(monkeypatch):
    answers = iter(["n", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    verdict, comment = human_eval.prompt_operator_verdict(1, rows_replayed=50, elapsed_sec=3.0)
    assert verdict == "fail"
    assert comment == ""


def test_prompt_operator_verdict_reprompts_on_unrecognized_input(monkeypatch):
    """No safe default for pass/fail -- must re-ask, not silently pick one."""
    answers = iter(["maybe", "asdf", "yes", "comment text"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    verdict, comment = human_eval.prompt_operator_verdict(2, rows_replayed=10, elapsed_sec=1.0)
    assert verdict == "pass"
    assert comment == "comment text"


# --------------------------------------------------------------------------- #
# build_report — pass_rate math and the three-way status fields
# --------------------------------------------------------------------------- #


def test_build_report_pass_rate_over_completed_only():
    records = [
        human_eval.build_episode_record(0, "confirmed", "completed", "pass", "", "t0"),
        human_eval.build_episode_record(1, "confirmed", "completed", "fail", "dropped it", "t1"),
        human_eval.build_episode_record(2, "failed", "not_attempted", None, None, "t2"),
        human_eval.build_episode_record(3, "confirmed", "timed_out", None, None, "t3"),
    ]
    report = human_eval.build_report("ds", "real", "0:4", records, "start", "end")
    s = report["summary"]
    assert s["n_total"] == 4
    assert s["n_completed_replay"] == 2
    assert s["n_failed_to_replay"] == 2
    assert s["n_operator_pass"] == 1
    assert s["n_operator_fail"] == 1
    # pass_rate divides by n_completed_replay (2), NOT n_total (4) -- the
    # homing failure and the timeout must not silently worsen the rate.
    assert s["pass_rate"] == 0.5
    assert s["n_homing_confirmed"] == 3
    assert s["n_homing_failed"] == 1


def test_build_report_pass_rate_null_when_nothing_completed():
    records = [human_eval.build_episode_record(0, "failed", "not_attempted", None, None, "t0")]
    report = human_eval.build_report("ds", "real", "0", records, "start", "end")
    assert report["summary"]["pass_rate"] is None


def test_build_report_auto_verify_null_for_real_target():
    """--target real never runs a verifier — auto_verify stays None on every
    episode, and the summary must not fabricate a 0/0 pass_rate for it."""
    records = [
        human_eval.build_episode_record(0, "confirmed", "completed", "pass", "", "t0"),
    ]
    report = human_eval.build_report("ds", "real", "0", records, "start", "end")
    assert report["episodes"][0]["auto_verify"] is None
    s = report["summary"]
    assert s["n_auto_verify_pass"] == 0
    assert s["n_auto_verify_fail"] == 0
    assert s["auto_verify_pass_rate"] is None


def test_build_report_auto_verify_pass_rate_over_fake_episodes():
    records = [
        human_eval.build_episode_record(
            0, "skipped", "completed", "pass", "", "t0",
            auto_verify={"all_passed": True, "max_pos_err_m": 0.0001, "max_rot_err_deg": 0.01},
        ),
        human_eval.build_episode_record(
            1, "skipped", "completed", "pass", "", "t1",
            auto_verify={"all_passed": False, "max_pos_err_m": 0.05, "max_rot_err_deg": 3.0},
        ),
        # Timed out before the verifier ever finalized -- auto_verify is None,
        # not counted as a fail.
        human_eval.build_episode_record(2, "skipped", "timed_out", None, None, "t2", auto_verify=None),
    ]
    report = human_eval.build_report("ds", "fake", "0:3", records, "start", "end")
    s = report["summary"]
    assert s["n_auto_verify_pass"] == 1
    assert s["n_auto_verify_fail"] == 1
    assert s["auto_verify_pass_rate"] == 0.5
