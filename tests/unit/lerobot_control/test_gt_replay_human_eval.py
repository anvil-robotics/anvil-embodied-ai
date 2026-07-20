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
    # nominal = 300 / 30 = 10s; fake profile is (1.5, 10.0) -> 10*1.5 + 10 = 25
    got = human_eval.resolve_timeout_sec(
        explicit=None, target="fake", n_frames=300, control_frequency=30.0,
        homing_enabled=False, homing_timeout_sec=30.0,
    )
    assert got == 25.0


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
