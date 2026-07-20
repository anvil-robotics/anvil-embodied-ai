"""Unit tests for SequenceStalenessGuard (no ROS/rclpy dependency).

lerobot_control is a ROS2 ament_python package, not a uv workspace member —
rclpy isn't installed in this venv, so only rclpy-free modules from it are
importable here. This mirrors gt_replay_correctness_test.py's sys.path
convention for reaching lerobot_control.dataset_reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))

from lerobot_control.ee_obs_sequence_guard import SequenceStalenessGuard  # noqa: E402


def test_first_read_is_never_stale():
    guard = SequenceStalenessGuard()
    is_stale, streak = guard.check("left", 1)
    assert is_stale is False
    assert streak == 0


def test_stationary_arm_with_advancing_sequence_is_not_stale():
    """Identical pose across ticks is NOT staleness — only sequence matters.

    The guard only ever sees the sequence number (never the pose), so a
    stationary arm is exercised here simply by feeding a strictly-increasing
    sequence repeatedly and confirming none of it is flagged.
    """
    guard = SequenceStalenessGuard()
    for seq in range(1, 21):
        is_stale, streak = guard.check("left", seq)
        assert is_stale is False, f"seq={seq} incorrectly flagged stale"
        assert streak == 0


def test_repeated_sequence_is_stale():
    guard = SequenceStalenessGuard()
    guard.check("left", 1)
    guard.check("left", 5)  # genuine advance -> ends the warm-up grace period
    is_stale, streak = guard.check("left", 5)
    assert is_stale is True
    assert streak == 1


def test_non_advancing_sequence_is_stale_even_if_lower():
    """A sequence that regresses (not just repeats) is also stale."""
    guard = SequenceStalenessGuard()
    guard.check("left", 1)
    guard.check("left", 10)  # genuine advance -> ends the warm-up grace period
    is_stale, streak = guard.check("left", 7)
    assert is_stale is True
    assert streak == 1


def test_stale_streak_accumulates_and_resets_on_recovery():
    guard = SequenceStalenessGuard()
    guard.check("left", 1)
    guard.check("left", 2)  # genuine advance -> ends the warm-up grace period
    _, streak1 = guard.check("left", 2)
    _, streak2 = guard.check("left", 2)
    assert (streak1, streak2) == (1, 2)

    # A fresh advancing sequence resets the streak.
    is_stale, streak = guard.check("left", 3)
    assert is_stale is False
    assert streak == 0

    is_stale, streak = guard.check("left", 3)
    assert is_stale is True
    assert streak == 1


def test_arms_are_tracked_independently():
    guard = SequenceStalenessGuard()
    guard.check("left", 5)
    guard.check("right", 100)
    guard.check("right", 101)  # genuine advance -> ends right's warm-up grace period

    # Repeating "right"'s sequence must not be affected by "left"'s state.
    is_stale, streak = guard.check("right", 101)
    assert is_stale is True
    assert streak == 1

    # "left" should still accept its own next (genuinely advancing) sequence
    # normally, unaffected by right's staleness.
    is_stale, streak = guard.check("left", 6)
    assert is_stale is False
    assert streak == 0


def test_is_fault_fires_exactly_once_at_threshold():
    guard = SequenceStalenessGuard(stale_fault_threshold=3)
    guard.check("left", 1)
    guard.check("left", 2)  # genuine advance -> ends the warm-up grace period

    streaks = []
    for _ in range(5):
        _, streak = guard.check("left", 2)
        streaks.append(streak)
    assert streaks == [1, 2, 3, 4, 5]

    faults = [guard.is_fault(s) for s in streaks]
    # Fires only at the exact crossing point, not before or repeatedly after.
    assert faults == [False, False, True, False, False]


def test_invalid_threshold_rejected():
    import pytest

    with pytest.raises(ValueError):
        SequenceStalenessGuard(stale_fault_threshold=0)


def test_degraded_after_streak_below_fault_threshold_rejected():
    import pytest

    with pytest.raises(ValueError):
        SequenceStalenessGuard(stale_fault_threshold=10, degraded_after_streak=5)


def test_peer_that_never_advances_at_all_is_never_flagged_or_degraded():
    """Warm-up grace period: a peer that has NEVER genuinely advanced past
    its first-ever reading (e.g. the mock hasn't processed its first command
    yet, still mid container-startup) must not be flagged stale or degraded,
    no matter how long this persists. This is indistinguishable from "hasn't
    started yet," which is expected and can legitimately take many seconds
    (image-worker spawn, DDS discovery) at the mock's 100Hz echo rate — found
    live: without this exemption, ~12s of startup at 100Hz accumulates far
    more than degraded_after_streak stale reads before any real motion even
    begins, permanently degrading the guard before it ever protects anything.
    Only a peer that HAS genuinely advanced at least once and THEN gets stuck
    is treated as a real fault (see
    test_recovering_sequence_does_not_prevent_degradation_math)."""
    guard = SequenceStalenessGuard(stale_fault_threshold=3, degraded_after_streak=5)
    guard.check("left", 0)  # first read, always accepted as baseline

    results = [guard.check("left", 0) for _ in range(1000)]
    assert all(is_stale is False for is_stale, _ in results)
    assert all(streak == 0 for _, streak in results)
    assert guard.is_degraded("left") is False


def test_just_degraded_fires_exactly_once():
    guard = SequenceStalenessGuard(stale_fault_threshold=2, degraded_after_streak=4)
    guard.check("left", 1)
    guard.check("left", 2)  # genuine advance -> ends the warm-up grace period

    streaks = [guard.check("left", 2)[1] for _ in range(6)]
    assert streaks == [1, 2, 3, 4, 0, 0]

    just_degraded = [guard.just_degraded(s) for s in streaks]
    assert just_degraded == [False, False, False, True, False, False]


def test_recovering_sequence_does_not_prevent_degradation_math():
    """A peer that advances for a while, then gets stuck, still degrades based
    on its OWN stuck streak — recovery before getting stuck doesn't matter."""
    guard = SequenceStalenessGuard(stale_fault_threshold=2, degraded_after_streak=3)
    for seq in range(1, 11):
        guard.check("left", seq)  # advances cleanly, never stale
    assert guard.is_degraded("left") is False

    # Now it gets stuck at seq=10 forever.
    for _ in range(3):
        guard.check("left", 10)
    assert guard.is_degraded("left") is True


def test_degradation_is_per_arm_independent():
    guard = SequenceStalenessGuard(stale_fault_threshold=2, degraded_after_streak=3)
    guard.check("left", 0)
    guard.check("right", 0)
    guard.check("left", 1)  # left genuinely advances once...

    for _ in range(3):
        guard.check("left", 1)  # ...then gets stuck -> degrades

    assert guard.is_degraded("left") is True
    assert guard.is_degraded("right") is False

    # right's own advancing sequence is completely unaffected by left's fallback.
    is_stale, streak = guard.check("right", 1)
    assert is_stale is False
    assert streak == 0


def test_last_accepted_does_not_conflate_never_accepted_with_degraded():
    guard = SequenceStalenessGuard(stale_fault_threshold=2, degraded_after_streak=3)
    assert guard.last_accepted("left") is None  # never accepted yet

    guard.check("left", 0)
    guard.check("left", 1)  # genuine advance -> ends the warm-up grace period
    for _ in range(3):
        guard.check("left", 1)  # then gets stuck -> degrades
    assert guard.is_degraded("left") is True
    # last_accepted still reports the last real value it saw -- callers that
    # need "don't gate on this" must check is_degraded separately, exactly
    # as get_ee_obs_sequence_snapshot() does.
    assert guard.last_accepted("left") == 1
