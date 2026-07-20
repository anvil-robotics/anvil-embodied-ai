#!/usr/bin/env python3
"""Sequence-based staleness detection for the mock's /ee_pose_{arm} echo.

Fake-hardware-only. This mechanism exists to work around a bug specific to
test/fake_hardware/fake_hardware_node.py's echo: a pure-software echo with no
independent physical sensor driving it, which can occasionally skip a cycle
under load. A real robot's pose is driven by independent physical sensors
(FK from /joint_states, or an equivalent direct measurement) and structurally
cannot exhibit this failure mode — there is no "echo" step to skip. For that
reason, real hardware never constructs a ``SequenceStalenessGuard`` instance
at all (see strategies/multi_process.py's ``mock_ee_pose_echo`` gate); this is
not a general real-hardware safety feature that happens to be dormant there,
it simply does not apply.

Pure logic, no ROS/rclpy dependency — see strategies/multi_process.py's
``_make_mock_ee_cb`` for the integration point, and MockEEPose.msg's
``sequence`` field for the wire format (a fake-hardware-only message type;
real hardware's CommandedEEPose.msg has no such field and never will — see
that message's docstring for the wire-compatibility reasoning).

Why sequence, not header.stamp: wall-clock timestamps can be non-monotonic
across clock corrections, NTP sync, or cross-thread/cross-process jitter, and
only answer "when did this happen" — not "is this strictly newer than the
last message I consumed." A plain monotonically-incrementing counter answers
the latter directly, with zero dependency on clock behavior.

Graceful degradation, purely as a defensive fallback against a *mock* bug:
in the fake-hardware-only path where this guard runs, a sequence value that
never advances is expected to mean the mock itself has stopped incrementing
(e.g. a bug in fake_hardware_node.py), not a legitimate peer state — there is
no real-hardware scenario this needs to gracefully handle, since real
hardware never reaches this code path in the first place. Left unhandled,
a stuck sequence would flag every read after the first as stale forever,
freezing the consumer's observation state permanently (see
claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md's
known-limitations note). A LONG streak of stale reads is nonetheless a fairly
reliable signal specifically for "this peer has stopped advancing sequence
entirely", as opposed to a real transient connection problem: a genuinely
flaky/degraded link is far more likely to show up as *missing* messages (no
calls to ``check()`` at all during the gap) than as an uninterrupted flood of
same-sequence ones. So once an arm's streak reaches ``degraded_after_streak``
(deliberately well above ``stale_fault_threshold``, so the fault is reported
first as an early warning), the guard gives up on that arm — sticky for the
life of this instance — and falls back to accepting every read
unconditionally, exactly restoring the pre-sequence-guard "keep only latest"
behavior for it. This never re-arms itself: once genuinely degraded, a peer
isn't expected to spontaneously start advancing again in the same session.
This shouldn't happen against a correctly-behaving mock — if it does,
something is wrong with the mock, not this consumer.

Warm-up grace period, separate from the degraded-fallback above: an arm that
has never yet genuinely advanced past its first-ever reading (i.e. the mock
hasn't processed its first command yet) is exempt from staleness/degradation
tracking entirely — every such read is accepted unconditionally and doesn't
count toward anything. This matters because the mock's echo runs on its own
fixed-rate timer (``ee_pose_fps``, default 100Hz) independent of when the
consuming node actually starts publishing commands — a realistic node
startup (spawning image-worker subprocesses, DDS discovery, etc.) can easily
take several seconds, during which the mock is still happily re-publishing
its *un-echoed* initial sequence value at 100Hz. Without this exemption, that
alone accumulates far more than ``degraded_after_streak`` "stale" reads
before any real motion even begins, permanently degrading the guard (see
``is_degraded``) before it ever had a chance to protect anything — a
one-time discovery from live verification, not a hypothetical. A peer that
has genuinely never advanced is indistinguishable from one that simply
hasn't started yet, so it would be wrong to penalize it; only a peer that HAS
advanced at least once and THEN gets stuck is treated as a real fault.
"""


class SequenceStalenessGuard:
    """Tracks the last-accepted ``sequence`` per arm and flags stale reads.

    A read is stale iff its sequence does not strictly exceed the last
    *accepted* sequence for that same arm. Value equality of the pose itself
    (a genuinely stationary arm) is never staleness — sequence still advances
    every publish regardless of whether the pose changed, so staleness must
    be judged purely on sequence non-advancement, never on the pose value.
    """

    def __init__(self, stale_fault_threshold: int = 10, degraded_after_streak: int = 50):
        if stale_fault_threshold < 1:
            raise ValueError("stale_fault_threshold must be >= 1")
        if degraded_after_streak < stale_fault_threshold:
            raise ValueError("degraded_after_streak must be >= stale_fault_threshold")
        self.stale_fault_threshold = stale_fault_threshold
        self.degraded_after_streak = degraded_after_streak
        self._last_seq: dict[str, int] = {}
        self._stale_streak: dict[str, int] = {}
        self._degraded: dict[str, bool] = {}
        self._ever_advanced: dict[str, bool] = {}

    def check(self, arm: str, sequence: int) -> tuple[bool, int]:
        """Record one incoming ``sequence`` for ``arm``.

        Returns ``(is_stale, stale_streak)``. ``stale_streak`` is the number
        of consecutive stale reads for this arm *including this one* (0 when
        not stale). The very first read for an arm is always accepted (there
        is nothing to compare against yet).

        Before ``arm`` has genuinely advanced past that first-ever reading
        (see module docstring's warm-up grace period), every read is also
        accepted unconditionally and never counted toward staleness or
        degradation — there's nothing to distinguish "hasn't started yet"
        from "stuck" until at least one genuine advance has been observed.

        Once ``arm`` has degraded (see ``is_degraded``), every read is
        accepted unconditionally — sequence is no longer trusted for this
        arm, so there is nothing left to check.
        """
        if self._degraded.get(arm, False):
            return False, 0

        last = self._last_seq.get(arm)
        if last is not None and sequence <= last:
            if not self._ever_advanced.get(arm, False):
                # Still warming up: this arm has never genuinely advanced,
                # so a repeat/regression here isn't staleness — it's "hasn't
                # started yet." Don't count it toward anything (see module
                # docstring's warm-up grace period).
                return False, 0
            streak = self._stale_streak.get(arm, 0) + 1
            self._stale_streak[arm] = streak
            if streak >= self.degraded_after_streak:
                self._degraded[arm] = True
            return True, streak

        if last is not None:
            # A genuine advance beyond the first-ever reading — the warm-up
            # grace period ends here; staleness/degradation tracking now
            # applies from this point on.
            self._ever_advanced[arm] = True
        self._last_seq[arm] = sequence
        self._stale_streak[arm] = 0
        return False, 0

    def is_fault(self, streak: int) -> bool:
        """True exactly when ``streak`` first reaches the fault threshold.

        Deliberately an equality check (not ``>=``) so a fault is reported
        once per degradation episode rather than spamming on every
        subsequent stale read while the streak keeps climbing.
        """
        return streak == self.stale_fault_threshold

    def just_degraded(self, streak: int) -> bool:
        """True exactly on the tick an arm gives up and falls back (see module
        docstring). Equality check, same one-shot shape as ``is_fault`` — the
        caller uses this to log the fallback transition exactly once, not on
        every subsequent (now-unconditionally-accepted) read.
        """
        return streak == self.degraded_after_streak

    def is_degraded(self, arm: str) -> bool:
        """True once ``arm`` has given up on ``sequence`` and fallen back.

        Sticky for the life of this guard instance — see module docstring.
        """
        return self._degraded.get(arm, False)

    def last_accepted(self, arm: str) -> int | None:
        """The last sequence accepted (not flagged stale) for ``arm``.

        ``None`` if no read has ever been accepted for this arm yet. Lets a
        downstream consumer (e.g. the ee_delta publish loop) tell whether the
        anchor has genuinely advanced since it last consumed it, distinct
        from "did the subscription callback merely run again." Callers that
        need to distinguish "never accepted" from "degraded, don't gate on
        this" should check ``is_degraded`` separately — this method doesn't
        conflate the two.
        """
        return self._last_seq.get(arm)
