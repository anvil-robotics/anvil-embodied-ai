# Make ee_delta fake-hardware correctness reliable + build a clear live per-tick console tracer

## Context

Real-hardware `ee_delta` GT-replay testing surfaced a genuine bug (delta compose racing ahead
of the robot's actual physical progress) that's now fixed via a position-proximity hold-gate
in `inference_node.py`'s `_publish_loop` (already implemented, committed, and confirmed to be
a no-op against the mock in two correctness-test runs). But the very next two fake-hardware
correctness-test runs for `ee_delta` then FAILED (both arms diverging at the identical tick —
the exact signature of the pre-existing, already-documented, never-fully-root-caused residual
flakiness described in `claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md` §15). A
third run with `DEBUG=true` passed cleanly. This is consistent with the known ~1-in-5-8 flaky
rate, not a new regression from the position-proximity change — but it's never actually been
root-caused, and Patrick wants it nailed down and fixed before trusting `ee_delta` further,
starting with making it pass reliably on the mock.

To debug this properly, Patrick wants a clear, live, per-tick console trace of the exact
control-flow state machine — printed to the terminal in real time (not recovered after the
fact via `docker logs`), at a user-configurable `control_frequency` (10 or 30 Hz) — showing,
for each tick `t`: the popped `action[t]` (delta), the `obs[t]` used to compose it, the
computed absolute command actually published to `/commanded_ee_{arm}`, and confirmation that
the mock's `/ee_pose_{arm}` echo reflects the robot having moved to that target. The existing
`--debug` logging (the `[ee_delta] ... PUBLISHED/HELD(seq)/HELD(pos)` line added this session)
is close but not quite this: no tick counter, no explicit "did it actually move" comparison,
and — critically — it's currently only recoverable via `docker logs`/the saved-log-on-failure
mechanism, not visible live while the test runs, because `gt_replay_correctness_test.py`
brings every container up detached (`-d`).

## Part 1 — Foreground streaming (no new tooling, just how the replay container is run)

`docker compose up -d` detaches the container; its stdout is invisible until you separately
`docker logs`/`compose logs -f` it. The fix is simply to run the **`replay`** service (the one
whose logs matter for this debugging) attached instead of detached, while `mock-robot` and
`gt-replay-verify` stay detached (their logs aren't what we're tracing). Both
`gt_replay_correctness_test.py` and `docker-compose.fake-hardware.yml` already support this
without new code — `docker compose -f docker-compose.fake-hardware.yml --profile replay-verify
up replay` (omit `-d` for just that one service, after `mock-robot`/`gt-replay-verify` are
already up detached) streams its output directly to whatever terminal invokes it. Document
this as the standard manual debug recipe rather than building a new wrapper around it.

`CONTROL_FREQ` is already a compose env var (`${CONTROL_FREQ:-30.0}`, read by both
`docker-compose.fake-hardware.yml` and `dataset_gt_replay.launch.py`) — already
user-configurable per the request, no change needed.

## Part 2 — Redesign the ee_delta per-tick trace into one clear, numbered block

Location: `inference_node.py`'s `_publish_loop` (`is_ee_delta` branch — the same block already
touched this session for the position-proximity hold-gate) and `_obs_update` (where
`_ee_delta_latest_obs_quat` is written).

Add a monotonic per-arm-set tick counter (`self._ee_delta_tick: int`, incremented once per
`_publish_loop` ee_delta invocation, independent of `_replay_cursor` so it still counts HELD
ticks, not just published ones — needed to see *how many ticks* a hold lasts, which is exactly
what's needed to characterize the residual flakiness). Replace today's single-line
`[ee_delta] ... PUBLISHED/HELD(...)` log with an explicit, labeled sequence for each tick,
still gated behind `self._debug` (no change to non-debug behavior):

```
[ee_delta t=142] action[142] popped: delta=[...]
[ee_delta t=142] obs[142] latest: [...]  (seq=..., row=45/593)
[ee_delta t=142] cmd_abs = obs ∘ delta -> publishing to /commanded_ee_left, /commanded_ee_right: [...]
[ee_delta t=142] obs[143] after publish: [...]  moved=yes pos_delta=0.0031m rot_delta=0.4deg
```
(last line logged at the *start* of the next tick, comparing this tick's freshly-read obs
against the previous tick's published target — this is the explicit "did it actually move"
confirmation, reusing `pose_arrival_error` already imported for the hold-gate.)

For a HELD tick, print the equivalent explicit block instead:
```
[ee_delta t=143] HELD(pos): action[143] NOT popped (queue unchanged) -- arm=0 pos_err=0.031m > tol 0.025m
```
so it's visually obvious from the tick number and "NOT popped" language that the queue didn't
advance, without needing to cross-reference two log lines to infer it.

Reuse: `pose_arrival_error` (`ee_runtime.py`, already imported this session),
`ee_poses_from_chunk` (already imported), the existing `_ee_delta_last_commanded_quat` state
(already added this session for the hold-gate) — no new dependencies.

## Part 3 — Use the new tracer to actually root-cause the residual flakiness, then fix it

With Part 2's live trace, reproduce the failure (rerun the correctness test — or the manual
foreground `replay` command — repeatedly until it reproduces; historically ~1-in-5-8, so a
handful of attempts should catch it) and read the tick-by-tick trace around the failing index
directly instead of inferring from `first_failures`' pre/post index values. Specifically look
for: a HELD streak right before the divergence point (confirms it's a timing/anchor issue the
existing gates *should* catch but don't quite, e.g. an off-by-one tick where compose happens
one cycle too early), vs. no HELD at all (would point to something in the compose math itself,
or the mock's echo/sequence bookkeeping, rather than the hold-gate timing).

Do not assume the fix in advance — this is exploratory. Once the tick-level cause is visible,
propose the smallest correct fix (likely a refinement to the existing hold-gate's timing/
condition, not new machinery) and verify by running the correctness test repeatedly
(10+ times) until confident the ~1-in-5-8 rate is gone, not just papered over.

## Verification

- `uv run pytest tests/unit/lerobot_control/` — full suite must still pass (no unit tests exist
  for `inference_node.py` itself, rclpy dependency, consistent with existing precedent — this
  change is docker/live-verified only).
- `docker compose -f docker-compose.fake-hardware.yml build mock-robot replay gt-replay-verify`
  then run `tests/smoke/scripts/gt_replay_correctness_test.py` **repeatedly** (aim for 10
  consecutive clean passes on `ee_delta`, not just one) — this is the actual bar for "made sure
  fake-hardware passes on delta," not a single green run.
- Manually exercise the foreground-streaming recipe once (Part 1) to confirm the new tick-level
  trace (Part 2) is genuinely readable live in a terminal, at both `control_frequency=10` and
  `=30`, before considering Part 2 done.
