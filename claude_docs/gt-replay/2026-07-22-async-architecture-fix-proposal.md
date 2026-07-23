# GT-replayer async (dual-timer) optimization plan — updated

## Context

`dataset_gt_replayer_node` subclasses `LeRobotInferenceNode` and inherits its
split-timer architecture unchanged: `_obs_timer` and `_publish_timer`, both
created via `self.create_timer(1.0/control_freq, ...)` (`inference_node.py:175-184`)
— **same period by default (both default to 30Hz off the same `control_freq`
value), each independently scheduled** on its own `MutuallyExclusiveCallbackGroup`
/ OS thread via `MultiThreadedExecutor(num_threads=4)`. They are NOT phase-locked
to each other — two same-frequency, independently-ticking timers, whose exact
relative firing offset depends on executor scheduling, not a designed
synchronization. That independent-phase property is exactly why a race is
possible: `_obs_update` writes `_ee_delta_latest_obs_quat` at whatever instant
its own timer fires; `_publish_loop` reads it at a different, independently-scheduled
instant on a different thread. The single-thread GT-replay tool built this
session proved `ee_delta` is exact (`max_pos_err≈5e-8m`) once the arrival
check and the compose use the *same* read, in the *same* callback, with no
other thread able to interleave — this plan ports that fix back into the
dual-timer architecture without giving up the real benefit of two timers
(publish never blocked by a slow model forward pass).

**Scope decision (already made):** fix the *classic* ACT/Diffusion/GT-replay
path first (`_produce_action`/`_classic_action_deque`/`_publish_loop`'s
`is_ee_delta` branch). The VLA/RTC path (`self._is_vla`, `inference_node.py:620-709`)
already has a genuinely different, working pipelined-async implementation
(a third background daemon thread `_inference_loop`, `ActionQueue`,
RTC-style chunk merge/inpainting via guidance gradients) — that path is
**out of scope here**, not touched, and not something this bug needs.

## Two call patterns that funnel through the same fix

- **GT-replay** (`dataset_gt_replayer_node._produce_action`): returns one
  recorded dataset row per call, a plain array lookup — no model, negligible
  latency between "read obs" and "use the action."
- **Live classic inference** (`LeRobotInferenceNode._produce_action` base
  path, ACT/Diffusion, non-VLA): calls `self.model.select_action(observation)`
  (`inference_node.py:908`) — a blocking call that can take real wall-clock
  time, especially at chunk boundaries.

Both call `_produce_action` the same way and both take the **same** second,
fresh obs read right before composing — this is deliberately NOT
special-cased away for GT-replay just because its `_produce_action` happens
to be fast today. Two reasons: (1) GT-replay's whole purpose is to be a
faithful stand-in for the live-inference code path — if replay skips a step
live inference always does, it's no longer testing the same sequence of
operations, undermining exactly the property that makes GT-replay useful;
(2) it removes any implicit "this call is always fast" assumption from the
correctness argument — cheap to always do the second read, and it stops the
whole design from silently depending on `_produce_action`'s speed staying
negligible forever (e.g. if replay logic ever grows interpolation, logging,
or I/O). Both paths funnel through the exact same restructured
`_obs_update`/`_publish_loop` split below, unconditionally.

## Recommended restructuring

In `inference_node.py`:

1. **`_obs_update`** becomes the sole place ee_delta correctness is decided:
   - Read obs once (call this `obs_1`).
   - New `wait_until_arrived` param (bool, default `True` — same name/spirit
     as the single-thread tool's flag, kept togglable rather than hardcoded
     so both settings can be A/B tested here too, see Verification). If
     `ee_delta`, `wait_until_arrived=True`, and a previous absolute target is
     already pending/enqueued: check arrival (`pose_arrival_error` vs
     tolerance, same logic as today's position-proximity gate, just
     relocated) using `obs_1`. Not arrived → hold: return without calling
     `_produce_action`, without touching the queue. Retry next `_obs_timer`
     tick (same cadence as today; if the tick overran because the previous
     invocation was itself blocked, rclpy fires the next one back-to-back
     rather than skipping — same behavior as today, unchanged). If
     `wait_until_arrived=False`, skip this check entirely and always proceed
     — the point of testing this setting here is precisely that, unlike the
     single-thread tool, this path still has two independently-phased
     timers, so `False` is not guaranteed to be a no-op the way it was there.
   - Arrived (or no pending target yet, e.g. first tick): call
     `_produce_action(...)` → raw action (delta for `ee_delta`, already-absolute
     for `ee_abs`/joint).
   - Always (both GT-replay and live inference): if `ee_delta`, take a
     **second** fresh obs read (`obs_2`) right here, after `_produce_action`
     returns, before composing — never reuse `obs_1`. For live inference this
     minimizes the staleness window to just "queue + network," not
     "inference + queue + network." For GT-replay the gap is negligible
     today, but the read still happens, uniformly, for the reasons in
     "Two call patterns" above.
   - If `ee_delta`: compose immediately (`ee_delta_restore_step(delta, obs_2)`).
     If `ee_abs`/joint: pass the action through unchanged.
   - Push the resulting **already-absolute** action onto a renamed,
     mode-agnostic ready-to-publish structure (was `_classic_action_deque`;
     same deque, just now always holds resolved absolute targets, never raw
     un-composed deltas).
2. **`_publish_loop`** becomes one code path for every action type: pop the
   queue if non-empty and publish; if empty, republish the last published
   absolute target (preserves the steady-`control_hz`-regardless-of-stalls
   property that's the actual point of having two timers). Delete the
   `is_ee_delta` branch, both hold-gates, and every cross-thread read of
   `_ee_delta_latest_obs_quat`/`_ee_delta_latest_obs_seq` from the publish
   side entirely.
3. `_ee_delta_latest_obs_seq`/`SequenceStalenessGuard` becomes unused by the
   (now gate-free) publish path — may still be worth keeping as a
   debug/tracer assertion, not for gating anything.
4. Queue depth (`_classic_action_deque`'s `maxlen`) stays a tunable
   throughput knob (current default 10), not forced down to 1–2 — once every
   queued item is a pre-resolved, already-verified absolute target, depth is
   purely a latency/smoothing knob, not a correctness lever. (Rejected
   Gemini-suggestion #4 from the earlier critique, for the live-inference
   case; a tight/lockstep queue remains the right choice specifically for the
   single-thread GT-replay debug tool, not for this shared production path.)

## What this does NOT change

- The two timers stay independently scheduled at the same `control_freq`
  (no phase-locking added, none needed — the fix removes the *shared mutable
  state* they'd otherwise race on, not the independence itself).
- VLA/RTC path (`_inference_loop`, `ActionQueue`, merge/inpainting) — untouched.
- Per-arm state stays joint/combined (single shared obs snapshot for all
  arms) — per the earlier critique, splitting per-arm was assessed as
  treating a symptom, not helpful once the race itself is removed, and would
  risk bimanual-coordination side effects for no correctness benefit.

## Verification

- **A/B both `wait_until_arrived` settings on THIS (dual-timer) path** —
  the same experiment already run on the single-thread tool (5 repeats ×
  `ee-abs`/`ee-delta` × `true`/`false`, 20/20 passed, no observable
  difference there). Rerun it here, on `dataset_gt_replayer_node`, after the
  restructuring above: `tests/smoke/scripts/gt_replay_correctness_test.py`
  repeated (10+ runs) with `wait_until_arrived=true`, then again with
  `wait_until_arrived=false`. This is a genuinely different, more
  informative test than the single-thread A/B: since two independently-phased
  timers are still involved here (unlike the single-thread tool, which has
  none), `false` is NOT guaranteed to be harmless — if `false` starts
  flaking here while `true` stays clean, that is direct, positive evidence
  the arrival check is still load-bearing specifically because of the
  remaining cross-thread timer-phase independence, not just a leftover
  real-hardware-only precaution. If `false` ALSO passes cleanly here, that
  would suggest today's fake-hardware mock is simply too fast/exact to ever
  exercise this gate regardless of architecture — informative either way,
  don't skip this comparison.
- Re-run `tests/smoke/scripts/gt_replay_correctness_test.py` (production
  `dataset_gt_replayer_node`/`gt_replay_verifier_node` path) repeatedly
  (10+ times) on `ee_delta` post-change with `wait_until_arrived=true`
  (the recommended default) — this is the actual regression bar, since the
  change touches the shared `inference_node.py` base class both the
  production GT-replayer and live inference subclass depend on.
- Confirm `ee_abs` and joint-space paths are unaffected (`uv run pytest`,
  plus one live-inference smoke run if a model checkpoint is available).
- Manually confirm publish keeps firing at steady `control_hz` during an
  artificially slow forward pass (e.g. inject a `time.sleep` in a test
  double for `select_action`) — the property the whole timer split exists to
  protect, must not regress.
- Confirm VLA/RTC path behavior and tests are completely unaffected (this
  change should touch zero lines in the `self._is_vla` branches).
