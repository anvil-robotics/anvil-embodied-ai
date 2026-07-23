# Single-timer `dataset_gt_replayer_node.py` + homing — subclass-only, zero changes to `inference_node.py`

## Context

This session proved (via a standalone single-thread GT-replay tool) that `ee_delta`
compose is exact (`max_pos_err≈5e-8m`) once obs-read/arrival-check/compose/publish
happen atomically in one thread, and confirmed via robosuite's own
`OperationalSpaceController` source (`osc.py`) that this is the *correct* reference
model — `set_goal()` composes every tick against the live state unconditionally,
trusting a continuous inner-loop controller (`run_controller()`, analogous to
anvil-workcell's real arm controller) to track it. Production's
`dataset_gt_replayer_node.py` still inherits `LeRobotInferenceNode`'s racy
dual-timer architecture (`_obs_timer`/`_publish_timer` on separate threads),
which the correctness tests show still diverges even after this session's
partial fixes.

**Explicit scope decision (Patrick's call):** fix this ONLY inside
`dataset_gt_replayer_node.py`, as a subclass-local override. Do **not** touch
`inference_node.py` at all — live model inference (production ACT/Diffusion/VLA)
stays completely untouched, zero risk, zero diff. This is lower cost than either
(a) porting homing into the separate experimental single-thread tool (would
duplicate homing/launch/human-eval integration that already exists and is
tuned), or (b) refactoring the shared base class (higher blast radius, affects
live inference too). The single-thread tool
(`single_thread_gt_replayer_node.py`) stays as-is, unused by this change — this
plan makes the *production* replayer itself single-timer instead.

## Why this is achievable with zero `inference_node.py` changes

Everything `dataset_gt_replayer_node.py` needs already exists as reusable,
unmodified base-class methods or session additions, callable directly from a
subclass override:

- `self.strategy.get_latest_ee_state_quat()` — added to `MultiProcessStrategy`
  this session (`strategies/multi_process.py`) specifically as a lightweight,
  image-free EE pose read. No `inference_node.py` involvement.
- `self._check_homing_arrival()` / `self._publish_home_target()` — already
  defined on `DatasetGtReplayerNode` itself, real-hardware-tuned. Their only
  precondition is `self._last_raw_ee_obs_np` being fresh, currently set by the
  base `_obs_update` — the new tick method sets it directly instead.
- `self._produce_action(None, None)` — `DatasetGtReplayerNode`'s own override
  ignores both arguments entirely (returns `self._gt_actions[cursor]` or `None`
  per existing backpressure/dry-run/end-of-episode logic) — callable as-is.
- `self._publish_action(action)` — base-class dispatcher (metrics, smooth
  tracker, `_has_published`) — reused unchanged, same call `_publish_loop`
  makes today.
- `ee_delta_restore_step` / `pose_arrival_error` / `ee_poses_from_chunk`
  (`.ee_runtime`) — free functions, already used elsewhere; import fresh into
  `dataset_gt_replayer_node.py`.
- `self._ee_delta_last_commanded_quat`, `self._wait_until_arrived`,
  `self._ee_delta_anchor_atol_pos_m`/`_rot_deg` — already declared/populated by
  the base class's `__init__`/`_setup_config` (this session's earlier work);
  simply stop being written by the base `_obs_update`/`_publish_loop` (since
  this subclass no longer calls them) and get written/read by the new method
  instead. No new fields needed in the base class.

## Implementation

In `dataset_gt_replayer_node.py` only:

1. **Add an `__init__` override** (class currently has none — relies on the
   inherited one): call `super().__init__()`, then destroy the two base timers
   (`self.destroy_timer(self._obs_timer)`, `self.destroy_timer(self._publish_timer)`)
   and create one new timer (`self._replay_timer`) at `1.0/self.control_freq`
   calling a new `_replay_step` method.

2. **New `_replay_step(self)` method** — the merged tick, in order:
   - `obs_quat = self.strategy.get_latest_ee_state_quat()`; return early if
     `None` (startup warm-up, no EE pose yet).
   - `self._last_raw_ee_obs_np = obs_quat` (satisfies the existing homing
     methods' precondition).
   - If homing not confirmed: call `self._check_homing_arrival()`; if still
     not confirmed, call `self._publish_home_target()`; return either way
     (mirrors today's split-timer behavior exactly, just sequential).
   - ee_delta wait-until-arrived gate (only if `self.is_ee_delta and
     self._wait_until_arrived and self._ee_delta_last_commanded_quat is not
     None`): per-arm `pose_arrival_error` vs tolerance; hold (return) if any
     arm exceeds it. (No hold-timeout tracking needed here — unlike the
     standalone single-thread tool, a stuck homing/arrival already has its own
     timeout via `_check_homing_arrival`'s `homing_timeout_sec`; mid-replay
     arrival is not expected to time out separately in v1 — flag this as an
     open question in Verification if Patrick wants a parallel safety net.)
   - `action = self._produce_action(None, None)`; return if `None` (dry-run,
     backpressure — vestigial now since the deque is never populated by this
     subclass, but harmless; end-of-episode).
   - If `self.is_ee_delta`: compose via `ee_delta_restore_step(action,
     obs_quat)`, recompute `self._ee_delta_last_commanded_quat` via
     `ee_poses_from_chunk` (same snippet already used in `inference_node.py`,
     copied not shared — acceptable minor duplication to avoid touching the
     base file).
   - `self._publish_action(action)`.

3. **Leave unchanged**: `_setup_config`, `_validate_required_params`,
   `_load_run_metadata`, `_setup_model`, `_setup_homing`,
   `_check_homing_arrival`, `_publish_home_target`, `_write_signal`,
   `_produce_action`, `main()`. `_classic_action_deque` simply goes unused by
   this subclass (nothing appends to it anymore) — leave it declared in the
   base class untouched; `_produce_action`'s existing backpressure check
   against it becomes a permanent no-op, harmless.

4. **Update the class docstring** — the current text says "obs reading, the
   shared EE-conversion head in `_obs_update`, the classic action deque,
   `_publish_loop`... is inherited unchanged," which will no longer be true;
   rewrite to describe the new single-timer `_replay_step`.

## What does NOT need to change

- `inference_node.py` — zero diff. Verify with `git diff` before considering
  this done.
- `dataset_gt_replay.launch.py`, `docker-compose.yml`'s `gt-replay-real`,
  `docker-compose.fake-hardware.yml`'s `replay`/`gt-replay-verify` — all
  invoke the same entry point (`dataset_gt_replayer_node`) with the same
  params; nothing about the external interface changes.
- `scripts/gt_replay_human_eval.py` — same completion-signal contract,
  same homing-status field, no changes needed.
- The standalone `single_thread_gt_replayer_node.py` tool — left as-is,
  unused going forward but not deleted (still useful for isolated debugging).

## Verification

- `uv run pytest tests/unit/lerobot_control/` — full suite green.
- `git diff -- ros2/src/lerobot_control/lerobot_control/inference_node.py`
  is empty — hard confirmation of the "don't touch production inference"
  constraint.
- `tests/smoke/scripts/gt_replay_correctness_test.py --wait-until-arrived both
  --repeat 10` on fake hardware — both `ee-abs` and `ee-delta`, both gate
  settings, all clean. This is the real bar (production path was still
  failing before this change).
- Fake-hardware homing sanity: one run with `HOME_BEFORE_REPLAY=true` (a
  setting not exercised by the correctness test, which defaults it off),
  confirm homing still ramps/confirms/times-out correctly through the new
  single-timer path.
- **Real hardware**: this drives a physical robot arm — Patrick runs
  `scripts/gt_replay_human_eval.py --target real` himself, watching the arm,
  not run autonomously. I'll prep the exact command but won't execute it
  against real hardware without him present/watching.
