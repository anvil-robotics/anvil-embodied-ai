# Plan — Real-hardware GT-replay evaluation via human judgment, and scoping GtReplayVerifierNode

**Status: implemented.** See `claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md`
§10/§14/§15 for the as-built description (this doc remains the design record — read it for
*why*, the architecture doc for *what exists now*). All of Part 1 and Part 2 below landed as
designed, with one addition found during implementation: `mock-robot`'s
`-p ee_seed_pose:=${EE_SEED_POSE:-}` crashed rclpy's arg parser on an empty value (never hit
before since every prior caller always computed a real seed) — fixed via Compose's
`${EE_SEED_POSE:+...}` conditional interpolation.

## Context

`GtReplayVerifierNode` (built earlier this session, see
`claude_docs/gt-replay/2026-07-18-correctness-test-plan.md`) compares a live replay run against a dataset's own
recorded trajectory to tight numeric tolerance (`atol_pos_m=1e-4`, `atol_rot_deg=0.5`). Those
tolerances only make sense against `fake_hardware_node.py`'s mock, whose echo is a perfect,
instantaneous, dynamics-free reflection of whatever was published — a real robot's
physically-measured pose will legitimately deviate from the recorded trajectory (actuation
dynamics, control-loop tracking error, physical latency), and that deviation is not a bug.
Nothing today makes this scoping explicit in the code or the architecture doc, and real
hardware needs a **separate** evaluation tool: replay one or more episodes, let a human
operator watch the robot and judge task success per episode (not a numeric check), and
produce a structured report distinguishing "replay never finished" from "replay finished but
the operator judged it a failure."

Research grounding this plan:
- Two existing episode-range parsers already implement exactly the needed Python-slice-like
  grammar: `parse_episode_index_spec` (1-based, `packages/mcap_converter/src/mcap_converter/cli/convert.py:131-166`)
  and `parse_episodes_spec` (0-based, `packages/mcap_converter/src/mcap_converter/cli/dataset_viz.py:60-102`)
  — the 0-based one is the shape to port. **Decided with the user: match its exact grammar —
  comma list + `start:end` ranges, non-negative integers only, no step, no negative
  indices** (simpler, consistent with the only two parsers that already exist in the repo;
  full Python slice grammar with `-1`/step was considered and explicitly declined).
- The repo's only interactive-prompt precedent is `migrate_config.py`'s `_confirm()`
  (`packages/mcap_converter/src/mcap_converter/cli/migrate_config.py:67-69`): raw `input()`,
  no framework, tested by monkeypatching `builtins.input`. The whole repo uses `argparse`,
  never click/typer, for CLI tools.
- No real-hardware compose entry exists for the GT replayer today — `docker-compose.yml` is
  the production/real-hardware compose file (`network_mode: host`, `privileged: true`, real
  `ROS_DOMAIN_ID`/`CYCLONEDDS_URI`), used via `scripts/run_inference.sh`, but it only launches
  `inference.launch.py`. `dataset_gt_replay.launch.py` is already fully hardware-agnostic (all
  topics come from the config YAML — no mock-specific hardcoding), so no launch-file changes
  are needed for real hardware, only a new compose service.
- No existing tool captures a human pass/fail judgment anywhere in the repo — this part is
  genuinely new. `inference_monitor_node` (Rerun-based live viewer, `--monitor-enable`)
  already exists for the "watch" half and needs no changes.

## Part 1 — Make `GtReplayVerifierNode`'s scope explicit (small doc fix)

Confirmed: nothing in the current design assumes real-hardware use — it's only ever wired
into `docker-compose.fake-hardware.yml`'s `replay-verify` profile, alongside the mock. This
is already true by construction; it just isn't stated anywhere.

- Add an explicit "Scope" note to `gt_replay_verifier_node.py`'s module docstring: this node
  is fake-hardware-only; real-hardware evaluation uses the new human-judgment tool (Part 2),
  not a numeric tolerance check, because physical dynamics/tracking error/latency are
  expected, legitimate deviation on real hardware, not bugs.
- Update `claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md` §10
  (`GtReplayVerifierNode`) with the same explicit statement, and add a forward-reference to
  the new tool once it exists.

## Part 2 — New tool: `scripts/gt_replay_human_eval.py`

### Architecture: thin host-side wrapper + one small node addition (not a node redesign)

`dataset_gt_replayer_node.py` already does nearly everything needed for one episode:
replay-to-completion, then `hold_last` freezes the last published pose. The only missing
piece is a way for something *outside* the ROS process to know "this episode finished" (or
"it didn't"). Rather than building multi-episode/pause-and-resume logic into the node itself
(a new IPC surface: a ROS service or topic to signal "resume"), the design mirrors
`gt_replay_correctness_test.py`'s existing pattern — a plain host Python script polling a
JSON sentinel file, with docker compose up/down per stage — and adds only:

- **Node-side** (`dataset_gt_replayer_node.py`): a new optional `completion_signal_path`
  param. When set, the node writes a small JSON file (`{"episode": idx, "rows_replayed": N,
  "status": "complete"}`) at the exact point it already logs episode completion (inside
  `_produce_action`'s `if self._replay_cursor >= len(self._gt_actions):` branch, right after
  `self._episode_done_logged = True`). The existing `main()` `SIGTERM` handler also writes
  `{"status": "interrupted"}` if the node is killed before that point — this is what lets the
  wrapper tell "crashed/killed mid-replay" apart from "finished normally."
- **Launch file** (`dataset_gt_replay.launch.py`): passthrough `completion_signal_path` arg.
- **`docker-compose.yml`**: one new service mirroring the existing `replay` service's command
  in `docker-compose.fake-hardware.yml`, but with production networking (`network_mode: host`,
  `privileged: true`, real DDS env) and no `mock-robot` dependency — the real robot's own
  controller stack is assumed already running independently; this tool never starts or stops
  it. New env vars: `EPISODE`, `COMPLETION_SIGNAL_PATH` (mirrors `REPORTS_DIR`'s existing
  mount convention).
- **`docker-compose.fake-hardware.yml`**: passthrough `COMPLETION_SIGNAL_PATH` on the existing
  `replay` service (no new service — dry-run mode reuses it as-is).

The wrapper (`scripts/gt_replay_human_eval.py`, argparse, matches the repo's CLI convention —
`--dataset`, `--episodes`, `--target {real,fake}`, `--config-file`, `--report-path`,
`--completion-timeout-sec`), for each resolved episode index in order:

1. Bring up (`docker compose ... up -d <service>`) that one episode: `EPISODE=<idx>`,
   `HOLD_LAST=true`, `LOOP=false`, `COMPLETION_SIGNAL_PATH=<mounted path>`.
2. Poll the sentinel file up to `--completion-timeout-sec`; classify `replay_status` as
   `"completed"`, `"timed_out"` (file never appeared), or `"crashed"` (container exited
   non-zero first).
3. If `"completed"`: prompt the operator (see UX below) and record `operator_verdict` +
   `comment`. If not completed: record `operator_verdict: null`, skip the prompt, log a
   warning, move on — **never conflate a replay failure with an operator "fail"**.
4. Tear the episode's container down (`docker compose down`) before the next episode, exactly
   like `gt_replay_correctness_test.py`'s per-scenario `finally: down` — keeps DDS/process
   state from leaking between episodes.
5. After all episodes: write the JSON report and print a compact pass/fail summary to stdout.

**Both real and fake hardware are supported** (`--target real|fake`), fake as an explicit
dry-run/rehearsal mode — same reasoning `gt_replay_correctness_test.py` itself is built on:
validate the harness (episode parsing, prompt loop, report shape, timeout/crash
classification) against the free, always-available mock before trusting it with scarce robot
time, and to onboard a new operator to the interactive flow without needing a robot.

### Completion-timeout ownership (decided)

`--completion-timeout-sec` is a **wrapper-only** concern. `dataset_gt_replayer_node.py` has no
notion of "replay timed out" — it just replays until its data runs out (writes the `complete`
sentinel) or gets killed (writes `interrupted`, or crashes and writes nothing). Only the host
wrapper polls-with-a-deadline, exactly like `gt_replay_correctness_test.py`'s existing
`VERIFY_TIMEOUT_SEC`/`REPORT_POLL_MARGIN_SEC` — no ROS param is added to the node for this.

Default value scales with the episode's own nominal duration rather than a flat guess (episode
lengths vary across datasets), with a per-target multiplier + fixed margin — real hardware
gets a larger allowance since tracking error/operator intervention can genuinely extend
wall-clock time beyond the nominal `n_frames / control_frequency`, where the mock has none of
that variance:

```python
# in scripts/gt_replay_human_eval.py
_TIMEOUT_PROFILE = {  # (multiplier, fixed_margin_sec) applied to nominal episode duration
    "fake": (1.5, 10.0),
    "real": (3.0, 30.0),
}
```
`nominal_duration = n_frames / control_frequency` (n_frames from
`dataset_reader.load_episode_actions(...)` — a cheap parquet read, done once per episode
during the upfront episode-list resolution, before any container is brought up).
`--completion-timeout-sec`, if explicitly passed, overrides this computation entirely for
every episode in the run. If homing (below) is enabled, its own `homing_timeout_sec` is added
on top of this budget, since the wrapper's outer poll must outlast both phases combined.

### Pre-replay homing & arrival confirmation

**Recommended: inside `DatasetGtReplayerNode` (Option A), not a separate node.** The node
already has everything homing needs, and the wrapper already tears down/rebuilds a fresh
container per episode — a separate node would only add another sentinel-file handshake
duplicating machinery that already exists, for no isolation benefit the per-episode container
lifecycle doesn't already provide:

- The frame-0 target comes from `dataset_reader.load_episode_observations_quat(...)[0]` —
  already available, no new reader needed.
- The live "did we arrive" signal reuses `self._last_raw_ee_obs_np` (already populated by
  `_obs_update`, quat layout) — its capture condition (currently
  `self._monitor_enable or self.is_ee_delta`, `inference_node.py`) just needs `or not
  self._homing_confirmed` added, so it also populates during homing for `ee_abs` without
  monitor enabled.
- The one-shot(-repeated) homing publish reuses `_publish_ee_action` directly — convert the
  frame-0 quat observation to rot6d via the same `ee_obs_abs_forward` already used elsewhere
  in `_obs_update`, then call `self._publish_ee_action(home_action_rot6d)`. No new message
  construction code.
- Mechanically: both split timers gain an early bypass while `not self._homing_confirmed`:
  `_publish_loop` publishes the frozen home target every tick instead of popping/composing
  from the deque; `_obs_update` checks live-pose-vs-target distance instead of calling
  `_produce_action` (so `self._replay_cursor` never advances until homing is confirmed). This
  reuses the existing control-loop cadence — no new timer.
- Arrival tolerance is a **separate, more lenient** pair of params from
  `GtReplayVerifierNode`'s tight tolerances (which check "does this match the recording to
  near machine precision" — meaningless here): `home_atol_pos_m` (default `0.01`, i.e. 1cm)
  and `home_atol_rot_deg` (default `5.0`) — aligned with `anvil_eval`'s own existing
  real-hardware EE pass/fail thresholds (`packages/anvil_eval/src/anvil_eval/metrics.py:16`,
  position <0.02m / orientation <5°), not invented from scratch. Propose extracting the
  quaternion-angle-distance formula (currently only inline in `gt_replay_verifier_node.py`)
  into a small shared `pose_arrival_error()` helper in `ee_runtime.py`, used by both — avoids
  a second copy of the same math.
- New node param `homing_timeout_sec` (default `30.0`) — this **is** a node-internal timeout
  (unlike replay completion, above): the node itself is the one actively comparing live pose
  to target every tick, so only it can decide "give up homing." On timeout, write
  `{"episode": idx, "status": "homing_failed"}` to the **same** `completion_signal_path`
  sentinel immediately (no new IPC channel — just a third possible `status` value alongside
  `complete`/`interrupted`) so the wrapper detects the failure right away instead of waiting
  out its full outer timeout, and skip replay entirely for that episode.
- New node param `home_before_replay: bool` (default `true`). The wrapper sets it `false`
  automatically for `--target fake` (the mock's existing `ee_seed_pose` param already provides
  an instant, exact seed — homing against it would just be redundant), so **no separate code
  path is needed**: it's the same boolean gate either way, defaulted per-target by the
  wrapper. (Homing *would* still work correctly against an unseeded mock too, confirmed in
  ~1 tick since the echo is instantaneous — worth keeping available behind an explicit flag
  for rehearsing the homing feature itself, but off by default for `fake`.)
- Scope: designed and described here for EE mode (where this session's work has focused).
  Joint mode would use the same phase-gating mechanism with a joint-angle distance metric
  instead of pos/quat — analogous, not detailed further here; flag as a follow-on if needed.

**Why not a separate node (Option B), and when it *would* become the right call:** the one
concrete reason to prefer a separate node is reusability against *live-model* inference
(`inference_node.py`), not just the replayer — "move to a known start pose before starting
any inference session" is a generically useful capability, not replay-specific. That need
doesn't exist today (this request is scoped to GT-replay evaluation only), so building it into
`DatasetGtReplayerNode` now is right-sized. If live-inference homing is ever wanted, the
correct move then is hoisting these same methods up to the shared `LeRobotInferenceNode` base
class — not spinning up a separate node — since the observation/publish infrastructure they
depend on already lives there.

### Episode selector (decided: match existing precedent exactly, no negatives/step)

Port `dataset_viz.py`'s `parse_episodes_spec` (0-based, end-exclusive ranges) into
`dataset_reader.py` as `parse_episode_spec(spec: str, total_episodes: int) -> list[int]` —
**ported, not imported**, same reasoning as `resolve_action_type`'s deliberate avoidance of
`mcap_converter.config.loader.ConfigLoader`: the inference Docker image doesn't ship
`mcap_converter`, so `lerobot_control` package code can't depend on it.

Grammar: comma-separated tokens, each a single non-negative int or a `start:end` range
(`start` defaults 0, `end` defaults `total_episodes`, end exclusive). Rejects (clear
`ValueError`, naming the bad token): non-integer tokens, `start >= end`, out-of-bounds
indices, negative indices, tokens with more than one `:`. Duplicate indices across tokens are
silently deduplicated; result is always sorted (matches existing precedent).

```
"0,1,2"      -> [0, 1, 2]
"0:10"       -> [0..9]              (total_episodes >= 10)
":3"         -> [0, 1, 2]
"5:"         -> [5 .. total-1]
"0,1:3,5"    -> [0, 1, 2, 5]

"3:1"        -> ValueError (start must be < end)
"abc"        -> ValueError (not an integer)
"-1"         -> ValueError (negative indices not supported)
"1:2:3"      -> ValueError (more than one ':')
"20"         -> ValueError (out of bounds, if total_episodes <= 20)
```

Test cases (`tests/unit/lerobot_control/test_dataset_reader.py`, new file or appended if one
exists by then): all of the above, both directions.

### Interactive prompt UX

Foreground, blocking, host-side (never inside a container) — same shape as
`migrate_config.py`'s `_confirm()`:
```
Episode 3 replay complete (247 rows replayed over 8.2s).
Did the robot complete the task successfully? [y/n]: n
Comment (optional, Enter to skip): gripper missed the handle on the second attempt
```
One deliberate deviation from `_confirm()`'s precedent: `_confirm()` treats anything other
than "y"/"yes" as a definitive "no" (fine for an abort-a-destructive-op prompt). A pass/fail
judgment has no safe default, so this prompt **re-asks** on truly unrecognized input (not
"y"/"n") rather than silently defaulting either way; the comment prompt is always shown after
and accepts empty input. Testable the same way `test_migrate_config.py` tests `_confirm()`:
monkeypatch `builtins.input`, no docker needed to test the prompt/report logic in isolation.

### Report schema

JSON (primary artifact, matches `gt_replay_verifier_node.py`'s convention) plus a compact
stdout summary (not a second file):
```json
{
  "dataset": "/path/to/dataset", "target": "real", "episodes_requested": "0,1,2:5",
  "episodes_run": [0, 1, 2, 3, 4], "started_at": "...", "finished_at": "...",
  "summary": {
    "n_total": 5,
    "n_homing_confirmed": 4, "n_homing_failed": 1, "n_homing_skipped": 0,
    "n_completed_replay": 4, "n_failed_to_replay": 1,
    "n_operator_pass": 3, "n_operator_fail": 1,
    "pass_rate": 0.75
  },
  "episodes": [
    {"episode": 0, "homing_status": "confirmed", "replay_status": "completed", "operator_verdict": "pass", "comment": "", "timestamp": "..."},
    {"episode": 1, "homing_status": "confirmed", "replay_status": "completed", "operator_verdict": "fail", "comment": "dropped the cup", "timestamp": "..."},
    {"episode": 2, "homing_status": "confirmed", "replay_status": "timed_out", "operator_verdict": null, "comment": null, "timestamp": "..."},
    {"episode": 3, "homing_status": "failed", "replay_status": "not_attempted", "operator_verdict": null, "comment": null, "timestamp": "..."}
  ]
}
```
Three orthogonal fields, never conflated:
- `homing_status`: `"confirmed"` | `"failed"` | `"skipped"` (skipped = homing disabled, e.g.
  `--target fake`'s default).
- `replay_status`: `"completed"` | `"timed_out"` | `"crashed"` | **`"not_attempted"`** (new —
  set whenever `homing_status == "failed"`, since GT playback never starts in that case).
- `operator_verdict`: `"pass"` | `"fail"` | `null` (`null` whenever `replay_status !=
  "completed"`, regardless of *why* it didn't complete).

`pass_rate` is still computed over `n_completed_replay`, not `n_total`, so episodes that
never finished replaying — whether from a homing failure, a timeout, or a crash — don't
silently make the task look worse than it is.

## Critical files

- NEW `scripts/gt_replay_human_eval.py` — wrapper: argparse CLI, per-episode docker
  compose orchestration, completion-signal polling, operator prompt, report writing
- `ros2/src/lerobot_control/lerobot_control/dataset_gt_replayer_node.py` — add
  `completion_signal_path` param; write the sentinel in `_produce_action`'s completion
  branch and in `main()`'s SIGTERM handler; add the homing phase (`home_before_replay`,
  `home_atol_pos_m`, `home_atol_rot_deg`, `homing_timeout_sec` params; early-bypass branches
  in `_obs_update`/`_publish_loop`; `homing_failed` sentinel status)
- `ros2/src/lerobot_control/lerobot_control/inference_node.py` — extend
  `_last_raw_ee_obs_np`'s capture condition to also fire during homing
  (`or not self._homing_confirmed`)
- `ros2/src/lerobot_control/lerobot_control/ee_runtime.py` — add shared
  `pose_arrival_error()` helper (quat-angle distance), used by both the new homing check and
  (refactored) `gt_replay_verifier_node.py`, so the math isn't duplicated
- `ros2/src/lerobot_control/lerobot_control/dataset_reader.py` — add `parse_episode_spec()`
- `ros2/src/lerobot_control/launch/dataset_gt_replay.launch.py` — passthrough
  `completion_signal_path` and homing params as launch args
- `docker-compose.yml` — new real-hardware replay service
- `docker-compose.fake-hardware.yml` — passthrough `COMPLETION_SIGNAL_PATH` on `replay`
- `ros2/src/lerobot_control/lerobot_control/gt_replay_verifier_node.py` — scope docstring
- `claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md` — §10 scope note +
  new section documenting this tool once built
- NEW `tests/unit/lerobot_control/test_dataset_reader.py` (episode-spec parser) and
  `tests/unit/lerobot_control/test_gt_replay_human_eval.py` (prompt/report logic,
  stdin-monkeypatched, no docker)

## Verification

- `uv run pytest tests/unit/lerobot_control/` — new parser + wrapper-logic unit tests
- Dry run end-to-end against the mock:
  `scripts/gt_replay_human_eval.py --target fake --dataset data/debug/ee-delta/ee-space-testing --episodes 0:2 --config-file configs/lerobot_control/inference_ee.yaml`
  — manually answer both prompts, confirm the report JSON shape and `pass_rate` math, confirm
  a deliberately-killed episode (e.g. `docker kill` mid-replay) shows up as `"crashed"` with
  `operator_verdict: null`, not skipped or miscounted
- Homing-specific: rerun with `--rehearse-homing` (or equivalent) against the mock with a
  deliberately wrong `ee_seed_pose`/no seed, confirm `homing_status: "confirmed"` appears
  quickly (mock echo is instantaneous); force a homing failure (unreachable
  `home_atol_pos_m`, e.g. `0.0`) and confirm `homing_status: "failed"` +
  `replay_status: "not_attempted"` + `operator_verdict: null`, and that the sentinel fires
  before the full outer timeout elapses
- Confirm `gt_replay_correctness_test.py` and the full existing unit suite still pass
  unaffected — `completion_signal_path` and all homing params are additive/optional, default
  to today's behavior when unset
