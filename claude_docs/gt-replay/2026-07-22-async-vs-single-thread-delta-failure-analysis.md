# Async (dual-timer) vs. single-thread replay: why `ee_delta` fails and `ee_abs` doesn't

## 1. The two architectures, side by side

### Async / dual-timer (`inference_node.py`, subclassed by `dataset_gt_replayer_node.py`)

Two independent ROS timers, both at `control_frequency`, each in its own
`MutuallyExclusiveCallbackGroup`, run **concurrently on separate threads**
under a `MultiThreadedExecutor` (`inference_node.py:172-184`):

- **`_obs_timer` → `_obs_update`**: reads the live observation, converts it,
  writes it into shared state under `_obs_lock`
  (`_ee_delta_latest_obs_quat`, `_ee_delta_latest_obs_seq` —
  `inference_node.py:819-823`), then calls `_produce_action` (overridden by
  `DatasetGtReplayerNode` to pop the next dataset row and hand back its raw
  delta) and appends the result to `_classic_action_deque`
  (`inference_node.py:839-841`). The dataset cursor (`_replay_cursor`)
  advances **here**, gated only by simple deque backpressure
  (`len(deque) >= maxlen-1`) — it has **no idea** whether the previously
  published target has actually been reached.
- **`_publish_timer` → `_publish_loop`**: independently, snapshots the
  shared obs state under the same lock (`inference_node.py:1022-1024`),
  runs **two independent hold-gates**, and — only if both pass — pops the
  oldest queued delta, composes `absolute_target = obs ∘ delta`
  (`ee_delta_restore_step`), publishes, and updates gate state.

The two gates:
1. **Sequence gate** (fake-hardware only): holds unless every arm's
   observed-pose sequence number has strictly advanced past the sequence
   used for the last publish (`inference_node.py:1042-1076`).
2. **Position-proximity gate** (both hardware types): holds unless the live
   observed pose is within `anchor_atol_pos_m`/`anchor_atol_rot_deg` of the
   last **commanded** target (`inference_node.py:1087-1109`).

Both gates are **heuristics** approximating one question — "has the robot
caught up to what I last told it?" — evaluated by a *different thread*, at
a *different moment*, than the thread that will use the answer.

### Single-thread (`single_thread_gt_replayer_node.py`, this session)

One ROS timer, one callback, one thread. Every tick, in strict order: check
arrival (wait-until-arrived) → pop → compose against the *same* obs read for
the arrival check → publish. There is no second timer, no second thread, no
possibility of another callback interleaving mid-tick — the executor only
ever runs one callback body at a time for this node.

## 2. Why `ee_delta` needs a pairing invariant that `ee_abs` doesn't

`ee_delta`'s entire correctness rests on one invariant: **the delta popped
at cursor `i` must be composed against the observation the robot has
actually reached after cursor `i-1`'s command landed** — because a delta
means "move this much *from wherever you physically are right now*"
(`ee_delta_forward`/`ee_delta_inverse`, world-frame, `ee_transform.py:299-467`).
Compose it against the wrong obs (stale, or not yet caught up) and the
composed absolute target is wrong by however far the true and assumed
anchors differ — and that error **compounds**, because the next tick
composes the next delta against wherever this wrong target lands.

`ee_abs` has no such invariant. Its "action" is *already* the absolute
target — `_publish_ee_action` sends it straight through
(`inference_node.py:1297-1389`), no compose step, **no read of
`_ee_delta_latest_obs_quat` at all**. This is the key asymmetry: whatever
timing hazard exists in the dual-timer architecture's shared obs state is
**completely invisible to `ee_abs`**, because `ee_abs`'s publish path never
touches that shared state. It's not that `ee_abs` "happens to be more
robust" — it structurally cannot be exposed to this class of bug at all.
This directly answers the "why does absolute replay easily but delta
struggles" question from earlier in this session.

## 3. Why the dual-timer design is structurally exposed to a race `ee_delta` can't tolerate

Both hold-gates are read-then-decide checks performed by `_publish_loop`,
running on a thread independent of the thread that *wrote* the obs
state it's checking. The `_obs_lock` prevents a torn read of any single
snapshot, but it guarantees nothing about **when** relative to the true
physical/mock state that snapshot was taken relative to the *gate
decision* being made from it. Two independently-scheduled OS threads on a
`MultiThreadedExecutor` do not interleave in a fixed, predictable pattern —
jitter in scheduling means the exact obs value `_publish_loop` sees on any
given tick, and whether the gates judge it "arrived enough," is not fully
deterministic tick to tick.

Three consequences follow directly from this structure:

- **The gates are approximate, not a true rendezvous.** A single shared
  timing hazard can make a gate pass on an obs snapshot that is *subtly*
  wrong (not simply "old" — which the gates are specifically built to
  catch — but inconsistent in a way that still falls inside tolerance).
  When that happens, the compose step silently produces a slightly wrong
  absolute target, and the error compounds forward.
- **Both arms are composed from the exact same shared snapshot** —
  `_latest_obs` is one array covering every arm, snapshotted once per
  publish tick (`inference_node.py:1022-1024`), then sliced per arm for
  compose. If a bad snapshot slips past the gates on some tick, **every
  arm's compose that tick inherits it simultaneously** — this is a precise,
  structural explanation for the documented failure signature ("both arms
  diverge at the identical index," `claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md`
  §15), not a coincidence.
- **Backpressure decouples cursor advancement from arrival.** `_obs_update`
  keeps popping dataset rows into the deque based only on
  `len(deque) < maxlen-1`, with zero awareness of whether `_publish_loop`
  is currently holding. This is *not* itself unsafe (FIFO order is
  preserved, so a backed-up queue still drains in the correct sequence
  once unheld) — but it does mean the two loops' notions of "where we are
  in the episode" can diverge by up to `maxlen` rows at any moment,
  widening the window during which a stale/racy read could matter.

## 4. Why the single-thread design removes this class of bug structurally, not just empirically

Collapsing "read obs for the arrival check" and "read obs to compose the
next target" into the **same value, read once, in the same callback
invocation**, with no other thread able to run concurrently, does not make
the race *less likely* — it removes the precondition for the race to exist
at all. There is only one moment per tick, one thread, one obs value, used
consistently for both the gate decision and the compose. The two heuristic
gates collapse into one (`wait_until_arrived`), and the deque/cursor's
advancement is no longer a separate, independently-scheduled process that
can run ahead of what has been verified to be published.

The single-thread replayer's first real test run reproduced errors up to
0.32m/62° — but that was traced (via direct inspection of the dataset,
not the async architecture) to a **separate, unrelated bug in the new
dedupe-based verifier** (deduping by pose value instead of
`MockEEPose.sequence`, causing per-arm index desync whenever a bimanual row
left one arm's target unchanged — see this session's fix). That bug lives
entirely in the *verification tool*, not in either replay architecture, and
should not be read as evidence about the async design's race — it was a
measurement bug in the harness built to check the single-thread design, not
a control-loop bug. After that fix, single-thread `ee_delta` passed at
floating-point precision (`max_pos_err≈5e-8m`) on the first clean run.

## 5. Honest limits

This report's core claim — the dual-timer/shared-cross-thread-state
architecture is structurally *capable* of producing exactly the observed
failure signature, and `ee_abs` is structurally immune because it never
reads that shared state — is derived directly from reading the actual
`inference_node.py` control flow and matches the documented failure
signature precisely. What it is **not**: a captured, live trace pinpointing
the exact scheduling condition that triggers a bad snapshot on the ~1-in-5-8
occasions it has been observed. The original plan's Part 3 (build an
enhanced per-tick tracer and use it to catch the async version in the act)
was superseded by the decision to build a structurally-race-free
alternative instead of continuing to debug the racy one. The single-thread
design's repeated-run stress test (currently running) is the empirical side
of this argument — if it holds up over many repeated runs where the async
version was flaky, that is strong (though not 100% mechanistic) corroborating
evidence for this explanation.
