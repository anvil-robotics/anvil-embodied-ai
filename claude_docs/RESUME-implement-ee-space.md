# Resume guide — `patrick/implement-ee-space`

Paused: 2026-07-23. 46 commits ahead of `main`, 167 files changed
(`git diff --stat main...HEAD`).

## Purpose

Adds end-effector (EE) Cartesian-space support across the whole pipeline —
`mcap_converter` (EE schema/encoding, abs + delta actions), `anvil_trainer`
(EE delta transform/stats), inference (EE runtime, delivery) — plus a
GT-replay tool: replays a recorded dataset's actions through the real ROS2
control loop against fake/real hardware and verifies published commands
match ground truth, used as both a correctness gate and a human-eval harness
before trusting a checkpoint on the physical arm.

## Status as of pause (2026-07-23)

**Committed, considered working** (see `git log -20 --oneline`): EE schema
versioning/encoding rename in `mcap_converter`; EE runtime
(`ee_runtime.py`, world-frame delta compose/inverse); the dual-timer
`dataset_gt_replayer_node`/`gt_replay_verifier_node` fake-hardware pipeline;
a position-proximity hold-gate for `ee_delta` (`950638e`); sequence-scoping
fixes for fake-hardware-only `MockEEPose` (`73b6559`, `2f357ed`); GT-replay
human-eval CLI with auto-verify for `--target fake` (`277ba1e`).

**WIP checkpoint, commit `62c7d36`** ("chore(pause): checkpoint WIP before
repo pause", 25 files, +2420/-74) — not fully validated:
- New single-thread GT-replay verifier/replayer pair
  (`single_thread_gt_replayer_node.py`, `single_thread_gt_replay_verifier_node.py`),
  wired into `docker-compose.fake-hardware.yml` as services
  `replay-single-thread` / `gt-replay-verify-single-thread`.
- Rework of the **production** `dataset_gt_replayer_node.py` toward a
  single-timer design (per the 2026-07-23 plan below).
- Homing/gate tuning in `strategies/base.py`, `strategies/multi_process.py`,
  `fake_hardware_node.py`, `dataset_gt_replay.launch.py`.
- New fake-hardware debug tool `ee_pose_latency_check.py`.
- New unit test `tests/unit/anvil_trainer/test_resume_dataset_root_inherit.py`.
- Two new debug mcap-converter v1.1 EE-bimanual configs (abs + delta).
- `docker-compose.yml`/`docker-compose.fake-hardware.yml` service additions.

Six design notes in `claude_docs/ee-delta/` and `claude_docs/gt-replay/` are
the primary source of truth — read them before making changes; this file
only summarizes and cites them.

## The open problem

`ee_delta` GT-replay on fake hardware fails intermittently (~1-in-5–8 runs):
both arms diverge at the **same** dataset index, by 5–60mm (verifier
tolerance is 1e-4m); `ee_abs` never shows this. Investigation, in order:

1. **Diagnosis** (`claude_docs/gt-replay/2026-07-22-async-vs-single-thread-delta-failure-analysis.md`):
   the production replayer inherits `LeRobotInferenceNode`'s dual-timer
   design — `_obs_timer`/`_publish_loop` on independent threads share obs
   state under a lock that guarantees atomicity but not *timing* relative to
   two hold-gates (sequence-based, position-proximity). `ee_delta` composes
   `absolute = obs ∘ delta` fresh every tick, so a subtly-wrong-but-in-tolerance
   snapshot silently produces a wrong target for **both arms at once** (one
   shared snapshot) — matching the observed signature exactly. `ee_abs`
   never reads that shared state, so it's structurally immune.
2. **Proof of concept**: a standalone single-thread tool
   (`single_thread_gt_replayer_node.py`) collapses obs-read/arrival-check/
   compose/publish into one timer/thread. Its first run showed large errors,
   traced to an unrelated verifier bug (deduping by pose value instead of
   sequence); after that fix, single-thread `ee_delta` passed at
   `max_pos_err≈5e-8m` (same doc, §4-5).
3. **Two fix plans, unclear which finished**: (a)
   `claude_docs/gt-replay/2026-07-22-async-architecture-fix-proposal.md` —
   port the single-thread fix into shared `inference_node.py`, affecting
   live inference too; (b)
   `claude_docs/gt-replay/2026-07-23-single-timer-dataset-gt-replayer-plan.md` —
   make only `dataset_gt_replayer_node.py` single-timer via subclass
   override, zero `inference_node.py` diff. The WIP commit's large
   `dataset_gt_replayer_node.py` diff looks like plan (b), **but this is not
   confirmed** — check the diff against plan (b)'s "Implementation" steps.
4. An earlier plan (`claude_docs/gt-replay/2026-07-21-ee-delta-tracer-plan.md`,
   live per-tick tracer) was superseded by building the single-thread
   alternative instead, per explicit note in doc #1 above, §5.

Separate, unrelated finding: `claude_docs/ee-delta/2026-07-21-anvil-trainer-vs-lerobot-train-diff.md`
shows the production `ee_delta` *training* pipeline (`anvil_trainer`'s
`EEDeltaTransform` + 9 monkeypatches over stock `lerobot-train`) has never
been validated at any scale; top suspect for near-zero motion at inference
is under-training (~19 actual epochs vs LIBERO's validated ~105), not
necessarily a pipeline bug. Independent of the GT-replay bug above.

## Known issues / open threads

- Async flakiness root cause is a strong structural argument, not a captured
  live trace of the exact scheduling condition (analysis doc §5).
- Whether WIP commit `62c7d36` fully implements the 2026-07-23 single-timer
  plan or is partial is unconfirmed.
- 2026-07-23 plan flags explicitly: mid-replay `wait_until_arrived` hold has
  no timeout safety net of its own (only homing does) — undecided.
- `wait_until_arrived=true` vs `false` A/B on the dual-timer production path
  (spec'd in the 2026-07-22 proposal's Verification section) not confirmed run.
- Real-hardware validation is explicitly deferred to Patrick running it
  himself, watching the arm — not to be run autonomously (2026-07-23 plan,
  Verification).
- Standalone single-thread tools are intentionally left unused-but-not-deleted;
  decide later whether to keep both or retire one.

## Concrete next steps to resume

1. Diff `dataset_gt_replayer_node.py` against the 2026-07-23 plan's
   "Implementation" steps to determine how much is actually finished.
2. Confirm `git diff -- ros2/src/lerobot_control/lerobot_control/inference_node.py`
   is empty relative to before this WIP — the plan's explicit constraint.
3. Run the fake-hardware correctness test repeatedly (10+ times, per both
   plans) on `ee_delta` with `wait_until_arrived=true`, via both the
   production `replay`/`gt-replay-verify` services and
   `replay-single-thread`/`gt-replay-verify-single-thread`.
4. If clean, run the `wait_until_arrived` true/false A/B for positive
   evidence the arrival check is load-bearing.
5. Decide: keep both replayer variants, or retire the standalone one.
6. Only after fake-hardware is solid: real-hardware run, executed by Patrick
   directly — do not run autonomously.
7. Separately (training-side): check the wandb loss curve for the
   early-stopped `ee_delta` checkpoint and consider a small-scale
   `anvil_trainer` validation run, per the ranked next-steps in the
   2026-07-21 trainer-diff doc.

## How to run/test

```bash
uv sync --all-packages

uv run pytest tests/unit/anvil_trainer/test_resume_dataset_root_inherit.py
uv run pytest tests/unit/lerobot_control/          # full suite, must stay green
uv run python tests/smoke/scripts/single_thread_gt_replay_test.py
uv run python tests/smoke/scripts/gt_replay_correctness_test.py   # ee-abs + ee-delta
```

GT-replay fake-hardware services (`docker-compose.fake-hardware.yml`):

```bash
# Production dual-timer replayer (profile: replay-verify)
docker compose -f docker-compose.fake-hardware.yml --profile replay-verify \
  build mock-robot replay gt-replay-verify
DATASET_PATH=/path/to/episode DEBUG=true \
  docker compose -f docker-compose.fake-hardware.yml --profile replay-verify up

# Experimental single-thread replayer (profile: replay-verify-single-thread)
DATASET_PATH=/path/to/episode \
  docker compose -f docker-compose.fake-hardware.yml \
  --profile replay-verify-single-thread up

# Live per-tick debug: bring mock-robot + gt-replay-verify up -d first, then
# run `replay` without -d to stream its stdout live (see
# claude_docs/gt-replay/2026-07-21-ee-delta-tracer-plan.md Part 1).
```

Real-hardware GT-replay: `gt-replay-real` in `docker-compose.yml` — run only
by Patrick, watching the physical arm.
