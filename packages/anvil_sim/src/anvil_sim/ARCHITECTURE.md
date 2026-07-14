# `anvil_sim` — harness vs. study split

Written 2026-07-14 as part of the LIBERO EE-space Stage 1 close-out (see
`research/libero_ee/stage1-closeout.md`), ahead of a Stage 2 that will need
to reuse this harness against a different simulation backend (e.g.
ManiSkill). This is a **light** documentation + safe-extraction pass, not a
package restructure — see "What's still coupled" below for what's deferred.

## The seam

The harness (`bench_runner.py`, `bench_spec.py`, `eval_replay.py`) is meant
to be study-agnostic: it sequences pipeline stages, gates on GT-replay, and
records a ledger, but knows nothing about any specific sim/dataset/action
representation. Everything study-specific lives behind three frozen
dataclasses in `study.py`:

- **`Study`** (`study.py:89-123`) — dataset roots, math validators, spec
  legality, command builders, ledger `condition_status` classification.
- **`GtReplayConfig`** — which action type/control-mode is the GT-replay
  baseline, and how to detect a spec that IS the baseline.
- **`ReplayAdapter`** — the per-treatment processor pipelines GT-replay must
  reproduce byte-for-byte against the real closed-loop eval path, plus (as of
  this pass) the per-step divergence metric and its "notable divergence"
  thresholds.

A study registers itself once, by name, via `register_study()` (see
`study.py`'s `register_study("libero_ee", _build_libero_ee)`); the harness
resolves it through `get_study(name)` / `BenchSpec.study`. Adding a second
study means writing a new `build_*_study()` factory and registering it — the
harness files should need zero changes.

## What moved behind the seam this pass

Two pieces of LIBERO/robosuite-specific logic that were living as module-level
constants/inline code in the generic `eval_replay.py` are now supplied by the
study's `ReplayAdapter` instead (`studies/libero_ee/replay_adapter.py`):

- **`divergence_pos_threshold` / `divergence_rot_threshold`** — the "notable
  divergence" bar for the `state_divergence_first_exceed_t` diagnostic.
  libero_ee sets these to robosuite OSC_POSE's own `output_max`
  (`libero_convert.OSC_OUTPUT_MAX_POS/ROT`); a different backend supplies its
  own scale.
- **`state_divergence(demo_state, actual_state)`** — the per-step
  position/rotation error computation. Its exact index layout (which slice is
  position, which is rotation, and in what representation — axis-angle for
  the recorded demo state, quaternion for the live probe state) is an
  EE-encoding assumption specific to this study, moved verbatim (behavior
  unchanged) into `_state_divergence()`.

## What's still coupled — known Stage-2 work, deliberately deferred

`eval_replay.py`'s rollout loop still has real backend dependencies not
addressed by this pass (a full restructure was explicitly out of scope):

- `eval_replay.py:62-67` — hard imports of `anvil_shared.ee_transform`
  (`ee_rel_forward`/`ee_rel_world_forward`) and four `lerobot.envs.*` /
  `lerobot.datasets` / `lerobot.utils.constants` symbols. The replay driver
  has a compile-time dependency on the LIBERO/lerobot backend.
- `eval_replay.py:122` — `GtActionProvider.__call__` calls
  `ee_rel_forward`/`ee_rel_world_forward` directly; the (1,10)/(1,8)
  rot6d+quat array layout is EE-representation-specific.
- `eval_replay.py:176-179` — `LiberoEnvConfig` + `make_env` + the
  `{suite:{task_id:env}}` unwrap: env construction is LIBERO-specific inside
  the otherwise-generic `replay()`.
- `eval_replay.py:225-226,262-263` — `preprocess_observation`/`add_envs_task`
  and success extraction via `info["final_info"]["is_success"]` are
  lerobot/libero rollout conventions.
- `eval_replay.py:327-361` (`main()`) — CLI defaults `--task=libero_goal`,
  `--task-id=8`.

A future Stage 2 pass should push these behind `ReplayAdapter` too (or a new
`EnvAdapter`), the same way this pass moved the divergence metric — at that
point `eval_replay.py` would import zero backend-specific code. Until then,
adding a second study means it must still supply LIBERO-shaped state/action
arrays through the existing seam, OR this file's remaining couplings get
generalized first.

## Hard-won invariants (grep `# INVARIANT:` for the exact code)

Config combinations that look plausible but are silently wrong (never raised
until forced to, hence "hard-won" — each was caught after the fact, not by
design):

- **`libero_processor.py`** (`ZeroCalActionProcessorStep.__post_init__`) —
  `mode="abs"` + `per_frame_anchor=True` raises. `mode="abs"` never reads
  `anchor` in `action()`, so `per_frame_anchor=True` would silently be a
  no-op rather than doing what the caller likely intended. Corollary,
  confirmed empirically (not just by this guard): `per_frame_anchor` and
  `n_action_steps`/`L` are BOTH no-ops for every `mode="abs"` condition
  (`native_ctrlgoal*`, `afo_*`, `native_abs`) — an L-sweep or anchor-sweep
  over those conditions produces byte-identical results at every setting.
- **`libero_processor.py`** (same `__post_init__`) — `mode`/`deliver`/
  `gripper_mode`/`action_encoding` are each validated against a closed set of
  strings at construction time, because `action()`'s if/elif chains have no
  else-raise — an unrecognized value would otherwise silently fall into the
  wrong branch.
- **`studies/libero_ee/study.py`** (`_legality`, via `_REQUIRED_CONTROL_MODE`)
  — a treatment's `deliver` mode and `env.control_mode` must be paired
  correctly. This is the Experiment 7 lesson: mismatching them produces no
  crash, no test failure, and a normal-looking training loss — only a 0%
  closed-loop rollout, discovered after a full training sweep. Now a
  load-time `ValueError` instead.
- **`bench_runner.py`** (`_replay_baseline`) — the GT-replay baseline cache
  key includes `n_episodes`, not just `task_index`. Bug found 2026-07-13:
  the old key was `task_index`-only, so a baseline computed at one
  `n_episodes` (e.g. 10) was silently reused to gate every later condition
  on the same task run at a different `n_episodes` (e.g. 49/50) — see
  `stage1-closeout.md` for the numeric fallout.
