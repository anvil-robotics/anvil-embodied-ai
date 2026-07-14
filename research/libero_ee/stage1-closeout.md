# Stage 1 close-out — LIBERO EE-space action-representation study

_Written 2026-07-14 to summarize (not re-derive) the state of this investigation before
branching to a new simulation backend ("Stage 2", out of scope here). Read this first if
picking the work up cold — it links back to `report.md`/`diary.md`/the ledger for evidence,
rather than repeating it._

## Terminology

- **Delta = n-(n-1)** — each step's target is relative to the immediately-preceding REAL state
  (a per-frame anchor). This is `native`'s own representation.
- **Relative = n-0** — each step's target is relative to the state at CHUNK START (a
  chunk-start anchor), baked at convert time in this sim harness.
- **Absolute** — the target is the raw absolute pose itself, never relativized.
- **OAA (Observation-As-Action)** — a target derived from a future OBSERVED state
  (`state[t+h]`), not from a recorded controller command. This replaces the earlier "AFO"
  naming used in this branch's history (`afo_abs*`/`afo_relative` action types) — the two names
  refer to the same idea. OAA experiments themselves are Stage 2 territory; nothing new here.
- Each of Delta/Relative/Absolute can be paired with either a **commanded** (derived from the
  recorded controller command) or an **OAA** (observation-derived) target source.

## 1. Validated (with confidence + caveats)

Everything below is a **provisional, LIBERO/robosuite-specific finding** — none of it is
confirmed to generalize past this simulation harness, pending Stage 2 cross-validation (§4). See
`report.md`/`ledger/RESULTS.md` for the numbers and `⚠ Provisional`/`⛔ Invalid` flags now added
inline at each citation (Task 1-1 of this close-out pass).

- **Delta / `native`** (commanded, n-(n-1), world frame, relative delivery) — **the most
  reliable representation measured**: 71.43% fresh GT-replay baseline (n=49, corrected — see
  "Ruled out" for the stale 60% this superseded), 80–90 `pc_success` across task10/14/11 (ACT),
  88–100 (Diffusion). High confidence. This is what production already uses.
- **Absolute / `native_ctrlgoal_relconv`** (commanded, absolute target reconstructing the
  controller's own scaled internal goal, delivered via the `relative_converted` two-step
  physical conversion — subtract live state, ÷ `output_max`, clip, deliver relative) —
  **87.76% GT-replay**, the headline "absolute delivery can work" result of this investigation.
  **Caveat, load-bearing: see Doubt 1 below** — this number exceeds the Delta/`native` baseline
  by an unexplained margin that undercuts a simple "absolute is validated" reading.
- **Recipe for relative-position training** (`report.md` §8, unaffected by the doubts below —
  this is the one conclusion NOT flagged provisional): world frame + per-frame (Delta) anchor +
  rot6d-OK + recovered-delta relative delivery, validated on ACT + Diffusion, task10 + task11.
  High confidence; this is the actionable deliverable for production's `ee_rel` fix.

## 2. Ruled out / deprecated

- **Historical `native_n0`** — ⛔ **definitively invalid**, not merely provisional. It was
  structurally locked to `per_frame_anchor=True` at convert time, so it never tested a real
  chunk-start (Relative/n-0) anchor at all — see `diary.md:77,80` and `report.md` §3.4's own
  admission. Its `pc_success` numbers (86/98 on task10, 76/98 on task11) stand as data but must
  not be read as "n-0 anchor works fine" — they're a degenerate case that collapses to the Delta
  command itself.
- **The pre-unit-mismatch `goal` family interpretation** — `report.md` §3.5's `goal-abs`/
  `goal-world-n0`/`goal-hand-n0` numbers stand, but the underlying target construction
  (`state + native_delta`, UNSCALED) was later discovered (this investigation) to carry no
  consistent physical unit — a coincidental cancellation trick that happens to be recoverable to
  floating-point precision, not a genuine absolute pose in `observation.state`'s units. Do not
  cite these results as an "absolute pose" test; they are a different, formal representation
  that happens to train well in this harness.
- **The π-singularity explanation for `afo_abs_h1`'s 8.2% GT-replay failure** — superseded.
  `orientation_error()` (`robosuite/utils/control_utils.py:109`) is provably a function of the
  RELATIVE angle between goal and current orientation, not either orientation's absolute
  magnitude; a direct round-trip test showed the axis-angle math is exact to `1e-16` near θ=π.
  **The true root cause of the 8.2%/88° divergence is still not fully confirmed** — see §3.

## 3. Open doubts (the load-bearing unresolved questions)

These are the two doubts that block treating this investigation's headline "Absolute" result as
settled, plus the one mechanism that was never nailed down. Stage 2 should reference these
directly rather than re-deriving the reasoning.

**Doubt 1 — does `native_ctrlgoal_relconv`'s 87.76% mean "Absolute is validated," or is it an
artifact of this one reconstruction formula?** If `native_ctrlgoal` truly reconstructs the
controller's own internal goal (`state[t] + clip(native_delta[t]) × output_max`), it should at
most MATCH the Delta/`native` baseline (71.43%), not exceed it by ~16 points. The leading
hypothesis: this specific reconstruction happens to produce a trajectory that is easier for the
controller to track — a property of the formula, not evidence that "Absolute" as a
representation family is validated. Unresolved; no experiment in this investigation
distinguishes these two readings.

**Doubt 2 — is the `relative_converted` delivery mechanism a generalizable finding, or a
LIBERO/robosuite-specific workaround?** This delivery (subtract live current state, divide by
`output_max`, clip, deliver via genuine relative mode) leans entirely on `output_max` — a
robosuite OSC-controller hardware-abstraction constant (the normalized `[-1,1]` command range
distinct from physical state units). If a production controller (e.g. OpenArm's) has no such
normalized-command-vs-physical-state split, this entire conversion mechanism may have ZERO
transfer value — its validated role may be strictly "makes LIBERO simulation work," not "solves
EE-space absolute-target delivery in general." Unresolved; only a unit-consistent backend
(Stage 2) can tell the two apart.

**The `afo_abs_h1` 8.2% mechanism — confirmed delivery-level, not construction-level, but the
exact cause was never pinned down.** What's ruled out: target content (the observed-future-pose
construction is not inherently unlearnable — `native_ctrlgoal`, a genuine physical-unit absolute
pose, reaches 87.76% via `relative_converted` delivery); the π-singularity explanation (§2);
rotation alignment at `env.reset()` (checked, small). **What was never checked: per-step state
alignment DURING chunk execution** (as opposed to at episode reset) — the leading remaining
hypothesis for the raw-`absolute`-delivery failure mode, left open when this investigation moved
to the `relative_converted` delivery instead of continuing to debug raw `absolute` delivery.

## 4. What only Stage 2 can settle

Doubts 1 and 2 both require a simulation backend where `observation.state` and `action` are
genuinely unit-consistent by construction (not via a LIBERO/robosuite-specific `output_max`
conversion) — e.g. ManiSkill's `pd_ee_pose` controller with `normalize_action=False`, a real
Cartesian EE-pose action space. Running the same Delta/Relative/Absolute × commanded/OAA matrix
there, without needing any `relative_converted`-style conversion step, is the only way to tell
whether this investigation's "Absolute works" result is a real representation finding or an
artifact of one reconstruction formula plus one delivery workaround. The harness seam
(`anvil_sim/study.py`'s `Study`/`GtReplayConfig`/`ReplayAdapter`, documented in
`packages/anvil_sim/src/anvil_sim/ARCHITECTURE.md`) is the intended reuse surface for that work.

## 5. Pointers

- Numbers and per-factor analysis: `report.md` (now carries inline `⚠ Provisional`/`⛔ Invalid`
  markers pointing back here) and `diary.md` (chronological log, unedited).
- Machine-generated ledger: `ledger/RESULTS.md` / `results.json` — now carries a `status` column
  surfacing the same flags per-row; see its own footnote about the corrected baseline (§ below).
- Baseline correction: the `native` GT-replay baseline used throughout most of this
  investigation was stale (`replay/baseline-task10/replay_info.json`, n=10, `pc_success=60.0`,
  dated 2026-07-07) — a genuine bug (`bench_runner._replay_baseline` keyed its cache on
  `task_index` alone, so an n=10 baseline was silently reused to gate every later n=49/50
  condition). Fixed 2026-07-13/14: the cache key now includes `n_episodes`
  (`baseline-task{N}-n{n_episodes}`). The corrected fresh n=49 baseline is 71.43%
  (`replay/baseline-task10-fresh-n49/`). Only future `gt-replay` stage runs pick up the fix;
  historical per-experiment `stage_status.json` gt-replay numbers were NOT retroactively
  recomputed (out of scope for this pass — see `ARCHITECTURE.md`).
- GT-replay is confirmed genuinely closed-loop (live physics every step, never reset to a
  dataset-recorded value) — traced through `eval_replay.py`'s rollout loop; this is why every
  `pc_success` figure in this document reflects real controller behavior, not a replay artifact.
- Experiment artifacts for the conditions in Doubts 1/2:
  `packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_native_ctrlgoal_act.yaml`,
  `..._native_ctrlgoal_relconv_act.yaml`, `..._afo_relative_act.yaml`,
  `..._afo_abs_{h1,h5,h10}_act.yaml`, `..._afo_abs_rel_h1_act.yaml` — all GT-replay-only (no
  training/eval run), hence absent from `ledger/RESULTS.md`.
- Architecture / harness-vs-study seam: `packages/anvil_sim/src/anvil_sim/ARCHITECTURE.md`.
