# Resume brief — `patrick/sim-valid-dev`

## Purpose

This branch validates Anvil's end-effector (EE) space training + closed-loop inference pipeline
(`anvil_trainer`, `anvil_shared.ee_transform`, ROS2 `lerobot_control`) against LeRobot's LIBERO
benchmark, in a new `anvil_sim` package, to decide **which action representation the real robot
should use** before committing to it in production. It answers: is the LIBERO simulation result
trustworthy enough to change how the real pipeline encodes/delivers EE actions? See
`research/libero_ee/report.md` Part 1 ("Goal") and `docs/simulation.md` (harness rationale: "this
project burned four full training sweeps on broken eval paths before this harness existed").

## Status as of pause (2026-07-23)

- Working tree is clean; tip commit `660934d` ("native_ctrlgoal/relative_converted
  absolute-delivery experiments + Stage 1 close-out") is a deliberate close-out point, not a
  mid-experiment cutoff — nothing uncommitted, nothing hanging.
- **Stage 1 is closed.** It swept EE action representation on the native LIBERO task10/11/14
  suite (ACT + Diffusion) via a gated pipeline (`anvil-sim-bench`), reached one solid, unflagged
  recipe recommendation, and explicitly deferred everything that needs a different sim backend to
  a not-yet-started "Stage 2" (out of scope on this branch — `research/libero_ee/stage1-closeout.md:1-6`).
- The authoritative resume document is **`research/libero_ee/stage1-closeout.md`** — written
  specifically so a cold read doesn't need to re-derive anything from `report.md`/`diary.md`. Read
  it first, in full, before touching `report.md`.

## Key findings so far

(Full detail/numbers: `research/libero_ee/report.md` Part 1 + §3–§8; raw per-run rows:
`research/libero_ee/ledger/RESULTS.md`, git-ignored/regenerable via `anvil-sim-bench status`.)

- **Recipe for relative-position training (the one unflagged, high-confidence conclusion):**
  world frame + per-frame ("Delta", n-(n-1)) anchor + rot6d-OK + recovered-delta relative
  delivery — validated on ACT and Diffusion, task10 and task11. Actionable fix for production's
  `ee_rel` (`stage1-closeout.md` §1, §5; `report.md` §8).
- **Factor ranking** (native-family single-flip sweep, task10 ACT, n=50): frame (world vs hand,
  −24 pc) ≫ encoding (axis-angle vs rot6d, −12) > delivery (relative vs absolute, −8) > anchor
  (≈0 on ACT). World-frame beats hand-frame — opposite the UMI intuition (`report.md` conclusion 2).
- **Anchor is architecture-dependent:** per-frame (Delta) anchoring is Diffusion-robust (98 pc);
  chunk-start (n-0) re-encoding collapses Diffusion to 16 pc. Mechanism pinned down (G2,
  `analysis/mechanism.{json,png}`, `analysis/collapse.png`): collapsed commands shrink to ~1/3
  magnitude / ~1/2 spread of demonstrations — **Diffusion mode collapse**, not target-magnitude/OOD
  (`report.md` conclusions 5, 8).
- **Generalizes past task10 (G1):** per-frame relative holds on rotation-heavy task11/Diffusion
  (98 ≈ 96 for native) (`report.md` conclusion 8).
- **Harness earned its keep:** GT-replay gate caught 5 real eval-path bugs, invisible to training
  loss and unit tests (`report.md` conclusion 7; per-bug detail in `diary.md`).
- **Headline but provisional:** `native_ctrlgoal_relconv` (absolute target,
  `relative_converted` delivery) reaches 87.76% GT-replay, exceeding the `native` Delta baseline
  (71.43%) — NOT settled, see Open issues below.
- `docs/relative_ee_failure_analysis.md` is the **original** (2026-06-18, pre-study) root-cause
  hypothesis for `ee_rel`'s real-robot failure. Its own banner marks it **superseded**: the real
  mechanism was the Diffusion mode-collapse finding above, not its anchor-mismatch/OOD hypothesis
  (H1) — kept only for the UMI/DP code-comparison background in §1–§4.

## Known issues / open threads

All from `research/libero_ee/stage1-closeout.md` §2–§3 ("Ruled out" / "Open doubts"):

1. **Doubt 1 — is 87.76% `native_ctrlgoal_relconv` really "Absolute delivery validated"?** If it
   truly just reconstructs the controller's own internal goal, it should at best match the 71.43%
   Delta baseline, not beat it by ~16 points. Leading hypothesis: an artifact of this one
   reconstruction formula being easier to track, not a general "Absolute" finding. No Stage 1
   experiment distinguishes the two readings.
2. **Doubt 2 — does the `relative_converted` delivery trick generalize past LIBERO/robosuite?**
   It leans entirely on `output_max`, a robosuite OSC-controller normalization constant. A
   production controller (e.g. OpenArm's) with no equivalent split may make this delivery
   mechanism worthless outside LIBERO.
3. **`afo_abs_h1`'s 8.2% GT-replay failure — mechanism never pinned down.** Ruled out: target
   construction and the π-singularity explanation (round-trip math exact to 1e-16 near θ=π).
   **Never checked:** per-step state alignment *during* chunk execution (vs. at episode reset) —
   left open when the investigation moved to `relative_converted` delivery instead.
4. **Historical baseline bug, fixed but not backfilled:** the GT-replay baseline cache keyed on
   `task_index` alone, so a stale n=10 baseline (60.0% pc) silently gated n=49/50 runs. Fixed
   2026-07-13/14 (cache now keyed on `n_episodes` too; corrected fresh n=49 baseline = 71.43%).
   Historical per-run `stage_status.json` gate numbers were **not** retroactively recomputed.
5. Two result families are marked invalid, not merely provisional (`stage1-closeout.md` §2):
   historical `native_n0` (⛔ structurally locked to `per_frame_anchor=True`, never tested a real
   chunk-start anchor) and the pre-unit-mismatch `goal` family (no consistent physical unit — a
   coincidental floating-point cancellation, not a genuine absolute pose).

## Concrete next steps to resume (Stage 2, per `stage1-closeout.md` §4)

Only what the doc actually states — nothing extrapolated:

1. Move to a sim backend where `observation.state`/`action` are **genuinely unit-consistent by
   construction**, not via robosuite's `output_max` conversion — doc names **ManiSkill's
   `pd_ee_pose` controller with `normalize_action=False`** as the candidate.
2. Re-run the same **Delta / Relative / Absolute × commanded / OAA** matrix there, with no
   `relative_converted`-style conversion step needed.
3. Use that to resolve Doubt 1 (real representation finding vs. formula artifact) and Doubt 2
   (does the delivery trick generalize?) — the only stated way to tell them apart.
4. Reuse the harness seam: implement a new `Study` plugin (`anvil_sim/study.py`'s
   `Study`/`GtReplayConfig`/`ReplayAdapter`, documented in
   `packages/anvil_sim/src/anvil_sim/ARCHITECTURE.md`) rather than forking the harness.
5. If revisiting `afo_abs_h1` instead: check per-step state alignment *during* chunk execution
   (not just at `env.reset()`) — the one still-open, never-checked hypothesis for its 8.2%
   GT-replay failure.

Nothing in the docs commits the branch to a start date for Stage 2 — this is a deliberate, not
scheduled, pause.

## How to run / test

Setup (from `docs/simulation.md`):

```bash
uv sync --package anvil-sim --extra dev

# LIBERO's first import shows an interactive prompt that EOF-crashes non-interactively.
# Pre-seed its config once (see docs/simulation.md "Setup" for the exact python -c snippet).
mkdir -p ~/.libero && python3 -c "..."   # full snippet: docs/simulation.md lines ~46-56

export HF_HOME=/some/writable/cache      # optional, avoids polluting ~/.cache/huggingface
```

Build datasets once (~5-6 min; add `--max-episodes=2` for a fast smoke run):

```bash
uv run --package anvil-sim anvil-libero-convert
```

Reproduce an existing treatment through the full gated pipeline:

```bash
uv run --package anvil-sim anvil-sim-bench run \
  packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_native_n0_diffusion.yaml
uv run --package anvil-sim anvil-sim-bench status --study libero_ee   # print the ledger
```

Regenerate the G2 mode-collapse mechanism analysis:

```bash
uv run --package anvil-sim python -m anvil_sim.studies.libero_ee.analysis.mechanism_analysis
```

Debug a specific eval path with GT-replay (no policy, verbatim from `docs/simulation.md`):

```bash
uv run --package anvil-sim anvil-libero-replay \
  --action-type native_n0 \
  --dataset-root data/datasets/ee-space/libero-task10-native-n0 \
  --control-mode relative --task libero_goal --task-id 8 \
  --n-episodes 10 --n-action-steps 100 \
  --output-dir research/libero_ee/replay/debug
```

Tests:

- `anvil_sim`'s own unit tests live under the package, not `tests/unit/`: `packages/anvil_sim/tests/`
  (`test_bench_runner.py`, `test_bench_spec.py`, `test_eval_replay.py`, `test_libero_convert.py`,
  `test_libero_processor.py`).
- Related tests touched by this branch elsewhere: `tests/unit/anvil_shared/test_ee_transform.py`,
  `tests/unit/anvil_trainer/test_ee_abs_transform.py`,
  `tests/unit/anvil_trainer/test_ee_rel_transform_smoke.py`,
  `tests/unit/anvil_trainer/test_ee_validation.py`, `tests/unit/anvil_trainer/test_umi_features.py`,
  `tests/unit/mcap_converter/test_ee_encoding.py`.
- `anvil_sim` has only the `dev` extra (pytest/pytest-cov/ruff), no sim-specific extra
  (`packages/anvil_sim/pyproject.toml`).

## Confidence and scope

High confidence — every claim traces to `research/libero_ee/stage1-closeout.md`, `report.md`
Part 1, `docs/simulation.md`, or `docs/relative_ee_failure_analysis.md`'s superseded-banner, plus
`git log` and repo listings directly inspected. Not checked: full body of `research/libero_ee/diary.md`
and `report.md` §2–§7 beyond the Part 1 summary — pointed to above for anyone needing more depth.
