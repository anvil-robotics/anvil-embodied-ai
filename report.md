# EE-Space Action Representation on LIBERO — Status Report

_Anvil embodied-AI · worktree `ee-libero-benchmark` · task TASK-006_

## 1. What this effort is

Validate Anvil's end-effector (EE) space training + closed-loop inference pipeline
(`anvil_trainer`, `anvil_shared.ee_transform`, `anvil_sim`) using LeRobot's LIBERO benchmark,
instead of building a custom robot sim. Concretely: take the public `lerobot/libero` dataset,
derive Anvil-format EE datasets from it, train ACT / Diffusion policies, and run closed-loop
eval in the real `LiberoEnv` — comparing several action-space **representations** against the
untouched **native** LIBERO format as the gold reference.

- **Task:** `task_index=10` "put the bowl on the plate" (`libero_goal` suite, internal
  `task_id=8`), 49 episodes, ~93 steps/episode. Chosen as the shortest/simplest candidate.
- **Metric:** `pc_success` (closed-loop success rate), 10 eval episodes per condition.
- **Native format:** 8-dim state `[pos(3), axis-angle(3), gripper_qpos(2)]`, 7-dim action
  `[Δpos(3), Δaxis-angle(3), gripper]` — LIBERO's own recorded delta command, fed to
  robosuite's `OSC_POSE` controller.

## 2. Result ledger (all experiments so far)

| # | Condition | Representation | ACT | Diffusion |
|---|---|---|---|---|
| 1 | `native` | native delta, relative mode (gold reference) | **80%** | **100%** |
| 2 | `ee_abs` | Anvil act-from-obs absolute pose (quat+rot6d), calibrated | 40% | 50% |
| 3 | `ee_rel` | UMI-style SE(3) relative, hand frame, calibrated | 10% | 10% |
| 4 | `ee_delta` | world-frame consecutive delta, rot6d, calibrated | 10% | — |
| 5 | `native_rot6d` | native scale, rotation re-encoded axis-angle→rot6d, zero-cal | 60% | 100%* |
| 6 | zero-cal re-run | act-from-obs targets, `control_mode="absolute"`, no calibration | abs 20% / world-n0 20% / hand-n0† 10% / world-seq 10% | (not run) |
| 7 | `goal` family | formal `state+native_delta` targets (see §4) | abs 0% / world-n0 0% / hand-n0 0% / world-seq 10% / hand-seq 10% | (not run) |

\* native_rot6d Diffusion reached 100% in a follow-up run.
† Exp-6 naming used `Rel-rot6d-*`; mapped here to the world/hand + n-0/n-(n-1) grid.

**Three real bugs were found and fixed along the way** (documented in `packages/anvil_sim/README.md`):
`ee_rel` chunk-anchor mismatch; `ee_rel` double-relativization; `native`+Diffusion upstream
`lerobot` `EpisodeAwareSampler` crash. Plus two more in Experiment 7 (see §4).

## 3. The central question

Across every experiment the ranking has been stable: **native (its own delta command, fed to
relative mode) ≫ every Anvil EE re-encoding.** The open question the last several rounds have
chased is *why*, and specifically whether the gap is:

- **calibration error** (the `NATIVE_POS_SCALE`/`NATIVE_ROT_SCALE` regression, rotation R²=0.49) — **ruled out** by Exp 6 (removing it did not help; `ee_abs` even got *worse*, 40%→20%);
- **rot6d vs axis-angle rotation encoding** — measured at ~20pp (Exp 5: native 80% vs native_rot6d 60%), real but secondary;
- **the target *definition*** (absolute pose vs relative delta; act-from-obs "achieved next
  state" vs commanded goal) — the subject of Experiment 7.

## 4. Experiment 7 (just completed): negative result

**Hypothesis.** `ee_abs`/`ee_rel` use act-from-obs targets `action[t] = encode(state[t+1])`,
i.e. the physically *achieved* next state. But at eval, `control_mode="absolute"` treats the
policy output directly as a *goal*. Those differ by the impedance controller's tracking error.
Exp 7 tried to remove that mismatch by defining a single "goal" target and running all 5
control-variable conditions (abs / world-n0 / hand-n0 / world-n(n-1) / hand-n(n-1)) from it.

**Two bugs found and fixed** (both caught by validating recovery math against real episode data
to floating-point zero error *before* spending training compute — a check that was missing the
first time and let a bad run consume a full 5×50k-step sweep):

1. **Wrong scale assumption → all 5 conditions 0%.** `goal = state + native_delta × 0.05`
   (robosuite's `OSC_POSE` `output_max`) assumed 0.05 reconstructs the controller's target.
   Real per-step displacement is only **~22% of that** (fitted ≈0.011): the impedance
   controller never reaches its internal aim point in one step (normal OSC behavior). The
   ~4.5×-too-large "goal" destabilized every rollout. **Fix:** stop scaling ourselves — compose
   `state + native_delta` *formally* (unscaled, like `native_action_to_rot6d` treats rotation),
   then at eval recover a delta relative to the **real current state**, clip to [-1,1], and feed
   `control_mode="relative"` so robosuite's own `scale_action` applies the true (unknown-to-us)
   scale.
2. **Consecutive-goal construction mismatch for the seq conditions.** Building n-(n-1) from
   consecutive *goals* introduced a constant per-chunk offset (~0.26) and, because those targets
   are physical-unit displacements, would be double-scaled if delivered via "relative" (error
   0.94). **Fix:** the n-(n-1) conditions use **real consecutive states** like the existing
   `ee_delta` (so `world-n(n-1)` **reuses the existing `ee_delta` checkpoint unchanged**), and
   are delivered via `control_mode="absolute"`. Only `hand-n(n-1)` needed a genuinely new dataset.

**Result (ACT, task_index=10, 10 episodes):**

| Condition | Target construction | pc_success |
|---|---|---|
| `abs`, `world-n0`, `hand-n0` | formal `state + native_delta` (unscaled) | **0% / 0% / 0%** |
| `world-n(n-1)` | real consecutive states (= reused `ee_delta`) | **10%** |
| `hand-n(n-1)` | real consecutive states (hand frame) | **10%** |

Training losses were all normal (0.034–0.037, comparable to the working `ee_delta`'s 0.033) —
**not** a training crash or a stats-normalization bug (`_force_rot6d_identity` was checked and
ruled out).

**Core finding.** The 3 conditions built on the *formal composite goal* all get exactly 0%; the
2 built on *real achieved-state deltas* keep their established numbers. This is the **opposite**
of the hypothesis: the act-from-obs "target ≠ achieved" mismatch is not the dominant problem —
**the formal composite goal is itself a target the network cannot fit well enough to close the
loop.** Plausible mechanism: it jumps by the raw per-step command magnitude each frame (not a
smooth physical trajectory like `state[t+1]`), so small regression errors get amplified when the
current real state is subtracted back out to recover a usable delta.

## 5. Literature check — our result is corroborated, not anomalous

- **No public precedent for absolute-EE-pose ground truth on LIBERO.** LIBERO ships delta EE
  actions; OpenVLA / Octo and other common LIBERO users keep that delta convention. Nobody
  appears to re-derive LIBERO into absolute EE-pose targets.
- **"Demystifying Action Space Design for Robotic Manipulation Policies"** (arXiv 2602.23408,
  2026; 13k+ real bimanual rollouts, 500+ models) finds **delta beats absolute by 10–20pp**
  (ACT overall 78.4% delta vs 63.4% absolute), attributing it to global coordinates having
  **lower "local coherence"** (harder learning target) and to absolute control needing a
  **longer observation/execution horizon**. This directly matches our "formal absolute goal is
  hard to learn" outcome. (Their Proposition 4.1 asymptotics read the other way — noted as a
  discrepancy to verify against the source, not relied on.)
- **Caveats / non-contradictions:** ACT's original paper predicts absolute *joint* angles
  successfully — but that is joint space, not EE/task space. Diffusion Policy found absolute
  *position* > *velocity* — but that is position-vs-velocity, a different axis than absolute-vs-
  delta EE pose.

Net: in the EE/task-space, per-step-command regime we're in, the evidence (ours + the newest
systematic study) favors **delta** representations.

## 6. Current blocking issues / open questions

1. **Premise invalidated.** Experiment 7's motivating fix (goal ≠ achieved) did not pan out;
   act-from-obs `state[t+1]` remains the only target definition that has ever produced non-zero
   closed-loop success here. Continuing to push "absolute EE goal" targets looks low-value given
   both our data and the literature.
2. **Unexplained specifics (not yet investigated).** *Why exactly* the `goalabs` conditions hit
   0% — e.g. eval-video inspection, or logging per-step recovered-action magnitudes against a
   live rollout — has not been done. This is the only concrete open thread on Exp 7 itself.
3. **Original task dimensions still untouched.** The user's earlier 3-axis plan (control
   variable × task type × ACT/Diffusion) intended, after settling the control-variable axis, to
   (a) find a rotation-heavy task vs task_index=10's rotation-light baseline, and (b) run
   Diffusion for the surviving conditions. None of that has started.
4. **Everything is uncommitted.** All of this work lives in the `ee-libero-benchmark` worktree
   (branch `feat/ee-libero-benchmark`); nothing has been committed in this multi-day session.
