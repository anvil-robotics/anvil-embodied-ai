# EE-Space Action Representation on LIBERO — Report

_Anvil embodied-AI · task TASK-006 · branch `feat/ee-libero-benchmark`_

---

# Part 1 — Summary

**Goal.** Validate Anvil's end-effector (EE) space training + closed-loop inference pipeline
(`anvil_trainer`, `anvil_shared.ee_transform`, `anvil_sim`) on LeRobot's LIBERO benchmark, to
decide **which action representation the real robot should use** — via a **gated validation
harness** (`anvil-sim-bench`) so bad configs die at cheap gates instead of after a 50k-step run.

**The clean result.** Every design factor is measured as a **single-variable flip from `native`**,
holding the observation (8-dim native state) AND the trainer (`lerobot-train` raw) FIXED — the
"native family" (task10, ACT, n=50):

| flip from `native` | condition | pc | Δ |
|---|---|---|---|
| — (reference: world-frame delta command, relative delivery) | `native` | **88** | — |
| #4 frame → hand | `native_hand` | 64 | **−24** |
| #1 encoding → rot6d | `native_rot6d` | 76 | **−12** |
| #2 abs/rel → absolute goal | `native_abs` | 80 | **−8** |
| #3 anchor → n-0 | `native_n0` | 86 | −2 |

**Factor magnitude: frame (−24) ≫ encoding (−12) > abs/rel (−8) > anchor (≈0).**

**Solid conclusions.**
1. **`native` (world-frame delta command + relative delivery) is the most reliable** — across
   tasks (80–90 on task10/14/11) AND architectures (ACT 88 / Diffusion 100). It is what
   production already uses; the sweep validates that choice.
2. **Frame: world ≫ hand** (the dominant factor). Body-frame commands are *harder* to learn —
   opposite the UMI intuition. Keep world-frame actions.
3. **Encoding: axis-angle > rot6d** (secondary, ACT only; neutral for Diffusion).
4. **Relative > absolute** (sign-corrected): the delta command beats the absolute-goal
   formulation once observation and encoding are held fixed. The earlier "absolute wins +18" was
   an observation-encoding confound → the goal-style production idea (P2) is **not worth pursuing**.
5. **Anchor ≈ no effect, and not independently isolable** in the command family: a true
   chunk-start (n-0) anchor can't be encoded in a static training column; it degenerates to the
   command. Genuine n-0 needs chunk-aware training.
6. **The harness earned its keep:** its GT-replay gate caught **5 real eval-path bugs**, every one
   invisible to training loss and to synthetic-value unit tests.

**Production recommendation:** keep the native world-frame delta command + relative delivery; do
not adopt goal-style targets or body-frame actions; avoid anchor-relative representations
(Diffusion-fragile). Details in Part 3 §8. **Caveats:** n=50, `libero_goal` tasks only,
goal-family Diffusion only on task10.

---

# Part 2 — Diary

Chronological log, **including the wrong turns**. Tags: `[exp]` ran an ablation · `[result]`
measured outcome · `[bug]` found a bug / flawed design · `[fix]` fixed it · `[infra]` built or
extended the harness / a representation · `[insight]` conclusion or reversal.

### 2026-07-03
- `[infra]` Chose LIBERO (native EE actions via robosuite OSC — no custom kinematics; public
  `lerobot/libero` dataset) over building a custom sim. Picked `task_index=10` "put the bowl on
  the plate" (shortest, ~93 steps). Set up 3 arms: `native` / `ee_abs` / `ee_rel`.
- `[bug]` `ee_rel` behaved visibly worse. It relativized against the *fresh* observation each
  step, but the policy replans only once per chunk — the reference must be the chunk-start obs.
  `[fix]` track the chunk boundary with a call counter.
- `[result]` `ee_rel` 0% on both ACT and Diffusion despite normal training loss. `[bug]`
  **double-relativization** — `libero_convert` pre-relativized the dataset AND the trainer
  relativized again. `[fix]` store absolute actions, relativize exactly once at train → `ee_rel` 10%.
- `[bug]` `native`+Diffusion crashed (`IndexError` in the DataLoader) — an upstream `lerobot`
  `EpisodeAwareSampler` bug (global frame indices on a filtered dataset). `[fix]` write a local
  unfiltered `native` dataset to bypass it.

### 2026-07-06
- `[bug]` Confirmed the `native`+Diffusion crash is purely the upstream `lerobot` sampler bug
  (reproduced with plain `lerobot-train`, no Anvil code) — not fixable inside anvil-trainer.

### 2026-07-07
- `[exp]` **Exp 4** `ee_delta` (world-frame delta, rot6d) = 10%. `[insight]` Confounded — it
  varied the rotation encoding AND removed the closed-loop correction at once; doesn't isolate rot6d.
- `[exp]` **Exp 5** `native_rot6d` (clean rot6d isolation, zero calibration) = 60% vs native 80%.
  `[insight]` rot6d encoding costs ~20pp — real but secondary.
- `[exp]` **Exp 6** zero-cal re-run (`control_mode=absolute`): `ee_abs` got *worse*. `[insight]`
  calibration ruled out as the cause.
- `[exp]` **Exp 7** goal family (formal `state+native_delta` targets, 5 conditions) — all 0%.
  `[insight]` (wrong) concluded "the formal goal is unlearnable" — a premature negative result.
- `[infra]` **Built the harness:** `anvil-sim-bench` (8-stage gated pipeline: convert →
  validate-math → dataset-validate → **gt-replay** → smoke → train → eval → record) +
  `anvil-libero-replay` (GT-replay: run the dataset's own ground truth through the eval path, no
  policy). Motivation: stop burning 50k-step runs on broken eval paths.
- `[bug]` **Exp 8 / Bug #4:** GT-replay's very first run caught a gripper-semantics bug —
  `goalabs` stored LIBERO's native ±1 gripper *command* but the bang-bang comparator expected
  qpos-scale targets, so the gripper never closed → every `goalabs` rollout pinned to 0%.
  `[fix]` `gripper_mode="native_cmd"`. `[insight]` **reversed Exp 7: goal-abs 0% → 100%** (best
  ACT result) — the "unlearnable" conclusion was a bug artifact.

### 2026-07-08
- `[bug]` **Exp 9 / Bug #5:** chunk-anchor state leaked across episodes (the call counter never
  reset per episode). Invisible at the gate's n_action_steps=1; `[insight]` caught by running
  GT-replay at n=100 (world-n0 scored 20% where the identity predicts 80%). `[fix]`
  `reset_episode_state()` per episode. `[result]` re-eval: world-n0/hand-n0 40/30 → 80/80 —
  **reversed the "anchor-relative is harmful" claim**.
- `[fix]` `--force-stage eval` was a silent no-op (reused the cached `eval_info.json`); made it
  actually re-run.
- `[exp]` **n=50** confidence sweep + **task14** (medium rotation) + **task11** (heavy).
  `[result]` goal-abs non-monotonic (94/72/76) — its task10 lead does NOT generalize; `native`
  most robust across all three.
- `[exp]` **E2 Diffusion** for the goal family. `[result]` goal-abs architecture-robust (ACT 94 /
  Diffusion 98); `[insight]` anchor-relative `world-n0` architecture-**fragile** (ACT 82 →
  Diffusion 16 collapse, with normal loss and a healthy GT-replay).
- `[infra]` **Frame-gap:** built `native_hand` (native command rotated into the hand/body frame)
  for a clean frame isolation. `[result]` world 88 vs hand 64 = **−24** — frame is the dominant
  factor; body-frame is harder to learn (contra UMI).
- `[bug]` **Methodology confound (user-caught):** #2/#3 had compared the goal family (10-dim obs,
  anvil-trainer) against native (8-dim obs, lerobot-train) — confounding the observation encoding.
  `[fix]` rebuilt `native_abs`/`native_n0` inside the native family. `[insight]` **#2 sign
  flipped: relative wins (−8), not "absolute +18"** — the goal-abs edge was the richer 10-dim obs.

### 2026-07-09
- `[bug]` `native_n0` closed-loop collapsed (10%); GT-replay healthy at n=1 but 0% at n=100 — a
  chunk-anchor mismatch. `[insight]` a true chunk-start (n-0) anchor CANNOT be encoded in a static
  lerobot-train column (chunks exist only at inference) → the anchor factor is not independently
  isolable in the command family. `[fix]` anchor the eval reconstruction per-frame
  (`per_frame_anchor`). `[result]` native_n0 = 86 ≈ native (degenerate, as predicted).
- `[result]` **Final clean native-family matrix** (Part 1): frame (−24) ≫ encoding (−12) >
  abs/rel (−8) > anchor (≈0); `native` is the most reliable representation.
- `[fix]` **P1 production:** fixed the analog of Bug #5 in `implement-ee-space` —
  `inference_node.reset_policy` never cleared the chunk-anchor state and nothing reset it at
  episode boundaries (branch `fix/inference-episode-reset`).

---

# Part 3 — Technical details

_The original per-factor analysis, ledger, bug catalog, literature check and production backlog.
Experiments 7 (§4) and the literature note (§5) are kept as originally written for the reasoning
trail — read them with Part 1/2 and the bug corrections above._

## 1. What this effort is

Validate Anvil's end-effector (EE) space training + closed-loop inference pipeline
(`anvil_trainer`, `anvil_shared.ee_transform`, `anvil_sim`) using LeRobot's LIBERO benchmark,
instead of building a custom robot sim. Concretely: take the public `lerobot/libero` dataset,
derive Anvil-format EE datasets from it, train ACT / Diffusion policies, and run closed-loop
eval in the real `LiberoEnv` — comparing several action-space **representations** against the
untouched **native** LIBERO format as the gold reference.

The primary deliverable is now the **validation harness itself** (`anvil-sim-bench`): a gated,
idempotent pipeline that dies at cheap checks before spending training compute, so any new
treatment can be verified with minimum effort. The experiments below doubled as its
requirements discovery and its end-to-end acceptance test.

- **Task (baseline):** `task_index=10` "put the bowl on the plate" (`libero_goal` suite, internal
  `task_id=8`), 49 episodes, ~93 steps/episode. Chosen as the shortest/simplest candidate.
- **Task (rotation-heavy contrast):** `task_index=11` "put the wine bottle on the rack"
  (`libero_goal`, `task_id=9`), rotation RMS 0.0258 ≈ **3.1×** task10's — see §7.
- **Metric:** `pc_success` (closed-loop success rate), 10 eval episodes per condition.
- **Native format:** 8-dim state `[pos(3), axis-angle(3), gripper_qpos(2)]`, 7-dim action
  `[Δpos(3), Δaxis-angle(3), gripper]` — LIBERO's own recorded delta command, fed to
  robosuite's `OSC_POSE` controller.

## 2. Result ledger — current (post bug #4 + bug #5, n=50 for live conditions)

Machine-generated ledger of record: `outputs/bench/RESULTS.md` (regenerated by
`anvil-sim-bench`, never hand-edited). The surviving conditions were re-run at **n=50**
episodes (experiment E1); deprecated and `seq` rows remain at n=10 (noted). task10:

| Condition | Representation | ACT | Diffusion | n |
|---|---|---|---|---|
| `native` | native delta command, WORLD frame, relative delivery (gold reference) | **88** | 100 | 50 / 10 |
| `native_rot6d` | native family, action rot6d (#1 encoding flip) | **76** | 100 | 50 / 10 |
| `native_abs` | native family, absolute goal `state+native_delta` (aa) (#2 abs/rel flip) | **80** | — | 50 |
| `native_n0` | native family, goal per-frame-relativized (aa) (#3 anchor flip) | **86** | — | 50 |
| `native_hand` | native command rotated to HAND (body) frame (#4 frame flip) | **64** | — | 50 |
| `goal-abs` | formal `state+native_delta`, recovered-delta relative delivery (goal family, 10-dim obs) | **94** | **98** | 50 |
| `goal-world-n0` | goal target, world-frame anchor-relative (n-0), relative | **82** | **16** | 50 |
| `goal-hand-n0` | goal target, hand-frame anchor-relative (n-0), relative | **76** | — | 50 |
| `goal-world-seq` | real consecutive states (n-(n-1)), **absolute** delivery | 50 | — | 10 |
| `goal-hand-seq` | real consecutive states, hand frame, **absolute** delivery | 30 | — | 10 |
| ~~`ee_abs`~~ (deprecated) | Anvil act-from-obs absolute pose, calibrated | 40 | 50 | 10 |
| ~~`ee_rel`~~ (deprecated) | UMI-style SE(3) relative, hand frame, calibrated | 30 | 10 | 10 |
| ~~`ee_delta`~~ (deprecated) | world-frame consecutive delta, calibrated | 10 | — | 10 |

task11 (rotation-heavy) and task14 (medium) live conditions are in §7. **Two additional
conditions — `goal-{world,hand}-seq` with RELATIVE delivery — were gate-rejected** (GT-replay
0%): recovered-delta relative delivery double-scales the physical-unit seq deltas (§3.4/§3.5).

Deprecated `ee_abs`/`ee_rel`/`ee_delta` are kept for the reasoning trail (their story is bugs
#1–#3) but **excluded from the control-factor analysis in §3**.

**Five real eval-path bugs were found and fixed** (documented in
`packages/anvil_sim/README.md`): (#1) `ee_rel` chunk-anchor mismatch; (#2) `ee_rel`
double-relativization; (#3) `native`+Diffusion upstream `lerobot` `EpisodeAwareSampler` crash;
(#4) `goalabs` gripper semantics; (#5) chunk-anchor state leaking across episodes. Every one
was invisible to training loss (all converged normally) and to synthetic-value unit tests;
each was caught only by running real data through the actual eval path.

**Five real eval-path bugs were found and fixed** across the effort (all documented in
`packages/anvil_sim/README.md`): (#1) `ee_rel` chunk-anchor mismatch; (#2) `ee_rel`
double-relativization; (#3) `native`+Diffusion upstream `lerobot` `EpisodeAwareSampler` crash;
(#4) `goalabs` gripper semantics; (#5) chunk-anchor state leaking across episodes. Every one
was invisible to training loss (all converged normally) and to synthetic-value unit tests;
each was caught only by running real data through the actual eval path.

## 3. Analysis by control factor — clean native-family single-flip design

Each factor is measured by flipping **exactly one** variable from `native`, holding the
observation (8-dim native state) AND the trainer (`lerobot-train` raw) FIXED — i.e. all
conditions live in the **native family**. This matters: the goal family (`goal-abs`,
`goal-world-n0`, …) trains via `anvil-trainer` on a **different 10-dim Anvil-EE observation**, so
comparing native against a goal-family condition confounds the design factor with the observation
encoding. Only same-family, single-variable flips are clean. All numbers task10, ACT, n=50.

### 3.0 The clean matrix (all native-family, single flip from `native`)

| flip from native | condition | pc | Δ vs native | GT-replay (math check) |
|---|---|---|---|---|
| — (reference: aa, delta, world, n-(n-1), relative) | `native` | **88** | — | baseline 60 |
| #4 frame → hand | `native_hand` | 64 | **−24** | 71 (healthy) |
| #1 encoding → rot6d | `native_rot6d` | 76 | **−12** | 60 (healthy) |
| #2 abs/rel → absolute goal | `native_abs` | 80 | **−8** | 86 (healthy) |
| #3 anchor → n-0 | `native_n0` | 86 | **−2** | 60 (healthy) |

Every condition's GT-replay ≈ the native baseline, so the eval paths execute ground truth
correctly and each gap is a genuine **learnability** difference, not an eval artifact. **Factor
magnitude: frame (−24) ≫ encoding (−12) > abs/rel (−8) > anchor (≈0).**

### 3.1 Frame — world vs hand: **−24 (dominant factor)**
`native` (world) 88 vs `native_hand` (same command rotated into the EE body frame) 64. **World
frame wins by 24pp.** The body-frame command is substantially *harder to learn* — **opposite** to
the common "body-frame is more local, hence easier" (UMI) intuition. (Goal-family cross-check,
different obs: world-n0 82 vs hand-n0 76, +6 — same direction, smaller.)

### 3.2 Rotation encoding — axis-angle vs rot6d: **−12**
`native` (axis-angle) 88 vs `native_rot6d` 76. axis-angle > rot6d for ACT; **neutral for
Diffusion** (100=100). Cross-task: the penalty **shrinks as rotation grows** (task14 −14, task11
only −2) — opposite the naive expectation.

### 3.3 Absolute vs relative target — **relative wins, −8** (sign-corrected)
`native` (relative delta command) 88 vs `native_abs` (absolute goal `state+native_delta`, same
command, recovered-delta relative delivery) 80. **The relative command beats the absolute-goal
formulation by 8pp.**
> **Correction.** An earlier ENCODING-CONFOUNDED pairing (goal-abs rot6d **94** vs native_rot6d
> 76) suggested the *opposite* — "absolute +18". On the clean native family (observation and
> encoding held fixed) the sign flips: relative wins. The apparent goal-abs advantage was the
> goal family's richer 10-dim observation, not the absolute representation. This is exactly why
> same-family isolation was necessary.

### 3.4 Anchor — n-(n-1) vs n-0: **≈ none (−2), and not independently isolable in this family**
`native` (n-(n-1)) 88 vs `native_n0` (n-0) 86 — no meaningful effect. But the deeper finding is
structural: **a true chunk-start (n-0) anchor cannot be encoded in a static `lerobot-train`
dataset column**, because chunks exist only at inference. A per-frame-consistent "n-0"
degenerates to the command itself (`ee_rel_world_forward(state+native_delta, state) ≈
native_delta`), which is why native_n0 ≈ native. Genuine chunk-start anchoring requires
**chunk-aware training** (the goal family / anvil-trainer). So the anchor factor is **entangled
with the training mechanism**, not an independent single-flip in the command family. (This was
surfaced by a chunk-anchor eval bug — native_n0 GT-replay healthy at n_action_steps=1 but 0% at
n=100 — fixed by anchoring the eval reconstruction per-frame to match the per-frame training
target; `ZeroCalActionProcessorStep.per_frame_anchor`.)

### 3.5 Secondary cross-checks (goal family — different 10-dim obs, NOT clean single-flips)
These use the anvil-trainer 10-dim observation, so read them as directional, not as clean isolations:
- **Delivery is not a free flip.** Formal-goal targets (`goal-abs`, `n-0`) are unscaled → recovered-delta
  **relative** is correct (goal-abs 94, world-n0 82). Consecutive-`seq` targets are physical-unit →
  must be **absolute** (world-seq 50, hand-seq 30); relative delivery double-scales them → 0%
  (gate-rejected).
- **Task robustness** (ACT, n=50): `native` 88/90/80 across task10/14/11 — the most robust.
  `goal-abs` is **non-monotonic** (94/72/76): tops native only on rotation-light task10, trails on
  the other two. Its task10 lead does not generalize.
- **Architecture** (task10): `goal-abs` robust (ACT 94 / Diffusion **98**); `goal-world-n0`
  **fragile** (ACT 82 → Diffusion **16** collapse, normal loss + healthy GT-replay — a real
  representation×architecture interaction, not a bug).

### Bottom line
- **Frame dominates: world ≫ hand (−24).** Keep world-frame actions; body-frame is harder to learn
  (contra UMI intuition).
- **Encoding secondary: axis-angle > rot6d (−12) for ACT, neutral for Diffusion.**
- **Relative > absolute (−8):** the native delta command beats the absolute-goal formulation once
  observation/encoding are held fixed — the sign is opposite to the confounded comparison.
- **Anchor ≈ no effect (−2), and not independently isolable** in the command family (true n-0 needs
  chunk-aware training).
- **Across tasks & architectures, `native` (world-frame delta command, relative delivery) is the
  most reliable** — which is what production already uses. `goal-abs` is a task-specific,
  obs-dependent option, not a blanket upgrade.
- **Standing caveat:** task10 primary (task11/14 for robustness), goal-family Diffusion only on task10.

## 4. Experiment 7 (original text): "negative result" — SUPERSEDED

_Kept verbatim for the reasoning trail. Its conclusion was reversed twice: bug #4 lifted
`goal-abs` from 0%→100% (Exp 8), and bug #5 lifted `world-n0`/`hand-n0` from 40/30→80/80
(Exp 9). Read §2/§3 for the current numbers._

**Hypothesis.** `ee_abs`/`ee_rel` use act-from-obs targets `action[t] = encode(state[t+1])`,
i.e. the physically *achieved* next state. But at eval, `control_mode="absolute"` treats the
policy output directly as a *goal*. Those differ by the impedance controller's tracking error.
Exp 7 tried to remove that mismatch by defining a single "goal" target and running all 5
control-variable conditions (abs / world-n0 / hand-n0 / world-n(n-1) / hand-n(n-1)) from it.

**Two dataset/math bugs found and fixed** (both caught by validating recovery math against real
episode data to floating-point zero error *before* spending training compute):

1. **Wrong scale assumption.** `goal = state + native_delta × 0.05` assumed 0.05 reconstructs
   the controller's target; real per-step displacement is only ~22% of that. **Fix:** compose
   `state + native_delta` *formally* (unscaled), recover a delta against the real current state
   at eval, clip to [-1,1], deliver via `control_mode="relative"` so robosuite applies the true
   scale.
2. **Consecutive-goal construction mismatch for seq conditions.** **Fix:** the n-(n-1)
   conditions use **real consecutive states** like `ee_delta` (so `world-n(n-1)` reuses the
   `ee_delta` checkpoint) delivered via `control_mode="absolute"`.

**Original (pre-bug-#4/#5) result:** `abs`/`world-n0`/`hand-n0` all 0%, `world-seq`/`hand-seq`
10%. The original "core finding" — that the formal composite goal is a target the network
cannot fit — is now known to be an artifact of bugs #4 (gripper) and #5 (anchor leak), not a
property of the target. See §3.

## 5. Literature check — still corroborated, with a sharper reading

- **No public precedent for absolute-EE-pose ground truth on LIBERO.** LIBERO ships delta EE
  actions; OpenVLA / Octo keep that delta convention.
- **"Demystifying Action Space Design for Robotic Manipulation Policies"** (arXiv 2602.23408,
  2026; 13k+ real rollouts) finds **delta beats absolute by 10–20pp**, attributing it to global
  coordinates having lower "local coherence". Our post-fix result refines rather than
  contradicts this: `goal-abs` is an **absolute representation with delta execution** — it keeps
  the drift-correcting absolute target while issuing a controller-native delta command, i.e. the
  "delta beats absolute" comparisons conflate representation with delivery. (Their Prop 4.1
  asymptotics read the other way — noted as a discrepancy to verify, not relied on.)
- **Caveats:** ACT's paper predicts absolute *joint* angles (not EE/task space); Diffusion
  Policy found absolute *position* > *velocity* (a different axis).

## 6. Real bug #5 (Experiment 9) — chunk-anchor state leaked across episodes

`ACT`/`Diffusion` `select_action()` only run the model every `n_action_steps` calls; the
stateful eval processors (`AnvilEEActionProcessorStep`, `ZeroCalActionProcessorStep`) track that
chunk boundary with a call counter and cache a per-chunk anchor. But `rollout()` calls
`policy.reset()` at every episode start (the policy replans from episode-local step 0) **without
resetting the env processors** — so the counter ran continuously across episodes. Unless an
episode's length happened to be a multiple of `n_action_steps`, every episode after the first
reconstructed its first targets against an anchor captured mid-chunk — initially one from the
*previous episode's* scene.

- **Why nothing caught it earlier:** training loss is eval-independent; unit tests used
  single-episode synthetic values; and the `gt-replay` gate runs at `n_action_steps=1`, where
  every step is a chunk start so the leak is invisible. It only bites at the policy's real
  `n_action_steps` (>1) from episode 2 on.
- **How it was exposed:** a GT-replay of `world-n0` at `n_action_steps=100` scored **20%**,
  where the forward/inverse identity predicts parity with the abs condition; after the fix the
  same replay scored **80%**.
- **Fix:** `reset_episode_state()` on both stateful steps (clears `_call_count`,
  `_chunk_anchor`, and the seq accumulator `_running_target`), called at every episode start via
  a `rollout` wrapper in `eval_libero_ee` (policy eval) and directly in `eval_replay` (GT replay).
  Regression tests in `test_libero_processor.py`. Corrected numbers: §2.

Two secondary findings from the same diagnostic sweep still stand:

- The Experiment 6 absolute-mode "undershoot" hypothesis is **refuted**: `delta`-family replay
  through `control_mode="absolute"` moves 1.48× MORE than the demo per step yet ends 0.675 m
  off-course — absolute delivery **diverges** in open loop, it does not lag. The
  controller-level mechanism remains open.
- With the eval path fully correct, anchor-relative `n-0` re-encoding is **not** a penalty on
  task10 (world-n0/hand-n0 = 80% = native); it neither helps nor hurts relative to native, and
  only `goal-abs` (plain absolute goal) clears native.

## 7. Robustness across the rotation axis — task10 / task14 / task11

The task10 ranking was tested on two more `libero_goal` tasks spanning the rotation axis, each run
end-to-end through the harness (fresh convert, new-task replay baseline, 50k training, eval,
ledger) with zero hand-written scripts — the harness's end-to-end acceptance on new tasks:

- **task10** "put the bowl on the plate" — rotation-light, rot_rms 0.0082
- **task14** "put the wine bottle on top of the cabinet" — medium, rot_rms 0.0125 (same object as task11)
- **task11** "put the wine bottle on the rack" — rotation-heavy, rot_rms 0.0258 (3.1× task10)

| Condition (ACT, n=50) | task10 | task14 | task11 |
|---|---|---|---|
| `native` | 88 | **90** | 80 |
| `native_rot6d` | 76 | 76 | 78 |
| `goal-abs` | **94** | 72 | 76 |
| goal-abs − native | +6 | −18 | −4 |

**Finding.** `goal-abs`'s task10 lead over native (+6) **does not generalize and is not even
monotonic** in rotation: on the medium task14 it trails native by 18pp, on the heavy task11 by 4pp.
`native` is 80–90 across all three (most robust); `native_rot6d` is flat (76–78). So `goal-abs` is
a rotation-light-task specialist, not a general upgrade — exactly the task-dependence the harness
exists to expose cheaply, rather than trusting task10's single headline number.

## 8. Production implications & open questions

**What the control-factor sweep says for the real-robot EE pipeline (`implement-ee-space`):**

1. **Keep `native`-style world-frame delta commands + relative delivery.** The clean
   same-family single-flip sweep (§3.0) confirms every flip AWAY from native costs: frame→hand −24,
   encoding→rot6d −12, target→absolute −8, anchor→n-0 ≈0. It is also the most robust across tasks
   (80–90) and architectures (ACT 88 / Diffusion 100), and is what production already uses. The
   sweep validates that choice.
2. **World frame, not hand frame** (§3.1): the dominant factor (−24 on the clean isolation);
   body-frame is harder to learn here (contra UMI). Production already delivers world-frame poses —
   keep it; do not switch to body-frame.
3. **Relative delta beats absolute goal (−8, sign-corrected §3.3), so DROP the goal-style (P2)
   idea for production.** On the clean native family (obs fixed), the absolute-goal formulation
   *loses* to the native delta command; the earlier "absolute wins +18" was an observation-encoding
   confound. `goal-abs` only looks good in its own 10-dim-obs goal family, and even there is
   task-specific (§3.5, wins only task10). → P2 (goal-style targets) is **not worth pursuing**.
4. **Avoid anchor-relative representations** (§3.7): architecture-fragile (Diffusion 16%).
5. **`native_rot6d` is a safe secondary** (flat 76–78 across tasks, 100 on Diffusion): production's
   rot6d choice costs ~12pp for ACT but is neutral for Diffusion — acceptable, no action needed.
6. **P1 shipped:** the production episode-boundary reset bug (`inference_node.reset_policy` not
   clearing `_delta_ref_state`/`_abs_shadow_queue`; no `/eval/episode_start` subscription) — the
   direct analog of sim bug #5 — is fixed on branch `fix/inference-episode-reset` (independent PR).

**Open questions (not blocking):**
- **Absolute-delivery divergence** (E5, not yet run): why `control_mode="absolute"` diverges
  open-loop even with exact per-step targets, and whether that is OSC-specific (would tell us if it
  matters for the real controller). Deferred.
- **Diffusion coverage** beyond task10 for the goal family.
- **Larger task set / non-`libero_goal` suites** to confirm the world>hand and native-robustness
  findings generalize.

**Status.** Sim work on `feat/ee-libero-benchmark` (pushed, no PR — per earlier decision).
Production fix on `fix/inference-episode-reset` (pushed). Harness registry refactor (§0 gap) still
pending. Ledger of record: `outputs/bench/RESULTS.md`.
</content>
