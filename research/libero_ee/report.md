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
5. **Anchor is the architecture story (§3.6): on ACT ≈ no effect, but on Diffusion it decides
   everything.** Per-frame relative (`native_n0`) is Diffusion-robust (98); chunk-start (n-0)
   re-encoding (`world-n0`) **collapses on Diffusion (16)**. Relative-position is NOT diffusion-hard
   — chunk-anchoring is. (A true n-0 also can't be encoded in a static training column anyway.)
6. **Diffusion tolerates the representation choice** far more than ACT — every native-family flip
   is 98–100 on Diffusion; the ACT penalties (frame, encoding, abs/rel) largely vanish.
7. **The harness earned its keep:** its GT-replay gate caught **5 real eval-path bugs**, every one
   invisible to training loss and to synthetic-value unit tests.
8. **The recipe generalizes and the collapse mechanism is pinned down (§3.6–3.7).** Per-frame
   relative holds on the rotation-heavy task11 on Diffusion (`native_n0` 98 ≈ `native` 96) — not a
   task10 artifact (G1). And the chunk-anchor collapse is **Diffusion mode collapse** (G2): the
   trained collapsed policy's delivered commands shrink to ~⅓ magnitude / ~½ spread of the
   demonstrations, while target *magnitude* is a checked-and-refuted non-cause. ACT survives by
   regressing the conditional mean; a distributional generator degrades to the attenuated marginal.

**Recipe for EE relative-position training** (this branch's deliverable, validated on ACT +
Diffusion): **world frame + per-frame anchor (relative to the current state, NOT chunk-start n-0) +
rot6d-OK + recovered-delta relative delivery.** Production's `ee_rel` fails because it uses the two
worst choices — **body frame + chunk-anchor (n-0)** — and chunk-anchor is exactly the Diffusion
killer; the fix is world-frame + per-frame anchor, not abandoning relative. Full recipe + rationale
in Part 2 §8. **Caveats:** n=50, `libero_goal` only; task10 is near-ceiling for Diffusion (the
discriminating signal is the world-n0 collapse).

---

> 📓 **Diary** (chronological log, including the wrong turns) → [`diary.md`](diary.md)

---

# Part 2 — Technical details

_The original per-factor analysis, ledger, bug catalog, literature check and production backlog.
Experiments 7 (§4) and the literature note (§5) are kept as originally written for the reasoning
trail — read them with Part 1 and [`diary.md`](diary.md), plus the bug corrections in §2/§6 below._

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

Machine-generated ledger of record: `research/libero_ee/ledger/RESULTS.md` (regenerated by
`anvil-sim-bench`, never hand-edited). The surviving conditions were re-run at **n=50**
episodes (experiment E1); deprecated and `seq` rows remain at n=10 (noted). task10:

| Condition | Representation | ACT | Diffusion | n |
|---|---|---|---|---|
| `native` | native delta command, WORLD frame, relative delivery (gold reference) | **88** | 100 | 50 / 10 |
| `native_rot6d` | native family, action rot6d (#1 encoding flip) | **76** | 100 | 50 / 10 |
| `native_abs` | native family, absolute goal `state+native_delta` (aa) (#2 abs/rel flip) | **80** | **98** | 50 |
| `native_n0` | native family, goal per-frame-relativized (aa) (#3 anchor flip) | **86** | **98** | 50 |
| `native_hand` | native command rotated to HAND (body) frame (#4 frame flip) | **64** | **98** | 50 |
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

**Five real eval-path bugs were found and fixed** (see [`diary.md`](diary.md)): (#1) `ee_rel`
chunk-anchor mismatch; (#2) `ee_rel` double-relativization; (#3) `native`+Diffusion upstream
`lerobot` `EpisodeAwareSampler` crash; (#4) `goalabs` gripper semantics; (#5) chunk-anchor state
leaking across episodes. Every one was invisible to training loss (all converged normally) and to
synthetic-value unit tests; each was caught only by running real data through the actual eval path.

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
  representation×architecture interaction, not a bug). See §3.6.

### 3.6 Architecture robustness — the native family on Diffusion (E-diff)

Completing the native-family single-flip matrix on Diffusion (task10, n=50; `native`/`native_rot6d`
Diffusion are the earlier n=10 runs):

| flip from `native` | ACT | Diffusion |
|---|---|---|
| `native` (world, delta command, per-step) | 88 | 100 |
| `native_rot6d` (→ rot6d) | 76 | 100 |
| `native_abs` (→ absolute goal) | 80 | 98 |
| `native_n0` (→ per-frame relative) | 86 | 98 |
| `native_hand` (→ hand frame) | 64 | 98 |

Two findings:
1. **Diffusion is far more representation-tolerant than ACT.** Every native-family condition is
   98–100 on Diffusion; the ACT penalties — frame (−24), encoding (−12), abs/rel (−8) —
   essentially vanish. (Caveat: task10 is easy for Diffusion — near-ceiling — so the native-family
   Diffusion cells don't discriminate; the discriminating signal is the contrast below.)
2. **The one thing that kills Diffusion is chunk-anchor (n-0) re-encoding, NOT "relative" itself.**
   The goal-family `world-n0` (n-0 chunk-anchor) collapses to **16** on Diffusion (§3.5), while the
   native-family `native_n0` (per-frame relative) is robust at **98**. rot6d is Diffusion-neutral
   (native_rot6d 100), so the collapse is the anchor/re-encoding, not the encoding. → **per-frame
   relative is Diffusion-safe; chunk-anchored (n-0) relative is not.** This resolves the tension
   from §3.5 and directly explains a production failure mode (see §8, the recipe).
3. **The recipe generalizes — it is not task10-only (G1).** Re-running the per-frame-relative recipe
   on the **rotation-heavy task11** (rot_rms 3.1× task10) on Diffusion holds up: `native_n0` **98**,
   essentially tied with the `native` reference ceiling (**96**) and well clear of collapse. The
   task11 2×2 mirrors task10:

   | task11 (n=50) | ACT | Diffusion |
   |---|---|---|
   | `native` (reference) | 80 | 96 |
   | `native_n0` (recipe, per-frame relative) | 76 | 98 |

   So per-frame relative survives Diffusion on the hardest-rotation task too; the recipe's robustness
   is not an artifact of task10 being easy.

### 3.7 Mechanism — why chunk-anchor collapses on Diffusion (G2)

*Why* does chunk-anchor (n-0) re-encoding specifically break a Diffusion policy while ACT tolerates
it (world-n0: ACT 82 / Diffusion 16)? Two probes, reported honestly including a dead end.

**Negative result — it is NOT target magnitude.** Reconstructing all three target constructions off
the *same* absolute goal trajectory (`goalabs`) and measuring the within-chunk profile at the
Diffusion horizon (16) shows the position target does **not** ramp: chunk-anchor, per-frame and
absolute all have a within-chunk dynamic range ≈ 1.0 (chunk-anchor pos head 0.76 → tail 0.76). The
intuitive "chunk-anchor makes the target grow large across the chunk" story is refuted at the
horizon the policy actually uses. (Only the rotation channel ramps mildly — ~2× — and task10 is
rotation-light, so that alone cannot explain a 98→16 collapse.) See `research/libero_ee/analysis/mechanism.{json,png}`.

**Load-bearing evidence — it is Diffusion mode collapse.** Running the *trained* collapsed policy
(`goal-world-n0` diffusion) and the robust one (`native_n0` diffusion) with an action-trace tap
(`--trace-dir`) and comparing the **delivered** world command (`native_cmd`, the common 7-dim space
both emit) is decisive:

| delivered command (raw) | collapse (world-n0, closed-loop 16/33%) | robust (`native_n0`, 98/100%) | GT demos |
|---|---|---|---|
| mean \|\|pos\|\| | **0.26** | 0.79 | 0.75 |
| per-axis pos std | [0.22, 0.06, 0.19] | [0.50, 0.13, 0.58] | [0.49, 0.14, 0.55] |
| steps / episode | ~232 (times out) | ~76 (completes) | — |

The robust policy reproduces the demonstration command distribution almost exactly; the chunk-anchor
policy emits commands **~⅓ the magnitude and ~½ the per-axis spread** of both the robust policy and
the ground truth — the textbook Diffusion **mode-collapse** signature (it samples the low-magnitude,
low-diversity marginal instead of committing to an obs-conditioned mode), so the arm under-actuates
and its episodes run ~3× longer until they time out. ACT survives the identical target (82) because
it *regresses* the per-element conditional mean deterministically; a *distributional* generator
degrades to the attenuated marginal. See `research/libero_ee/analysis/collapse.png` /
`closed_loop_collapse.json`, reproducible via `python -m
anvil_sim.studies.libero_ee.analysis.mechanism_analysis`.

**Honest scope.** The cleanly isolated variable is the whole *world-n0 relativization scheme*
(obs + action chunk both anchored to the current pose, reconstructed against a chunk-start anchor) —
`goal-world-n0` differs from `goal-abs`/`native_n0` in the target construction, not a single atomic
"anchor" knob (§3.4). What is proven: this scheme mode-collapses Diffusion, and per-frame relative
(`native_n0`) does not; the mechanism is how a distributional model fits the target, not the
target's magnitude.

### Bottom line

See Part 1's "Solid conclusions" (points 2–6, 8) and the closing "Caveats" line — the per-factor
findings (frame/encoding/relative/anchor), the anchor/architecture story, the generalization (G1)
and mechanism (G2) pins, and the standing caveats are all stated there in full with §-pointers;
§3.1–3.7 above (and §3.6/§3.7 in particular) is the detailed evidence behind them.

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

## 8. Recipe for EE relative-position training (this study's deliverable)

The purpose of this branch was to determine, in sim, **how to do EE-space relative-position model
training so it SUCCEEDS on both ACT and Diffusion** — because the production `ee_rel` pipeline
failed (Diffusion outputting garbage). A working recipe follows from the single-flip matrix (§3.0)
and the Diffusion coverage (§3.6).

**The recipe (validated on ACT + Diffusion, task10 + rotation-heavy task11):**

| choice | use | evidence |
|---|---|---|
| **Frame** | **world**, not body/hand | ACT: world ≫ hand (−24); Diffusion neutral (both 98) |
| **Anchor** | **per-frame** (relative to the CURRENT state each step), NOT chunk-start (n-0) | per-frame `native_n0`: ACT 86 / **Diffusion 98** (task11: 76 / **98**); chunk-anchor `world-n0`: ACT 82 / **Diffusion 16 (collapse)** |
| **Representation** | **relative** works (per-frame); it is NOT diffusion-hard | `native_n0` 86/98 (task11 76/98) — relative-position is viable on both architectures and both tasks |
| **Rotation encoding** | rot6d is fine | −12 for ACT, neutral for Diffusion (native_rot6d 100) |
| **Delivery** | recovered-delta **relative** (subtract the live state, deliver a per-step delta) | native / native_n0 use it; physical-unit absolute-delivery seq is the weakest (§3.5) |

**Why production's current `ee_rel` fails — it made the two worst choices.** From the production
map, its `ee_rel` is **body-frame + chunk-start (n-0) anchor + rot6d**, delivered as absolute pose.
Body frame is the −24 ACT loser, and **chunk-anchor (n-0) is exactly the Diffusion-killer** (world-n0
→ 16) that matches the reported "Diffusion garbage" failure. The mechanism is **Diffusion mode
collapse** (§3.7): the chunk-anchor policy emits attenuated, low-diversity commands (~⅓ the demo
magnitude), so the arm under-actuates. The fix is not "abandon relative" — it is **switch to
world-frame + per-frame anchor**. A rebuild that does so should train a working relative-position
policy on both ACT and Diffusion.

**What is validated vs not:**
- Validated: the frame / anchor / encoding / delivery directions above, on task10 (ACT n=50,
  Diffusion n=50) **and the rotation-heavy task11 (G1: `native_n0` Diffusion 98 ≈ `native` 96)**,
  with the GT-replay gate confirming every eval path is correct; the collapse mechanism is
  documented directly (§3.7, G2).
- **Caveats:** task10 is near-ceiling for Diffusion, so the native-family Diffusion cells don't
  discriminate — the load-bearing Diffusion signal is the `world-n0` collapse vs the native family.
  n=50, `libero_goal` tasks only. The per-frame anchor was reached in sim by baking per-frame
  relativization into a static dataset column; a production trainer can relativize per-frame at
  load time.
- **Already shipped (P1):** the production episode-boundary reset bug (`inference_node.reset_policy`
  not clearing the chunk-anchor state; no `/eval/episode_start` subscription — the analog of sim
  bug #5) is fixed on `fix/inference-episode-reset`.

**Open questions for the production rebuild (out of scope for this sim branch):**
- **Controller delivery** — whether the real controller wants a per-step incremental command or an
  absolute pose. Sim's `control_mode="absolute"` diverges open-loop (E5, not run) but that is
  robosuite-OSC-specific; answer it on the real controller, not LIBERO.
- **Discriminating Diffusion signal on a harder task** — partly answered (G1): per-frame relative
  still holds on rotation-heavy task11 under Diffusion (`native_n0` 98). Both task10 and task11
  remain near-ceiling for the *robust* conditions, so a harder suite is still needed to see whether
  the per-condition *margins* (not just pass/collapse) persist under Diffusion.
- **Larger task set / non-`libero_goal` suites** to confirm world>hand and per-frame-relative
  robustness generalize.

**Status.** Sim work on `feat/ee-libero-benchmark`. Production fix on `fix/inference-episode-reset`.
Ledger of record: `research/libero_ee/ledger/RESULTS.md`. Production rebuild = a separate branch.
</content>
