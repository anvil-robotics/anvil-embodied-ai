# EE-Space Action Representation on LIBERO — Diary

_Companion to [`report.md`](report.md) (conclusions) — the chronological log, including the
wrong turns. `§` references point to `report.md`; `experiments/<name>/` refers to this topic's
`experiments/` dir (see [`README.md`](README.md))._

---


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
- `[exp]` **E-diff — native-family Diffusion coverage** (the recipe decider). Ran `native_abs`,
  `native_n0`, `native_hand` on Diffusion to complete the ACT+Diffusion matrix, motivated by the
  production question "can a re-encoded RELATIVE representation succeed on Diffusion, given
  goal-world-n0 collapsed to 16?".
- `[result]` All three = **98** on Diffusion (native/native_rot6d were already 100). Crucially
  **`native_n0` (per-frame relative) = 98, NOT collapsed** like the chunk-anchored goal-world-n0 (16).
- `[insight]` **Relative-position is not Diffusion-hard — chunk-anchor (n-0) re-encoding is.** rot6d
  is Diffusion-neutral, so the world-n0 collapse is the anchor, not the encoding. Diffusion also
  tolerates the representation choice far more than ACT (all native flips 98–100). → **Recipe (§8):
  world frame + per-frame anchor + relative delivery works on both architectures; production's
  `ee_rel` (body-frame + chunk-anchor n-0) picked the two worst choices, and chunk-anchor is exactly
  its Diffusion failure mode.**
- `[exp]` **G1 — recipe generalization.** Ran the per-frame-relative recipe (`native_n0`) on the
  rotation-heavy **task11** (rot_rms 3.1× task10) on both ACT + Diffusion, plus a `native`
  Diffusion reference. `[result]` `native_n0` Diffusion **98** ≈ `native` **96**; ACT 76 vs 80.
  gt-replay 61% (≈ baseline 60). `[insight]` the recipe is **not a task10 artifact** — per-frame
  relative survives Diffusion on the hardest-rotation task too.
- `[exp]` **G2 — collapse mechanism.** Two probes into *why* chunk-anchor breaks Diffusion but not
  ACT. `[result]` **Negative result:** reconstructing all three constructions off one `goalabs`
  trajectory shows the position target does NOT ramp within a horizon-16 chunk (dynamic range ≈1.0)
  — the "target grows large" intuition is **refuted**; magnitude is not the cause. `[insight]`
  **Load-bearing:** action-trace tap (`--trace-dir`) on the trained collapsed vs robust diffusion
  policies shows the collapse is **Diffusion mode collapse** — the chunk-anchor policy's delivered
  commands shrink to ~⅓ magnitude / ~½ per-axis spread of both the robust policy and the demos
  (attenuated marginal), with ~3× longer episodes; the robust policy reproduces the demo
  distribution. ACT survives by regressing the conditional mean. New analysis module
  `packages/anvil_sim/src/anvil_sim/studies/libero_ee/analysis/mechanism_analysis.py`; artifacts under `research/libero_ee/analysis/`.
