# LIBERO validated EE-space vs. production `ee_rel`: implementation diagnosis

**Scope.** Diagnosis only, no fixes. Compares `patrick/sim-valid-dev` (LIBERO, 3
confirmed-successful conditions) against `patrick/implement-ee-space` (production,
stalled on real hardware). Every finding below is bucketed:

- **(a)** confirmed difference with a plausible causal link to the observed failure
- **(b)** real difference, unlikely to be causal
- **(c)** not fairly comparable — LIBERO's validated conditions never faced this problem

**Checkpoint under test.** `monitor_output/` was produced by the only ee_rel checkpoint
in this worktree for this task —
`model_zoo/ee-space/.../diffusion_20260702_145619/checkpoints/070000` — confirmed via
its `config.json`: `horizon=16, n_action_steps=8, n_obs_steps=2, noise_scheduler_type=DDIM,
num_train_timesteps=50, normalization_mapping={ACTION: MIN_MAX, STATE: MIN_MAX}`. All
numbers below (stats, magnitudes) are cross-checked against this exact checkpoint's own
saved normalizer, not just the raw dataset.

## Failure signature (read first, per the framing)

`monitor_output/inference_data.csv` + `inference_report.png`, 546 steps, 20 DOF (2 arms
× [xyz + rot6d(6) + gripper]). **Not** gross divergence, random motion, or drift: obs
(blue) and control_cmd (green) track the same gross trajectory throughout, and 3 sampled
wrist-camera frames across the 66 s clip show no collision or catastrophic event.

The actual defect, quantified (first-difference std of `control_cmd` vs `obs_state` per
DOF, `n=545` steps):

| DOF | obs Δstd | cmd Δstd | ratio |
|---|---|---|---|
| R_z | 0.00144 | 0.00532 | **3.70×** |
| R_r5 | 0.00497 | 0.01258 | **2.53×** |
| R_y | 0.00242 | 0.00530 | 2.19× |
| R_r0 | 0.00961 | 0.02067 | 2.15× |
| R_r1 | 0.00487 | 0.01073 | 2.20× |
| R_r2 | 0.00525 | 0.01165 | 2.22× |
| L_z | 0.00181 | 0.00405 | 2.24× |
| L_grip | 0.00301 | 0.00386 | 1.28× |
| R_grip | 0.00143 | 0.00140 | 0.98× |

Every non-gripper DOF runs **1.5–3.7× jitter over obs**; both grippers ≈1.0× (no excess).
Per-step commanded magnitude: position ~0.003–0.006 m, rot6d ~0.01–0.024. **The failure
signature to explain is systematic high-frequency command jitter concentrated in
position-z and rotation dims, not divergence.** (CSV caveat: `raw_output`/`control_cmd`
are the already-restored *absolute* rot6d command, not the model's raw normalized/relative
output — see Part 3 — so this measures the final commanded signal, which is what matters
for the real arm, but it can't show the pre-restore signal directly.)

---

## Part 1 — Dataset and data processing (primary)

### 1.1 The n-0 mechanism, confirmed exactly as suspected

On-disk `action` is `[20]` absolute rot6d (2 arms × [xyz, rot6d(6), gripper]); on-disk
`observation.state` is `[16]` absolute quaternion. `EERelTransform.apply`
(`packages/anvil_trainer/src/anvil_trainer/transforms.py:302-333`) relativizes the
**entire prediction horizon** to a **single anchor**, `obs_full[-1]`:

```python
anchor_tensor = obs_full[-1]                        # ONE anchor for the whole chunk
delta_np = ee_rel_forward(action_np, anchor_np)     # all L rows vs same anchor
```

`ee_rel_forward` (`packages/anvil_shared/src/anvil_shared/ee_transform.py:111-138`)
broadcasts that one `R_state`/`state_xyz` over every row when the anchor is 1-D. **L =
the full horizon (16 for this checkpoint), not the executed chunk** (`n_action_steps=8`).
The anchor is fixed within a chunk, re-derived per training sample — this is n-0 exactly.

`_compute_ee_rel_stats` (`packages/anvil_trainer/src/anvil_trainer/patches.py:337-359`)
computes normalization stats by pooling `ee_rel_forward` outputs **across all offsets k
in `action_delta_indices`** (i.e., across the whole horizon), not per-position. The rot6d
"identity trick" (`_force_rot6d_identity`, patches.py:60-77) then forces rot6d dims'
min/max to ±1.

This is the direct structural analog of LIBERO's own **`goal-world-n0`** condition
(chunk-start n-0 re-encoding), which is documented as showing **severe Diffusion
collapse: 16% vs 98% success for the per-frame equivalent** — "a real
representation×architecture interaction, not a bug" (`report.md:224-226,45-46`), with
collapsed-policy commands "shrink to ~⅓ magnitude / ~½ spread of the demonstrations."
Production combines the same n-0 chunk representation with a Diffusion architecture
(DDIM here, DDPM in other runs). **This is the strongest causal candidate.**
→ **(a)**

`native_n0` (LIBERO) is a different, invalid case — "structurally locked to
`per_frame_anchor=True`," never a genuine chunk-anchor test, degenerating to the Delta
command itself (`stage1-closeout.md:46-51`). Not a counterexample to the collapse
finding; it just never tested the mechanism at all.

### 1.2 Magnitude-vs-k — quantified, and matched against the deployed checkpoint's own stats

Computed on the real converted dataset (198 episodes, 60,616 frames,
`data/datasets/ee-space/pick-and-place-hand-switch-cat-plushie-open-env-s3`):

| k | pos median (m) | rot median (rad) |
|---|---|---|
| 1 (= LIBERO-style single-frame delta) | 0.00007 | 0.00038 |
| 8 (= n_action_steps, last *executed* step) | 0.01283 | 0.05149 |
| 16 (= full horizon) | 0.02887 | 0.10735 |

Growth is monotonic and large: **median magnitude at k=16 is 421× (position) / 281×
(rotation) the k=1 single-frame value.** k=1 n-0 targets are bit-identical to the
LIBERO-style single-frame delta, as expected. Pooling all k=1..16 together (what
`_compute_ee_rel_stats` does) inflates the effective [p1,p99] normalization range **7.1×
(position) / 6.8× (rotation)** relative to using single-frame deltas alone. This
growing-with-k pattern is **structurally absent** from all three validated LIBERO
conditions (single-step only, no k>1 exists).

**Confirmed against the actual deployed checkpoint's saved normalizer**
(`policy_preprocessor_step_3_normalizer_processor.safetensors`, `action.min`/`action.max`):
position-dim ranges are ~0.11–0.28 m (e.g. left-y: [-0.174, 0.255]) — squarely in the
range set by k≈8–16 displacements, not the k=1 (~0.0001 m) scale that dominates what's
*actually executed* before the next replan. Rot6d dims sit at exactly ±1.0 (identity
trick applied correctly for this checkpoint's MIN_MAX/MIN_MAX config — the v1-run
`STATE=MEAN_STD` inconsistency noted during exploration does **not** apply to the
checkpoint that produced this failure).

**Causal mechanism, position dims (a):** under MIN_MAX normalization
(`2*(x-min)/(max-min)-1`), a k=1-scale true target occupies a tiny fraction of the
normalized [-1,1] axis (range ~0.46 m built almost entirely from rare large-k
displacements). Any residual absolute noise in normalized output space — inherent
diffusion sampling/denoising stochasticity — gets denormalized by multiplying back by
the (wide) range, producing absolute command noise that is large *relative to the true
near-term signal*, without needing any bug in the transform math itself (confirmed exact
in Part 4). This plausibly explains the observed position-dim jitter (L_z 2.2×, R_y
2.2×, R_z 3.7×).

**Rot6d dims (a, weaker/more speculative):** the identity trick means rot6d components
are *not* range-inflated the way position is (rot6d components are bounded to [-1,1]
regardless of k, unlike position deltas which scale linearly with k). So the pooling
mechanism above does not directly explain rotation jitter the same way. A more likely
contributor for rotation specifically: the network must represent both near-identity
(k=1) and larger-angle (k=16) targets with the same 6 numbers and the same learned noise
floor; a fixed absolute noise level in rot6d-component space, reconstructed through
Gram-Schmidt back onto SO(3), can translate into disproportionate angular jitter when
the true target angle is small. This is plausible and consistent with rotation dims
showing the *worst* ratios in the failure signature (R_z 3.7×, R_r5 2.5×), but was not
independently quantified in this pass (flagged as a good follow-up: measure Gram-Schmidt
noise amplification directly). → **(a), medium confidence**

### 1.3 Quaternion/rotation-instability check — real data, θ≈π premise corrected

The original framing asked to "echo" a LIBERO axis-angle θ≈π finding. **That LIBERO
finding does not exist as stated** — the θ≈π singularity hypothesis for `afo_abs_h1`'s
8.2% GT-replay failure was *proposed and then disproved*: `orientation_error()` depends
only on the relative angle, and a direct round-trip test showed axis-angle math exact to
1e-16 near θ=π (`diary.md:130-132`, `stage1-closeout.md:59-63`). So there is nothing
validated to compare production against here — this is necessarily a **fresh** check,
not corroboration of a precedent.

Computed on real OpenArm data (`observation.state` quaternions, 60,616 frames, both
arms): **51–52% of all frames sit within 0.1 rad of π** rotation-from-identity (the
gripper-down operating pose is essentially a ~180° rotation from the identity
convention), and `qw` is split almost evenly between positive and negative (42%/44% left
arm) with **13.6% of frames at |qw| ≤ 0.01** — squarely on the quaternion double-cover
boundary where `q` and `-q` represent the same rotation and small state noise flips sign.

**This is real, but very likely not causal (→ b):** unlike axis-angle, `R(q) = R(-q)` is
an exact mathematical identity — the sign ambiguity is fully absorbed the moment the
quaternion is converted to a rotation matrix (`quats_to_matrices`) en route to rot6d.
Both the `ee_rel` transform's anchor rotation and the model's rot6d input/output go
through this matrix stage, so the double-cover sign flip in the raw on-disk quaternion
column does not propagate into the rot6d representation actually trained on or
predicted. Flagging this distinction explicitly: production's rotation encoding
(quaternion→matrix→rot6d) is structurally immune to exactly the kind of representation
discontinuity that made axis-angle a suspect on the LIBERO side (and that suspicion was
itself disproved there too).

### 1.4 rot6d identity-trick comparison

Production's trick (patches.py:60-77) forces rot6d min/max to ±1 in the **training-time
relative** stats so MIN_MAX normalization passes rot6d through unchanged. **Empirically
confirmed applied correctly** in the deployed checkpoint's own saved stats (§1.2). The
dataset-level `meta/stats.json` in the raw converted dataset is a red herring for this
comparison: it reflects the on-disk **absolute** action distribution, and its rot6d
min/max landing near [-1,1] there is a coincidence of rot6d being unit-vector components
(bounded to [-1,1] for any rotation, absolute or relative) — not evidence of the trick
itself, which only exists in a trained checkpoint's stats artifact.

LIBERO's three validated conditions have **no equivalent trick at all** — they train via
plain `lerobot-train` on LIBERO's native 8-dim axis-angle state / 7-dim delta action, no
`anvil_trainer` transform involved (`libero_convert.py:24-31`). `native_rot6d` succeeds
without any rot6d-specific stat patching. → **(c)**: not comparable (different code
path entirely), and the trick is confirmed working as designed where it is used, so not
independently a suspect. → **(b)** on its own merits.

---

## Part 2 — Model training

| | LIBERO validated (Diffusion arch, plain `lerobot-train`, unmodified defaults) | Production (deployed checkpoint) |
|---|---|---|
| horizon / n_action_steps | 16 / 8 | 16 / 8 (**identical ratio**) |
| n_obs_steps | 2 | 2 |
| normalization (ACTION / STATE) | MIN_MAX / MIN_MAX | MIN_MAX / MIN_MAX |
| scheduler / timesteps | DDPM / 100 | DDIM / 50 |
| EMA | not implemented in this lerobot fork (no such field anywhere) | anvil_trainer default is ON; **undetermined for this specific checkpoint** — no EMA shadow weights found in `optimizer_state.safetensors`, no `ema`/`ddpm_ip` keys in `anvil_config.json`, `train_config.json`, or the wandb run's logged config/args (this run is a `--resume` invocation; original launch flags not recovered) |
| DDPM-IP | not implemented in this lerobot fork | same as above — undetermined for this checkpoint |

**Key correction to the original framing:** the horizon/n_action_steps *ratio* is not
the differentiator — LIBERO's own Diffusion default is horizon=16, n_action_steps=8,
matching production exactly. The difference is entirely **structural**: LIBERO's 16
horizon positions are each an independent per-frame delta (n-(n-1)); production's are
all relative to one fixed chunk-start anchor (n-0). Same "shape," different target
semantics — this is what makes §1.1/1.2 the load-bearing finding, not a config mismatch.

**Additional unvalidated factors (flag as such, not asserted bugs):** EMA and DDPM-IP are
literally not features of the lerobot fork LIBERO's validated conditions ran on — they
were never tested there in any form, on or off. Whether they were active for *this*
production checkpoint could not be determined from any artifact inspected (checkpoint
configs, optimizer state, wandb logs). Since EMA's whole purpose is output smoothing, an
EMA that is *not* actually active (despite defaulting on) is a cheap, high-value thing to
verify directly against the training launch script/shell history — its absence would be
directly consistent with the observed jitter. → **(b)/open**, recommend verifying before
ranking further.

---

## Part 3 — Model inference

### Anchor handling — the original "mismatch" hypothesis is negated

`_delta_ref_state` (`inference_node.py:767-770`) is set once per new chunk to
`_ee_raw_obs_buf[-1]` and held fixed for the whole chunk — **the identical n-0 scheme as
training** (`transforms.py:306-309`, `obs_full[-1]`). Train and inference anchors agree.
There is no anchor-assignment bug to find here.

### → (c): production is solving a problem LIBERO's validated conditions never faced

LIBERO's three validated conditions use a **pure single-step "direct" eval/replay path**
(`replay_adapter.py:50-56`) — `obs_step_for_anchor=None`, no chunk-anchor object is ever
created, because their targets are always per-frame. The only LIBERO family that *did*
have to solve chunk-anchor reuse is the goal n-0 family — which is exactly the family
that collapsed (§1.1) or was invalid. So "does production's chunk-anchor handling match
LIBERO's" is not a fair comparison; LIBERO's validated conditions structurally sidestep
the entire question by never using a chunk anchor at all.

### Residual real asymmetry (b): obs-window staleness

Production's obs sampling and command publishing run on **separate timers/deques** —
`_obs_update` samples on a control-frequency timer; `_publish_loop` drains a separate
`_classic_action_deque` (maxlen 10). The anchor pose that relativized a chunk is
therefore generally staler by the time later steps in that chunk are actually published
than it was at chunk-generation time. This is a real train/inference asymmetry (training
has no such publish-lag), but it degrades gracefully with lag rather than explaining a
jitter signature — more likely a secondary contributor than the primary cause. → **(b)**

### CSV/monitor caveat

`raw_output` in `inference_data.csv` is misleadingly named — it is the **already-restored
absolute** rot6d command, not the model's raw normalized/relative output
(`inference_node.py:791-804`, restore happens before this value is captured). `delta_cmd`
is computed by the *monitor*, not the inference node, as a naive `cmd − obs` subtraction
— meaningless for rot6d dims (element-wise difference of two rotation encodings is not a
rotation). **None of the four CSV columns exposes the pre-restore normalized/relative
signal**, so this recording cannot directly show whether the jitter originates in
normalized model-output space (consistent with §1.2's SNR argument) or is introduced by
the restore step (already ruled out as a math bug in Part 4). A future monitor revision
that logs the raw pre-restore relative output would directly test the §1.2 hypothesis.

---

## Part 4 — Robot controller (light touch) + minimal GT-replay proposal

**Basic end-to-end sanity (light touch only, as scoped):** the failing run completed all
546 steps without crash or collision; 3 sampled wrist-camera frames across the 66 s clip
show ordinary workspace scenes, consistent with "degraded quality," not "pipeline
broken." LIBERO's validated conditions are independently confirmed running end-to-end via
their own GT-replay/eval harness (98–100% success). No further controller-level
investigation was done — correctly out of scope per the framing.

**Missing piece, addressed: minimal GT-replay for production.** Production has no
GT-replay capability isolating the transform pipeline from the trained model. Built and
ran a minimal diagnostic (not a port of LIBERO's gating infra): for 3 real episodes, take
ground-truth absolute EE poses → `ee_rel_forward` (both single-frame-anchor and n-0
chunk-anchor styles) → `ee_rel_inverse`, no model in the loop:

| episode | anchor style | max pos err (m) | max rot err (deg) |
|---|---|---|---|
| 0 | single-frame | 2.8e-17 | 1.7e-06 |
| 0 | n0-chunk (L=16) | 1.9e-16 | 1.7e-06 |
| 1 | single-frame | 2.8e-17 | 1.7e-06 |
| 1 | n0-chunk (L=16) | 1.4e-16 | 1.7e-06 |
| 2 | single-frame | 2.8e-17 | 1.7e-06 |
| 2 | n0-chunk (L=16) | 1.8e-16 | 1.7e-06 |

**Round trip recovers ground truth to machine precision for both anchor styles.** The
`ee_rel_forward`/`ee_rel_inverse` transform math is exact — **this rules out a geometry/
transform bug as the cause of the real-hardware failure**; whatever is happening lives in
the statistical/normalization treatment (§1.2) and/or the trained model's behavior under
it, not in the deterministic math.

Prototype scripts (read-only, not committed):
`/tmp/claude-1002/-workspace-patrick-anvil-embodied-ai/a1bbe758-aa79-4a6f-a33d-c2854ffb17db/scratchpad/ee_diag/{task1_magnitude_vs_k.py,task2_quat_distribution.py,task4_roundtrip.py}`.
Formalizing this into a permanent repo tool (e.g. `scripts/gt_replay_ee_rel.py` that any
future ee_rel checkpoint/dataset pair can be pointed at) is recommended as the concrete
next step, out of scope for this diagnosis pass.

---

## Summary ranking

1. **(a), highest confidence:** n-0 chunk-wise relativization + stats pooled across the
   full horizon → position normalization range set by rare large-displacement (k≈16)
   targets, compressing the near-term (k=1–8, actually executed) signal into a tiny
   fraction of the normalized dynamic range. Directly analogous to LIBERO's own
   documented `goal-world-n0` Diffusion collapse (98%→16%). Confirmed against the actual
   deployed checkpoint's saved normalizer stats, not just the raw dataset.
2. **(a), medium confidence:** rotation-dim jitter (the worst-hit DOFs in the failure
   signature) plausibly from the same heterogeneous-target-scale problem via a different
   channel (Gram-Schmidt reconstruction sensitivity), not yet independently quantified.
3. **(b), worth a cheap direct check:** whether EMA was actually active for the deployed
   checkpoint — no artifact inspected confirms it, and its absence would directly predict
   exactly this kind of excess jitter.
4. **(b), unlikely causal:** quaternion double-cover proximity to π — real, but
   structurally absorbed by the quat→matrix→rot6d conversion before it reaches the model.
5. **(c), not comparable:** production's chunk-anchor inference handling — LIBERO's
   validated conditions never used a chunk anchor at all; train/inference anchors agree
   in production, so no mismatch bug exists here.
6. **(b), minor:** obs-window staleness from async publish deque; residual asymmetry, but
   degrades gracefully rather than explaining a jitter signature.
7. **Ruled out:** the `ee_rel_forward`/`ee_rel_inverse` transform math itself (exact to
   1e-16 round-trip); train/inference anchor mismatch (anchors agree); v1's
   STATE=MEAN_STD inconsistency (this checkpoint uses MIN_MAX/MIN_MAX); the θ≈π
   axis-angle instability (disproved on LIBERO's own side, and inapplicable to rot6d
   regardless).
