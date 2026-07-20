# Implementation Plan — `ee_delta` support in anvil-eval (+ ROS encoding threading)

Branch: `patrick/implement-ee-space` (worktree `.worktrees/implement-ee-space`)
Basis: refines and expands `2026-07-20-anvil-eval-gaps-fix-plan.md` — every finding in that
doc was re-verified against current source before this plan was written, and its scope is
extended to include the ROS `observation_encoding` threading fix (deferred there, included
here per explicit decision).

## Context

`ee_delta` is the per-frame **Delta(n→n+1)** EE action space: mcap_converter bakes
`action[t] = ee_delta_forward(pose[t+1], anchor=pose[t])` (world-frame SE(3)) into the
on-disk `action` column when `action_encoding: delta`. It was built to replace the broken
`ee_relative` (n-0 chunk-anchor) mechanism that caused high-frequency command jitter on
hardware (see `2026-07-17-libero-vs-production-diagnosis.md`).

Training, transforms, and ROS live-inference for `ee_delta` are **done and tested**
(`anvil_trainer`, `anvil_shared/ee_transform.py`). The **offline eval path never got
`ee_delta`** — `anvil_eval` and `anvil_eval_ros` predate it. Today an `ee_delta` checkpoint
run through `anvil-eval` silently falls into the generic non-EE metrics branch, producing
meaningless MAE that mixes metres + rot6d + gripper units, generic joint axis labels, and no
EE pass/fail summary. This plan closes that gap so `ee_delta` checkpoints get the same
correct EE evaluation as `ee_abs`/`ee_relative`.

Two enabling facts, both verified firsthand against current source:
- **Root cause of several gaps:** `anvil_config.json` written at checkpoint time
  (`anvil_trainer/patches.py:864-869`) persists `action_type`/`is_ee`/`is_ee_relative` but
  **not `observation_encoding`**. Confirmed on the existing smoke checkpoint
  `tests/smoke/outputs/model_zoo/ee_delta_rot6d/smoke/checkpoints/000010/pretrained_model/anvil_config.json`
  — it has no `observation_encoding` key, so any downstream consumer must assume `quaternion`
  even for a rot6d dataset (wrong 8n-vs-10n state layout). This is a latent bug for **any**
  non-quaternion `ee_abs`/`ee_relative`/`ee_delta` checkpoint, not just `ee_delta`.
- **`ee_delta`'s restore is structurally simpler than `ee_relative`'s.** `ee_relative` uses a
  chunk-anchor + shadow-queue (`evaluator.py:141-224`) because it trains one fixed anchor per
  generated chunk. `ee_delta` trains a fresh anchor **every frame**, and `evaluate_episode`
  already calls `select_action()` once per real dataset frame — the same cadence — so the
  correct offline restore is simply: restore every frame's action (and GT) against *that
  frame's own* observation. No new-chunk detection, no shadow queue.

## Scope

**In scope (this pass):**
1. `anvil_trainer/patches.py` — persist `observation_encoding` (1 field; unblocks correct
   restore for non-quaternion checkpoints).
2. `anvil_eval/evaluator.py` — recognise `ee_delta`, obs conversion, per-frame restore of
   both prediction and GT.
3. `anvil_eval/metrics.py` — add `ee_delta` to the EE-metrics gate (1 line + stale comment).
4. `anvil_eval/plotting.py` — add `ee_delta` to `_is_ee` (axis labels) and `show_delta`.
5. `anvil_eval_ros/cli.py` — add `ee_delta` to the sole `is_ee` gate (1 line).
6. **ROS encoding threading** (`ee_runtime.py` + `inference_node.py`) — thread
   `observation_encoding` through the restore wrappers so live inference is correct for
   rot6d/axis_angle checkpoints. *(Deferred in the 2026-07-20 gaps-fix doc; included here.)*
7. Tests + smoke step-3 re-train verification.

**Out of scope / deferred:**
- The GT-replay anchor-staleness bug (`ee_delta` failing fake-hardware replay, ~frame 37
  drift) — a live publish-loop timing bug owned by a concurrent agent; unrelated to offline
  eval (pure post-hoc per-frame math, no timing dimension).
- Running smoke **step 4** (`anvil-eval-ros`, Docker live ROS) for `ee_delta` — its code fix
  (#5) lands, but running it now would just rediscover the known GT-replay bug.
- `action_encoding: "relative"` (reserved, unimplemented).

## Changes

### 1. `anvil_trainer/patches.py` — persist observation_encoding
In `anvil_cfg_base` (`patches.py:864-869`), add
`"observation_encoding": self.config.observation_encoding,`. Verify `config.py` exposes
`observation_encoding` (it feeds `EEDeltaTransform`/`_compute_ee_delta_stats`); if the field
name differs, use the actual accessor.

### 2. `anvil_eval/evaluator.py` (the core change)
- `__init__`:
  - `evaluator.py:85` `self.is_ee` tuple → add `"ee_delta"`.
  - After `evaluator.py:87`: `self.is_ee_delta: bool = self.action_type == "ee_delta"`.
  - Add `self.observation_encoding: str = anvil_cfg.get("observation_encoding", "quaternion")`
    (default keeps old checkpoints working).
- `evaluate_episode` import (`evaluator.py:104`): add `ee_delta_inverse`.
- Obs conversion (`evaluator.py:169`): broaden the `ee_abs` branch guard to
  `elif (self.is_ee_abs or self.is_ee_delta) and "observation.state" in obs:`
  (ee_delta obs mirrors ee_abs — absolute, 8n/quat → 10n/rot6d, no relativization), and pass
  `observation_encoding=self.observation_encoding` into `ee_obs_abs_forward` (also fixes the
  latent non-quaternion `ee_abs` bug).
- **Leave `ee_relative`'s `_needs_restore`/`_is_new_chunk`/shadow-queue block
  (`evaluator.py:141-224`) untouched.**
- Add a small testable helper (module-level, near the top) and call it after the
  `ee_relative` restore block:
  ```python
  def _restore_ee_delta_action(delta, obs_flat, observation_encoding):
      """Restore one baked ee_delta value to absolute, anchored to that SAME
      frame's own observation — matches mcap_converter's baking convention
      (action[t] = ee_delta_forward(state[t+1], anchor=state[t])) and the ROS
      runtime's per-tick fresh-anchor design. No chunk bookkeeping."""
      return ee_delta_inverse(delta[np.newaxis, :], obs_flat,
                              observation_encoding=observation_encoding)[0]
  ```
  Prediction restore (after the existing `if _needs_restore:` block, before
  `evaluator.py:226` `predicted_actions.append(action)`):
  ```python
  if self.is_ee_delta and _obs_flat is not None:
      action = _restore_ee_delta_action(action, _obs_flat, self.observation_encoding)
  ```
- GT restore (`evaluator.py:227`, currently unconditional
  `ground_truth_actions.append(gt_action)`): the dataset's baked `action` column IS delta for
  `ee_delta`, so restore GT the same way before appending —
  ```python
  if self.is_ee_delta and _obs_flat is not None:
      ground_truth_actions.append(
          _restore_ee_delta_action(gt_action, _obs_flat, self.observation_encoding))
  else:
      ground_truth_actions.append(gt_action)
  ```
- `raw_gt_list` (`evaluator.py:201-204`): **no change** — `ee_delta` correctly hits the
  `else: raw_gt_list.append(gt_action)` branch (baked delta already IS raw model-output
  space), giving a meaningful `show_delta` panel.
- Update the stale comment at `evaluator.py:129` ("always absolute from dataset") to note it
  is baked-delta for `ee_delta`.

After #2, `cli.py:194` (`if evaluator.is_ee:`) auto-converts the restored absolute rot6d to
quat layout via `ee_rot6d_to_quat_layout` before metrics, and `cli.py:229` runs the EE
pass/fail summary — no `cli.py` change needed.

### 3. `anvil_eval/metrics.py`
- `metrics.py:198`: add `"ee_delta"` to `("ee_abs", "ee_relative", "ee_rel")`. The `% 8 == 0`
  guard is satisfied because `cli.py` pre-converts to 8n quat layout for all `is_ee` types.
- `metrics.py:61`: update the "populated only for ee_abs / ee_relative" comment.

### 4. `anvil_eval/plotting.py`
- `plotting.py:148` `_is_ee` (EE axis labels): add `"ee_delta"` — **required**.
- `plotting.py:48` and `:140` `show_delta`: add `"ee_delta"` — parity fix (delta-space
  diagnostic panel), low risk since `raw_output`/`raw_ground_truth` are populated for it.

### 5. `anvil_eval_ros/cli.py`
- The sole `is_ee = action_type in ("ee_abs", "ee_relative", "ee_rel")` gate (dims_per_arm,
  arm-info fallback, EE command-topic setup all cascade from it): add `"ee_delta"`.

### 6. ROS encoding threading (`ee_runtime.py` + `inference_node.py`)
Today both restore wrappers call the (now encoding-aware) inverse transforms **without**
`observation_encoding`, silently defaulting to quaternion — wrong for rot6d/axis_angle
checkpoints. Fix:
- `ee_runtime.py`: add `observation_encoding: str = "quaternion"` param to
  `ee_relative_restore_chunk` (`:109`, passes to `ee_relative_inverse` at `:150`) and
  `ee_delta_restore_step` (`:153`, passes to `ee_delta_inverse` at `:199`). Add a
  `resolve_observation_encoding(cfg)` helper mirroring `resolve_action_type` (`:60`):
  `return cfg.get("observation_encoding", "quaternion")`.
- `inference_node.py`: near `:285` (`self.action_type = resolve_action_type(meta)`) add
  `self.observation_encoding = resolve_observation_encoding(meta)`, and pass it into the three
  call sites (`:923`, `:929`, `:1032`).
- Default `"quaternion"` keeps every pre-fix checkpoint working unchanged.

## Verification

1. **Unit tests** (`uv run python -m pytest tests/unit -q` stays green):
   - New `tests/unit/anvil_eval/test_evaluator_ee_delta.py`: bake a known delta via
     `ee_delta_forward`, restore via `_restore_ee_delta_action`, assert exact round-trip for
     both `quaternion` and `rot6d` `observation_encoding`; assert `EpisodeEvaluator.__init__`
     sets `is_ee`/`is_ee_delta`/`observation_encoding` from a synthetic `anvil_cfg` (with the
     key present, and absent → default-quaternion fallback for old checkpoints).
   - `tests/unit/anvil_eval/test_metrics.py`: add `ee_delta` gate case mirroring `ee_abs`.
   - Add/extend a `ee_runtime` unit test asserting the wrappers pass `observation_encoding`
     through (e.g. rot6d obs restored differently than the quaternion default).
2. **Smoke step-3 re-train** (re-train so the fixed `patches.py` persists
   `observation_encoding`; the existing checkpoint predates the fix and lacks it):
   ```
   uv run python tests/smoke/scripts/pipeline_smoke_test.py \
       --scenario ee_delta,ee_delta_rot6d --select 1,2,3 --force
   ```
   Confirm: step 3 completes without error for both encodings; the re-trained
   `anvil_config.json` now contains `observation_encoding`; `metrics_summary.json` has
   populated `ee` metrics (position/orientation/gripper), not NaN/empty; plots use EE axis
   labels; restored errors are small & physically plausible (dataset self-consistency — a
   weaker bar than GT-replay, but confirms the restore math + wiring).
   - **FORCE_COLOR gotcha:** unset `FORCE_COLOR`/`COLORTERM` before asserting on Rich CLI
     output.
3. Narrow `pipeline_smoke_test.py`'s "steps 3-4 gated" docstring caveat (`:31`) to "step 4
   only" for `ee_delta`/`ee_delta_rot6d`.

## Risks & coordination
- **Concurrent-agent contention (finding #7 in the 2026-07-20 gaps-fix doc):** change #6
  edits `inference_node.py`, the file a concurrent agent is actively debugging for the
  GT-replay anchor bug. Before editing, re-check `git status`/recent edits to that file and
  coordinate; keep the #6 diff minimal (param threading only, no logic changes) to ease any
  merge.
- Changes #1–#5 touch files the concurrent agent does not (`anvil_trainer/patches.py`,
  `anvil_eval/*`, `anvil_eval_ros/cli.py`) — no conflict risk.
- Commit per the git-worktree rule; update
  `2026-07-19-training-flow-gaps-fix-plan.md`'s appendix to mark the anvil_eval gaps
  resolved, keeping the GT-replay bug listed as remaining deferred work.
