# Fix Plan — anvil-eval bugs/gaps for ee_delta (+ research findings)

Date: 2026-07-20
Branch: `patrick/implement-ee-space`

## Context

Training-side `ee_delta` support (per-frame Delta(n→n+1) EE action space, baked
by mcap_converter with `action_encoding: delta`) is complete and verified
(`anvil_trainer`, `anvil_shared/ee_transform.py`, both encoding-aware across
quaternion/rot6d/axis_angle — see `claude_docs/ee-delta/2026-07-19-training-flow-gaps-fix-plan.md`).
The **eval side never got the same treatment** — `anvil_eval` and
`anvil_eval_ros` predate `ee_delta` entirely. This plan closes that gap, based
on a fresh, line-verified research pass (2026-07-19/20) across `anvil_eval`,
`anvil_eval_ros`, ROS inference, and the concurrent GT-replay work.

## Research findings (what's missing / needs optimizing)

1. **Root cause enabling several gaps**: `anvil_config.json` (written at
   checkpoint time, `anvil_trainer/patches.py:864-873`) never persists
   `observation_encoding` — only `action_type`, `is_ee`, `is_ee_relative` +
   git provenance. Every downstream consumer (`anvil_eval/evaluator.py`, ROS
   `inference_node.py`/`ee_runtime.py`) therefore has no way to know a
   checkpoint's dataset used `rot6d`/`axis_angle` obs instead of the default
   `quaternion` — a latent bug for **any** non-quaternion `ee_abs`/`ee_relative`
   checkpoint too, not just `ee_delta`.
2. **`anvil_eval/evaluator.py`**: `is_ee` (line 85) excludes `ee_delta`; no
   restore path exists for it at all. Critically, `ee_delta`'s correct restore
   is **NOT** the same shape as `ee_relative`'s existing chunk-anchor +
   shadow-queue mechanism (lines 142-224) — that mechanism exists because
   `ee_relative` trains against a **single fixed anchor per generated chunk**.
   `ee_delta` trains against a **different anchor every real frame**
   (`action[t] = ee_delta_forward(state[t+1], anchor=state[t])`), matching
   exactly how the ROS runtime's `_publish_loop` already handles it: restore
   each queued delta against the **freshest observation at that same tick**
   (`ee_runtime.py`'s `ee_delta_restore_step`, called per-publish, no
   chunk-level anchor bookkeeping). Since `evaluate_episode` already calls
   `select_action()` once per real dataset frame — the exact same cadence —
   the correct offline analog is trivial: restore **every frame** against
   *that frame's own* `_obs_flat`, no new-chunk detection, no shadow queue.
   Confirmed via reading `evaluator.py` in full. Also: `gt_action` is
   commented "always absolute from dataset" (line 129) — true for
   `ee_relative`/`ee_abs` but **false** for `ee_delta` (its action column is
   baked delta) — the GT list needs the identical per-frame restore.
3. **`anvil_eval/metrics.py`**: `compute_episode_metrics`'s gate (line 198)
   excludes `ee_delta`. Verified (by reading `cli.py:190-198`) that this is
   purely mechanical to fix: `cli.py` already converts the rot6d-restored
   absolute actions to quaternion layout via `ee_rot6d_to_quat_layout` before
   calling metrics, gated on `evaluator.is_ee` — so once finding #2's fix
   lands, metrics.py needs only the one-line tuple addition, nothing deeper.
4. **`anvil_eval/plotting.py`**: `_is_ee` (line 148, drives EE axis labels)
   excludes `ee_delta` — required fix. `show_delta` (lines 48, 140, optional
   delta-space panel) excludes it too — cosmetic parity with `ee_relative`,
   worth adding since `evaluator.py` already populates `raw_output`/
   `raw_ground_truth` correctly for every action type (captured before
   restore, unconditionally).
5. **`anvil_eval_ros/cli.py`**: single `is_ee` gate (line 363) excludes
   `ee_delta`. Confirmed (exhaustive grep) this is the **only** action-type
   branch in the entire file — dims_per_arm, arm-info fallback, and EE
   command-topic setup all cascade from this one gate. One-line fix.
6. **ROS inference (`inference_node.py`, `ee_runtime.py`) — NOT a gap,
   already fully implemented**: `is_ee_delta` flag, obs handling (mirrors
   ee_abs), and a complete decoupled fresh-anchor publish loop
   (`ee_delta_restore_step`) all exist today. The only related gap: the
   `ee_runtime.py` wrapper functions (`ee_relative_restore_chunk`,
   `ee_delta_restore_step`) never pass `observation_encoding` through to the
   now-encoding-aware `anvil_shared.ee_transform` functions — they silently
   default to quaternion. **Deferred** (see Scope below).
7. **CRITICAL, ACTIVE, OUT OF SCOPE — flagging only**: the concurrent
   GT-replay correctness test (`tests/smoke/.gt_replay_reports/ee-delta/`)
   shows `ee_delta` **failing** replay on fake-hardware (554/592 frames fail,
   error grows from ~frame 37 — anchor-staleness/drift signature) while
   `ee_abs` passes cleanly. A concurrent agent has **live debug scaffolding**
   in `inference_node.py` (TEMP DEBUG / `[DEBUG-ANCHOR]` logging, timestamps
   from hours ago) actively investigating this exact bug right now. This is a
   **live publish-loop timing/anchor bug**, not a math bug —
   `ee_delta_forward`/`ee_delta_inverse` round-trip exactly (21 passing tests
   including bimanual + all 3 encodings, verified today). It does **not**
   block this plan: `anvil-eval`'s restore (finding #2) is a pure post-hoc
   per-frame computation over already-recorded dataset frames, with no live
   timing/staleness dimension at all — a structurally different, unaffected
   code path.
8. **No existing `ee_delta` checkpoint** anywhere in `model_zoo/` (checked:
   `ee_abs` ×5, `ee_rel` ×31, `ee_delta` ×0 at the time of this audit) — need
   a real artifact to verify against. (Since this audit, a real production
   `ee_delta` training run was kicked off against
   `data/datasets/ee-delta/pbib-standard-env-merged` — once it checkpoints,
   that's available as a real-world verification artifact too, in addition to
   the smoke-test-trained one this plan uses.)
9. **No test coverage for `evaluator.py` at all** today (only `metrics.py` has
   tests, 16 passing, zero `ee_delta`/`ee_relative` cases). Noted as a
   pre-existing gap; this plan adds targeted tests for the new logic, not a
   full test-suite backfill (disproportionate to this task).

## Scope

**In scope**: `anvil_trainer/patches.py` (1 field), `anvil_eval/evaluator.py`,
`anvil_eval/metrics.py`, `anvil_eval/plotting.py`, `anvil_eval_ros/cli.py` (1
line), new/extended tests, extending the `ee_delta`/`ee_delta_rot6d` smoke
scenarios to step 3 (`anvil-eval`), training one real `ee_delta` checkpoint to
verify against.

**Deferred, not touched this pass**:
- ROS `ee_runtime.py`/`inference_node.py` encoding-threading (finding #6) —
  small and safe in isolation, but touches the exact file/region a concurrent
  agent is actively debugging (finding #7). Revisit once that investigation
  concludes.
- The GT-replay anchor-staleness bug itself (finding #7) — owned by the
  concurrent agent.
- Smoke-test step 4 (`anvil-eval-ros`, Docker + live ROS inference) for
  `ee_delta` scenarios — the code fix (`cli.py:363`) is included, but actually
  *running* step 4 right now would exercise the exact composition path known
  to be broken/under-investigation (finding #7); a "pass" wouldn't validate
  anything meaningful yet and a "fail" would just rediscover the known bug.

## Changes

### 1. `anvil_trainer/patches.py` — persist `observation_encoding`
`anvil_cfg_base` (lines 864-869): add
`"observation_encoding": self.config.observation_encoding,` alongside the
existing `action_type`/`is_ee`/`is_ee_relative` keys.

### 2. `anvil_eval/evaluator.py`
- Line 85: `self.is_ee` → add `"ee_delta"`.
- After line 87: add `self.is_ee_delta: bool = self.action_type == "ee_delta"`.
- In `__init__`: add
  `self.observation_encoding: str = anvil_cfg.get("observation_encoding", "quaternion")`.
- Line 104 import: add `ee_delta_inverse`.
- Lines 169-176 (obs conversion): broaden
  `elif self.is_ee_abs and "observation.state" in obs:` →
  `elif (self.is_ee_abs or self.is_ee_delta) and "observation.state" in obs:`
  (ee_delta's obs mirrors ee_abs exactly — no relativization). Pass
  `observation_encoding=self.observation_encoding` to `ee_obs_abs_forward`
  (fixes a latent bug for non-quaternion `ee_abs` checkpoints too).
- **Do not touch** `_needs_restore`/`_is_new_chunk`/the shadow-queue block
  (lines 141-224) — leave `ee_relative`'s mechanism exactly as-is.
- Add a new, independent, per-frame restore block right after it (~line 225),
  extracted as a small helper for testability:
  ```python
  def _restore_ee_delta_action(delta, obs_flat, observation_encoding):
      """Restore a single-frame ee_delta value to absolute, anchored to that
      SAME frame's own observation — matches mcap_converter's baking
      convention (action[t] = ee_delta_forward(state[t+1], anchor=state[t]))
      and the ROS runtime's decoupled fresh-anchor design. No chunk-level
      bookkeeping needed, unlike ee_relative."""
      return ee_delta_inverse(delta[np.newaxis, :], obs_flat,
                               observation_encoding=observation_encoding)[0]

  # after the existing `if _needs_restore:` (ee_relative) block:
  if self.is_ee_delta and _obs_flat is not None:
      action = _restore_ee_delta_action(action, _obs_flat, self.observation_encoding)
  ```
- GT absolute list (line 227, currently unconditional
  `ground_truth_actions.append(gt_action)`): make conditional —
  ```python
  if self.is_ee_delta and _obs_flat is not None:
      ground_truth_actions.append(
          _restore_ee_delta_action(gt_action, _obs_flat, self.observation_encoding)
      )
  else:
      ground_truth_actions.append(gt_action)
  ```
- `raw_gt_list` (lines 200-204): **no change** — `ee_delta` already falls into
  the existing `else: raw_gt_list.append(gt_action)` branch correctly (the
  dataset's baked delta already IS "raw model-output space").

### 3. `anvil_eval/metrics.py`
Line 198: add `"ee_delta"` to the tuple. Update the stale comment at line 61.
No other change (confirmed the `% 8` dim check is satisfied because `cli.py`
pre-converts to quat layout before this function ever sees the data).

### 4. `anvil_eval/plotting.py`
- Line 148 (`_is_ee`, EE axis labels): add `"ee_delta"` — required.
- Lines 48, 140 (`show_delta`): add `"ee_delta"` — optional parity fix, low
  risk (evaluator.py already populates `raw_output`/`raw_ground_truth`
  correctly for it).

### 5. `anvil_eval_ros/cli.py`
Line 363: `is_ee = action_type in ("ee_abs", "ee_relative", "ee_rel")` → add
`"ee_delta"`. No other change needed (confirmed sole action-type branch).

## Verification

1. **Unit tests**:
   - `tests/unit/anvil_eval/test_metrics.py`: add `ee_delta` gate cases
     (mirroring the existing `ee_abs`/`joint_abs` tests).
   - New `tests/unit/anvil_eval/test_evaluator_ee_delta.py`: unit-test
     `_restore_ee_delta_action` directly — bake a known delta via
     `ee_delta_forward`, restore via the helper, assert exact round-trip;
     cover quaternion + rot6d `observation_encoding`; assert
     `EpisodeEvaluator.__init__` correctly sets `is_ee`/`is_ee_delta`/
     `observation_encoding` from a synthetic `anvil_cfg` dict (with and
     without the new key present, to check the default-quaternion fallback
     for old checkpoints predating this fix).
   - Full `uv run python -m pytest tests/unit -q` stays green.
2. **Real end-to-end**: train an actual `ee_delta` checkpoint via the
   existing smoke scenario (`--scenario ee_delta --select 1,2 --force`,
   already verified working), then extend to step 3:
   `--scenario ee_delta,ee_delta_rot6d --select 1,2,3 --force`. Confirm:
   - Step 3 (`anvil-eval`) completes without error for both the quaternion and
     rot6d smoke datasets.
   - `metrics_summary.json` contains populated `ee` metrics (position/
     orientation/gripper error), not NaN/empty.
   - Plots are generated with correct EE axis labels (not generic joint
     labels).
   - Sanity-check restored position/orientation errors are small and
     physically plausible (not NaN, not wildly large) — this is dataset
     self-consistency (predicting close to what was recorded), a weaker bar
     than the GT-replay correctness test, but confirms the restore math and
     wiring are correct.
3. Update `pipeline_smoke_test.py`'s module docstring once step 3 is
   confirmed working for `ee_delta`/`ee_delta_rot6d` (remove/narrow the
   "steps 3-4 gated" caveat to "step 4 only").

## Notes
- All work in worktree `.worktrees/implement-ee-space`
  (branch `patrick/implement-ee-space`); commit per the git-worktree rule.
- None of the in-scope files (`anvil_trainer/patches.py`, `anvil_eval/*`,
  `anvil_eval_ros/cli.py`) are touched by the concurrent GT-replay agent
  (confirmed via `git status` — they've only touched `inference_node.py`,
  `setup.py`, `fake_hardware_node.py`, `docker-compose.fake-hardware.yml`,
  plus new `ros2/.../dataset_gt_replayer_node.py` and friends) — no conflict
  risk.
- Update `claude_docs/ee-delta/2026-07-19-training-flow-gaps-fix-plan.md`'s appendix to
  mark the `anvil_eval`/`anvil_eval_ros` gaps resolved once this plan lands,
  and keep the GT-replay anchor-staleness bug + ROS encoding-threading gap
  listed there as the remaining deferred items.

## Addendum — real production run in progress (2026-07-20)

While this plan was being drafted, a real `ee_delta` production training run
was started in parallel against
`data/datasets/ee-delta/pbib-standard-env-merged` (bimanual, `rot6d`
observation encoding, 301 episodes / 176,913 frames, merged from
`pbib-standard-env-1st-try` + `pbib-standard-env-2nd-try`). Notes from getting
that run working, relevant to future commands/tooling in this repo:

- **`merge-datasets` (`packages/mcap_converter/src/mcap_converter/cli/merge_datasets.py`)
  silently drops `conversion_config.yaml`** — it's a thin wrapper around
  lerobot's own `merge_datasets`/`modify_features`, which only knows the
  LeRobot dataset schema and has no awareness of mcap_converter's config file.
  Both source datasets here had identical `conversion_config.yaml`
  (`action_encoding: delta`, `observation_encoding: rot6d`, same
  `code_commit`), so it was safe to manually copy one into the merged output
  directory as a stopgap. **Not yet fixed at the tool level** — worth a small
  follow-up: after merging, copy the file over if all inputs agree, warn/error
  if they disagree. Not yet scheduled to a specific plan.
- **`anvil-trainer` CLI flag syntax gotchas** (draccus-based), confirmed by
  trial/error against the live CLI:
  - List/tuple flags like `--policy.resize_shape` must be a quoted JSON-array
    string: `--policy.resize_shape='[270,480]'` — **not** space-separated
    (`--policy.resize_shape 270 480` fails with "unrecognized arguments").
    Documented in `docs/training.md:204`.
  - There is **no** `--image_transforms=<file>` flag path for overriding
    `dataset.image_transforms` in this CLI's `TrainPipelineConfig` — tested
    directly, raises `draccus.utils.DecodingError: The fields 'image_transforms'
    are not valid for TrainPipelineConfig`. The working way to set a *custom*
    `tfs` dict (beyond the documented simple
    `--dataset.image_transforms.enable/max_num_transforms` flags in
    `docs/training.md`) is `--dataset.image_transforms.tfs='<json>'` — a
    single JSON-object string covering all transform entries.
  - `--policy.resize_shape` is required to activate Diffusion's crop
    augmentation (`--policy.crop_ratio`/`--policy.crop_is_random` are silently
    ignored without it) — already documented in `docs/training.md:198`, worth
    remembering since it's a genuinely easy trap.
