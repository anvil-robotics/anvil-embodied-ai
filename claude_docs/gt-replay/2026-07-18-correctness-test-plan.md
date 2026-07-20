# GT-replayer correctness test on fake hardware

## Context

`dataset_gt_replayer_node.py` (implemented earlier this session) replays a converted
dataset's recorded `action` rows through the real `inference_node.py` pipeline, injected
at the model-prediction seam, so the pipeline's own deque/restore/publish logic runs for
real against `ee_abs` and `ee_delta` datasets. It was manually verified once against
`docker-compose.fake-hardware.yml --profile replay` (both dataset encodings, clean
startup/shutdown, correct action-type detection, steady 30Hz publish, mock echo tracking).
That verification was ad hoc (eyeballing `ros2 topic echo`/`hz` output) and not repeatable.

This plan adds a real, rerunnable **integration test** that seeds the fake-hardware mock
from the dataset's own first recorded pose, then verifies that the full echoed
`/ee_pose_{arm}` trajectory reproduces the dataset's own recorded `observation.state`
trajectory — for both `ee_abs` (direct passthrough) and `ee_delta` (the decoupled
per-tick `obs ∘ delta` composition in `_publish_loop`). This is the strongest and most
direct version of "does replaying the converted dataset reproduce what was recorded,"
and — as a side effect of the 1-frame-lookahead identity (below) — it validates
mcap_converter's baked `action` column AND inference_node's restore/passthrough math in
a single assertion, not just the ROS-side restore in isolation.

## Revision log (this update, superseding the prior version of this plan)

1. **Seed the mock's initial pose from the dataset** (was: per-tick self-consistency
   only) — now full-trajectory comparison, see "Seeding" below.
2. **Drop `gripper_factor` scaling from the comparison** — compare the raw, unscaled
   gripper value on both sides. `gripper_factor` tunes live-inference feel; it's
   orthogonal to converter/pipeline correctness, which is what this test targets (same
   reasoning the now-superseded Dataset Action Playback plan used).
3. **Rename `gt_replay_dataset.py` → `dataset_reader.py`**, broadened scope, with an
   explicit, bounded consolidation decision (see "Scope decision" below).

## 1. Seeding — confirmed feasible for both encodings, one extra conversion for ee_delta

Research on `fake_hardware_node.py` confirmed:
- `self._ee_state[arm]` is a per-arm dict `{pos: xyz, quat: xyzw, gripper: float}`
  (`fake_hardware_node.py:195-202`), hardcoded default (`[0.4,0,0.5]`, identity quat,
  `0.02`), **no existing param seeds it**. `publish_ee_poses` (`:221-239`) publishes it
  as-is; `_ee_command_callback` (`:241-268`) overwrites it in-place — pure echo, ~10ms
  lag (`ee_pose_fps` default 100Hz).
- `observation.state` is **quaternion 8/arm** for the `ee_abs` fixture, **rot6d 10/arm**
  for the `ee_delta` fixture (confirmed against both `meta/info.json` files). The mock's
  state is quat layout, so ee_abs seeds directly; ee_delta needs one conversion step:
  `ee_rot6d_to_quat_layout` (`anvil_shared/ee_transform.py:493`, `(T,10n)→(T,8n)`).
- **1-frame-lookahead identity, confirmed algebraically**: `action[t] =
  ee_delta_forward(pose[t+1], anchor=pose[t])` at convert time, and
  `ee_delta_inverse`/`ee_delta_restore_step` is its exact algebraic inverse
  (`ee_transform.py:314-382`, `ee_runtime.py:147-194`). So if the mock's echoed obs at
  publish-tick `t` equals `observation.state[t]`, the published command at tick `t`
  equals `observation.state[t+1]` **exactly** — for `ee_delta` via the compose formula,
  and for `ee_abs` near-tautologically (mcap_converter defines `action[t]` as a rot6d
  re-encoding of `observation.state[t+1]` in the first place, and `_publish_ee_action`
  passes it straight through as rot6d→quat). **This unifies both encodings under one
  assertion**: `published_cmd[t] (quat) == dataset.observation.state[t+1] (converted to
  quat)`, for both `ee_abs` and `ee_delta`. No separate "self-consistency" formula needed.

### New mock param
Add `ee_seed_pose` (string ROS param, default `""`) to `fake_hardware_node.py`, declared
alongside the existing `ee_mode`/`ee_arms`/`ee_pose_fps` params
(`fake_hardware_node.py:61-67`). Format: comma-separated flat floats, `8 * n_arms` values
in `ee_arms` order (quat layout: `x,y,z,qx,qy,qz,qw,gripper` per arm) — matching
`_ee_state`'s own layout, zero conversion needed inside the mock. Parsed in
`_setup_ee_mode` (`:195-202`): if non-empty and correctly sized, use it per-arm instead of
the hardcoded default; if malformed, log a warning and fall back to the default (never
crash the mock over a bad seed string). Empty default preserves all existing behavior for
real (non-test) fake-hardware usage.

### Where the seed value comes from
`dataset_reader.py` (below) exposes `load_episode_observations_quat(dataset_root,
episode_idx) -> np.ndarray` — the **whole** episode's `observation.state` trajectory,
pre-converted to quat layout (passthrough if already quat, `ee_rot6d_to_quat_layout` if
rot6d). Row 0 is the seed; row `t+1` is the expected value for published command `t`. One
helper serves both needs. (axis_angle obs encoding is out of scope — neither fixture uses
it; the driver raises a clear error rather than mishandling an unsupported encoding if
ever pointed at one.)

## 2. Gripper comparison — no `gripper_factor`

The verifier compares the **raw** gripper value on both sides: expected = dataset's own
gripper (from `observation.state[t+1]`, no scaling), actual = the received
`CommandedEEPose.gripper` — but note `_publish_ee_action` in `inference_node.py` DOES
apply `gripper_factor`/clamp to the published value in real operation. To keep the
comparison meaningful without special-casing the production node, the driver launches
`replay` with an inference config whose `arms:` block sets `gripper_factor: 1.0` (and
wide-open `gripper_min/max`) for this test specifically — a small dedicated test config
(`configs/lerobot_control/inference_ee_gt_replay_test.yaml`, copied from
`inference_ee.yaml` with only the gripper knobs neutralized) rather than teaching the
verifier to reverse a config-driven transform. This keeps "what the verifier expects" and
"what the pipeline publishes" both raw, with no formula duplicated on either side.

## 3. `dataset_reader.py` — scope decision

### Audit findings (full detail — every hit, file:line)

**Tier A — hand-rolled parquet-glob per-episode readers (near-duplicates of the same
glob→filter→sort idiom):**
- `dataset_gt_replayer_node.py:168-193` (`_load_episode_actions`) — same package, single
  caller. **In scope, consolidate.**
- `packages/mcap_converter/src/mcap_converter/cli/validate.py:125-141`
  (`_load_episode_df`) — cross-package, pandas.
- `packages/mcap_converter/src/mcap_converter/utils/debug_plot.py:47-72` — cross-package,
  **pyarrow** (not pandas), slightly different glob (`rglob`).
- `packages/anvil_eval/src/anvil_eval/gt_replay.py` (deleted; recovered via `git show
  HEAD:...`) — the original source of this pattern; nothing left to consolidate, it's gone.

**Tier B — encoding/action_type detection (`conversion_config.yaml`):**
- `dataset_gt_replayer_node.py:126-151` (`_resolve_action_type`) — same package, **raw
  `yaml.safe_load`** by deliberate design (documented in the node: the inference Docker
  image does not ship `mcap_converter`, so it cannot import `ConfigLoader`). **In scope,
  consolidate — but keep the raw-yaml approach, do NOT switch to `ConfigLoader`.**
- `packages/anvil_eval/src/anvil_eval/gt_replay.py` (deleted) — used
  `mcap_converter.config.loader.ConfigLoader(strict=False)` (schema-migration-aware,
  handles the v1.0 `ee_action_encoding` rename). This *looks* strictly better in the
  abstract, but adopting it here would reintroduce the exact dependency the replayer
  deliberately avoided for Docker-image-size/no-mcap_converter-in-container reasons.
  **Correcting my earlier framing: this is not a "which is better" question, it's a hard
  constraint — raw yaml stays.**
- `packages/anvil_eval_ros/src/anvil_eval_ros/cli.py` (3 separate raw-yaml re-reads,
  `:143-158,192-217,291-333`) — cross-package, different path-resolution assumptions
  (operates one level up, on the raw-MCAP-adjacent layout, not a plain dataset root).

**Tier C — `meta/info.json` metadata helpers** (`anvil_eval/dataset.py`'s
`_load_joint_names`, `mcap_converter`'s `validate.py`/`dataset_viz.py`,
`anvil_trainer/config.py`'s feature-name reads) — small individually, spread across all
four packages, each extracts different fields. Low payoff to unify.

**Tier D — `LeRobotDataset`-backed readers** (`anvil_eval/evaluator.py`,
`mcap_converter/cli/merge_datasets.py`, `cli/upload.py`, `cli/dataset_viz.py`,
`cli/validate.py`'s `test_dataset`) — these lean on lerobot's own class (video decoding,
transforms, `hf_dataset` indexing). **Should NOT be folded into a hand-rolled reader** —
that would be a capability regression, not a consolidation.

### Decision for this pass (confirmed with user)
**In scope (needed by both nodes anyway, zero added risk, same package):**
`dataset_gt_replayer_node.py`'s `_load_episode_actions` (Tier A) and `_resolve_action_type`
(Tier B) fold into `dataset_reader.py`. This was already the plan's original scope: the
broadening is additive, not a change to what gets touched.

**Deferred, not folded in this pass (cross-package, separately risky per item, per the
existing project convention of scoping consolidations into their own passes — e.g. the
mcap_converter encoding-cleanup plan's explicit deferral of `is_ee`/`is_ee_relative`
consolidation across packages):**
- `validate.py`'s `_load_episode_df` and `debug_plot.py` (Tier A cross-package folds —
  `mcap_converter` would gain a dependency on the `ros2` package, an awkward direction
  since `mcap_converter` is the upstream producer).
- `anvil_eval_ros/cli.py`'s three yaml re-reads (Tier B, different path-resolution
  assumptions, 3 call sites).
- All of Tier C (small payoff, 4-package spread) and Tier D (would regress capability).

A memory note recording these deferred candidates (mirroring
`project_mcap_converter_config_deferred_work.md`'s existing pattern) will be written once
this plan is approved.

### `dataset_reader.py` final shape
```
load_info(dataset_root) -> dict                                   # meta/info.json, raw
resolve_action_type(dataset_root, logger=None) -> str              # raw yaml, NOT ConfigLoader
load_episode_columns(dataset_root, episode_idx, columns) -> pd.DataFrame   # generic Tier-A primitive
load_episode_actions(dataset_root, episode_idx) -> np.ndarray       # thin wrapper (existing shape)
load_episode_observations_quat(dataset_root, episode_idx) -> np.ndarray   # NEW: whole-episode obs, quat layout
```
No image/video reading: parquet doesn't contain image bytes (those live in `videos/*.mp4`
and need `LeRobotDataset`/video decode — a materially different kind of reader than the
parquet-glob pattern this module is built around, and Tier D above is explicitly staying
on `LeRobotDataset`). Neither `dataset_gt_replayer_node.py` nor the new verifier touches
images, so this is not a gap for the in-scope work — flagged as an explicit non-goal
rather than silently expanded into video-decoding work.

## New files

1. **`ros2/src/lerobot_control/lerobot_control/dataset_reader.py`** — shape above.
2. **`ros2/src/lerobot_control/lerobot_control/gt_replay_verifier_node.py`** — live
   comparator, modeled on `eval_recorder_node.py`. ROS2 params: `dataset`, `episode`
   (default 0), `atol_pos_m` (default `1e-4`), `atol_rot_deg` (default `0.5`),
   `report_path` (default `/workspace/reports/gt_replay_report.json`), `timeout_sec`
   (default `60.0`). No `config_file` param needed anymore — gripper scaling is
   neutralized at the source config (§2), and topic names follow the same
   `/ee_pose_{arm}`/`/commanded_ee_{arm}` convention the replayer already assumes.

   Behavior: load `obs_quat = load_episode_observations_quat(...)`,
   `action_type = resolve_action_type(...)`. Subscribe per arm to `/ee_pose_{arm}` (only
   to detect the very first message = the seed landing, for a startup sanity check — not
   needed for the per-command comparison anymore, since expected values come purely from
   `obs_quat`, no live obs pairing required) and `/commanded_ee_{arm}` (on the Nth message:
   expected = `obs_quat[N+1]` sliced per arm, compare pos/rot/gripper — raw, no
   gripper_factor — against tolerances). On reaching `len(obs_quat)-1` messages per arm or
   `timeout_sec`, write the JSON report and shut down. Simpler than the original design:
   no live obs/command pairing race to reason about, since the expected trajectory is
   fully known upfront from the dataset.

3. **`configs/lerobot_control/inference_ee_gt_replay_test.yaml`** — copy of
   `inference_ee.yaml` with `gripper_factor: 1.0`, wide `gripper_min/max` per arm.
4. **`tests/smoke/scripts/gt_replay_correctness_test.py`** — driver, following
   `pipeline_smoke_test.py`'s docker-orchestration conventions. For each fixture dataset
   (`data/debug/ee-abs/ee-space-testing`, `data/debug/ee-delta/ee-space-testing`):
   - Compute the seed string (`load_episode_observations_quat(...)[0]`, formatted as
     comma-floats) in plain Python (no docker yet).
   - `docker compose --profile replay-verify up -d mock-robot` with `EE_SEED_POSE=<seed>`
     env wired to the new param; wait healthy.
   - `docker compose --profile replay-verify up -d gt-replay-verify`; sleep ~3s for DDS
     discovery headroom.
   - `docker compose --profile replay-verify up -d replay` using
     `inference_ee_gt_replay_test.yaml`.
   - Poll the mounted report file, assert `all_passed`; dump docker logs on failure/timeout.
   - `docker compose --profile replay-verify down` in `finally`.
   Exit code reflects aggregate pass/fail across both datasets.

## Edited files
- `ros2/src/lerobot_control/lerobot_control/test/fake_hardware/fake_hardware_node.py` —
  add `ee_seed_pose` param (§1).
- `dataset_gt_replayer_node.py` — delegate `_load_episode_actions`/`_resolve_action_type`
  to `dataset_reader.py` (behavior-preserving).
- `ros2/src/lerobot_control/setup.py` — add `gt_replay_verifier_node` console_scripts entry.
- `docker-compose.fake-hardware.yml` — new `gt-replay-verify` service (profile
  `replay-verify`); `mock-robot`'s command gains `-p ee_seed_pose:=${EE_SEED_POSE:-}`;
  dataset + reports-dir mounts.

## Verification

1. Run the driver against both fixture datasets; confirm `all_passed: true` for both.
2. Deliberately break the compose logic once (e.g. flip a sign in `ee_delta_restore_step`,
   or swap left/right slices), rerun, confirm the test **fails** with a clear non-zero
   max error — proving it catches regressions. Revert immediately after.
3. `uv run pytest tests/unit/` — confirm the `dataset_reader.py` extraction didn't change
   `dataset_gt_replayer_node.py`'s existing behavior.
