# Plan — GT replay by injecting recorded actions at inference_node's model seam

**Status: design only, not yet implemented.**

This supersedes an earlier draft (`dataset-action-playback-plan.md`, removed) that proposed a
standalone node doing its own restoration + publishing. That approach was wrong for the goal —
it would have exercised none of the real inference pipeline.

## Context

**Goal:** replay a converted dataset's recorded ground-truth `action` values *as if they were
the model's own output*, injected at the exact point where model predictions normally appear,
so the entire rest of the real `inference_node.py` architecture runs **completely
unmodified** — observation reading, the classic action deque, the Item-2 decoupled delta-mode
publish loop (per-tick fresh-observation anchoring / `obs ∘ delta` composition), absolute
restoration, message building, and publishing. This validates that `mcap_converter`'s output
and the real inference pipeline's *consumption* of it are mutually consistent end-to-end — not
just that the stored numbers are internally consistent (which is all the old
`anvil-gt-replay` math tool checked).

The replayer's ONLY job: read the next row of recorded GT `action` (in whatever encoding it's
stored — `absolute`/`delta`, **without restoring or converting it**) and hand it to the
pipeline at the seam where the model's post-processed (physical-units) prediction normally
enters the deque. All restoration (e.g. `obs ∘ delta` for delta datasets) is done by the
existing publish loop, unchanged. If replaying the converted data can't reproduce the task,
that signals the converter (or an `action_encoding` choice) lost information the pipeline
needs — caught before any training compute.

Also: **delete `gt_replay.py` / `anvil-gt-replay` entirely** — the old math-only tool is not
needed going forward.

Grounded in a full read of `inference_node.py` (`_obs_update` `:652-845`, `_publish_loop`
`:847-896`, `__init__` `:56-174`, `_setup_config` `:176-270`, `_read_checkpoint_metadata`
`:291-320`), `ee_runtime.py`, and a repo-wide reference map for the deletion.

## THE KEY DECISION — the substitution mechanism

### Where the seam actually is
In the classic (ACT/Diffusion) path, `_obs_update` does, in order:
1. `:660` read observation via `self.strategy`.
2. `:669-738` EE obs conversions for the model input **and** — critically — the capture of
   `self._ee_delta_latest_obs_quat` (`:728-738`, under `_obs_lock`) that the delta publish
   loop consumes every tick.
3. `:740-760` inspect `self.model._action_queue` / `._queues` for chunk timing.
4. `:762-764` `self.preprocessor(...)` (normalize).
5. `:785` `action = self.model.select_action(observation)` — **NORMALIZED** output.
6. `:804-810` `self.postprocessor.process_action(...)` + `.squeeze().cpu().numpy()` — **this
   is the normalized→PHYSICAL-units transition.** First point a physical-units numpy action
   exists.
7. `:823-836` ee_relative chunk restore (guarded `if self.is_ee_relative`; not our case).
8. `:838` `self._classic_action_deque.append(action)`.

The dataset's on-disk `action` column is in **physical units**. So GT must enter the deque at
the physical-units point (step 6's output), NOT at the `select_action` seam (step 5) — the
postprocessor would erroneously denormalize physical GT. The per-type deque contents already
match the on-disk encoding for the three dataset-native types:

| action_type | deque holds | on-disk `action` | inject raw row? |
|---|---|---|---|
| `ee_delta`  | delta (physical) → publish loop composes `obs∘delta` per tick | delta | ✅ |
| `ee_abs`    | absolute rot6d | absolute rot6d | ✅ |
| `joint_abs` | absolute joints | absolute joints | ✅ |
| `ee_relative` | absolute (restored upstream from `model._queues`) | *(no such dataset encoding — `relative` is reserved/unimplemented in mcap_converter)* | N/A, out of scope |

So injecting the raw recorded row into `self._classic_action_deque` and letting the existing
publish loop run is exactly correct for `ee_delta` / `ee_abs` / `joint_abs`.

### Recommended mechanism: subclass + extract a `_produce_action` seam
`DatasetGtReplayerNode(LeRobotInferenceNode)` is a subclass. In the **base class**, extract the
model pipeline into one overridable method — a pure extract-method refactor with **zero
behavior change** for the real node:

- New base method `_produce_action(self, observation, ee_obs_window_rel) -> np.ndarray | None`
  containing today's `:740-836` (model-queue inspection → preprocessor → `select_action` →
  postprocessor → numpy → ee_relative restore) **plus** the `record_inference()` metric
  currently at `:839-840`. Returns the physical-units numpy action (or `None`).
- `_obs_update`'s classic branch becomes: run the shared head (`:660-738`, incl. the
  `_ee_delta_latest_obs_quat` capture), then `action = self._produce_action(observation,
  _ee_obs_window_rel)`, then `if action is not None: self._classic_action_deque.append(action)`.

`DatasetGtReplayerNode` overrides exactly `_produce_action` to ignore its args and return the
next recorded GT row (physical units, native encoding) or `None` at end-of-episode.
**Everything downstream — the deque, `_publish_loop`'s per-tick `ee_delta_restore_step`
composition against the freshest observed pose, `_publish_action`/`_publish_ee_action`/
`_publish_joint_action`, `_setup_publishers`, both timers — is inherited and runs byte-for-byte
as in real inference.** The shared head still runs, so `_ee_delta_latest_obs_quat` is updated
every tick from the live mock/robot observation, which is what makes the delta composition
real; its model-input EE conversions are wasted-but-harmless for the replayer.

### Why not the alternative (full "action-source strategy" injection)
Extracting model + pre/post-processors + the ee_relative chunk-restore state
(`self.model._queues`, `_relative_anchor_state`, `_abs_shadow_queue`, `_ee_raw_obs_buf`) into
an injected `ActionSource` object would be conceptually cleaner but is a **large, risky
refactor of working inference code** touching entangled node state. The subclass +
single extract-method keeps that state on the node untouched and changes the real path by
exactly one method boundary. Chosen for lower blast radius.

### Awkward spots in the current structure (flagged, not silently worked around)
1. **`_obs_update` is a ~190-line method** interleaving (a) publish-loop-needed obs capture,
   (b) model-input EE conversion, and (c) the model call + restore, sharing locals
   (`_ee_obs_window_rel`, `_raw_obs`). The extracted `_produce_action` takes
   `ee_obs_window_rel` as a param.
2. **The physical-units seam is the postprocessor output (`:810`), not `select_action`
   (`:785`).** A naive "fake the model object" approach would be corrupted by the postprocessor.
   `_produce_action` deliberately sits *after* the postprocessor so GT enters already-physical.
   This is the single most important reason the seam is `_produce_action`, not a fake model.
3. **VLA path is a separate seam** (`predict_action_chunk` in a background thread `:602-628`,
   feeding an RTC `ActionQueue`, with normalize/`original`/`inference_delay`/`merge`
   semantics). Injecting GT there means fighting RTC guidance. → **v1 is classic-only.**
4. **Config today is checkpoint-sourced.** `_setup_config` requires `model_path` (`:199-200`),
   reads `action_type`/`obs_state_dim`/`model_type`/`image_shape` from the checkpoint
   (`:237,:251-265`), and `__init__` gates `_setup_model` + publishers + timers behind
   `if not self.echo_topic_only` (`:101-142`). The replayer has a dataset, not a checkpoint.
   Handled by three small base-class seams (below), NOT by abusing `echo_topic_only` (which also
   disables publishers/timers — the wrong tradeoff).

## Base-class changes to `inference_node.py` (minimal, behavior-preserving)

1. **Extract `_produce_action`** as above (pure refactor; real node unchanged).
2. **Extract the model_path requirement** (`:199-200`) into `def _validate_required_params(self)`
   so the replayer overrides it to require `dataset` instead of `model_path`.
3. **Rename `_read_checkpoint_metadata` → `_load_run_metadata`** (behavior kept) so the replayer
   overrides it to return the SAME meta-dict shape
   (`action_type`, `obs_state_dim`, `model_type`, `image_shape`, `task_description`) sourced
   from the dataset. (Rename optional; could keep the name and just override.)
4. Confirmed no other change needed: `strategy` setup (`:64-76`), `_setup_publishers`
   (`:504-534`), and the timers (`:133-142`) are model-independent and already run before /
   independently of model state, so the replayer reuses them as-is.

## New files

- `ros2/src/lerobot_control/lerobot_control/dataset_gt_replayer_node.py` — the subclass. It:
  - declares a `dataset` param + replay params (`episode` default 0, `loop` default false,
    `hold_last` default true, `dry_run` default false);
  - overrides `_validate_required_params` (require `dataset`), `_load_run_metadata` (derive
    `action_type` from the dataset's `conversion_config.yaml` via
    `mcap_converter.config.loader.ConfigLoader.from_yaml(..., strict=False)`:
    `data_space=ee`+`action_encoding=delta`→`ee_delta`, `ee`+`absolute`→`ee_abs`,
    `joint`→`joint_abs`; `model_type` set to a non-VLA value so `_is_vla` is False;
    `image_shape`/`obs_state_dim` from `meta/info.json`), and `_setup_model` (no model — load
    the episode's `action` rows via a direct parquet read into `self._gt_actions` + a cursor);
  - overrides `_produce_action` to return the next GT row (physical units, native encoding) with
    **backpressure** (return `None` without advancing the cursor when the deque is near full, so
    rows are never dropped by the `maxlen=10` deque), and `None` at episode end;
  - handles episode end: log completion once; `hold_last` keeps last command; `loop` restarts;
    `dry_run` logs the would-be row but returns `None` so nothing is published.
- `ros2/src/lerobot_control/launch/dataset_gt_replay.launch.py` — mirror of
  `inference.launch.py`, wiring the new executable's params.

## Edits to existing files

- `ros2/src/lerobot_control/setup.py` — add one `console_scripts` entry:
  `"dataset_gt_replayer_node = lerobot_control.dataset_gt_replayer_node:main"`.
- `ros2/src/lerobot_control/lerobot_control/inference_node.py` — the three base-class seams
  above. No runtime behavior change for real inference.
- `docker-compose.fake-hardware.yml` — add a `replay` profile/service (mounts the converted
  dataset read-only; runs `dataset_gt_replay.launch.py`; `depends_on` `mock-robot` healthy),
  run against `mock-robot` with `EE_MODE=true`. The mock echoes each received
  `/commanded_ee_{arm}` back as the next `/ee_pose_{arm}` observation — closing exactly the same
  loop real ee_delta deployment uses, now driven by GT deltas.

## Deletion of the old math-only tool

Clean — only 3 code edits, rest is docs:
- Delete `packages/anvil_eval/src/anvil_eval/gt_replay.py` (323 lines) and
  `tests/unit/anvil_eval/test_gt_replay.py` (188 lines, sole importer).
- Remove `anvil-gt-replay = "anvil_eval.gt_replay:main"` from `packages/anvil_eval/pyproject.toml:46`.
- Confirmed no other code imports it; the `ConfigLoader(strict=False)` pattern survives
  (also used by `mcap_converter/cli/migrate_config.py:123`).

Doc/comment ripple to correct (not code):
- `claude_docs/ee-delta/2026-07-17-flow-plan.md` (+ `.zh-TW.md`): the "GT-replay = FIRST GATE / must pass
  before Item 4/6" assertions (`:313,:327,:372-375,:509,:529` and zh mirror) — reword: the math
  gate is removed; the new gate is this ROS2 replayer (model-free, robot-in-the-loop).
- `configs/mcap_converter/v1.1/TEMPLATE_all_fields.yaml:31` and
  `openarm_ee_bimanual_delta_debug.yaml:24`: the "gate with `anvil-gt-replay`" next-step lines —
  repoint to the new tool.
- `claude_docs/ee-delta/2026-07-16-architecture-report.md` (+ zh) and
  `claude_docs/ee-delta/2026-07-17-libero-vs-production-diagnosis.md`: historical mentions — add a "later removed"
  note; do not rewrite history.
- `tests/unit/mcap_converter/test_ee_encoding.py:588`: docstring mention only — safe, optional.

## Non-goals for v1

- **VLA policies** (pi0/pi05/smolvla) — separate background-thread + RTC seam; RTC guidance
  would distort injected GT. Classic (ACT/Diffusion deque) path only.
- **`ee_relative`** — not a dataset-native encoding; deque holds upstream-restored absolutes
  coupled to `model._queues`. Out of scope.
- **Automated task-success detection** — none exists in the repo; "did the task reproduce" stays
  human/video review. The mock is a pure echo (no physical dynamics), so mock-level replay
  validates software/timing/topic-wiring + composition correctness, not physical task
  completion — that only becomes meaningful on real hardware.
- **Real-hardware execution** — deferred until fake-hardware validates cleanly; `dry_run` +
  fake-hardware-first are the safety staging.

## Verification

1. `mcap-convert` `data/raw_sessions/ee-space-testing` into `ee-abs` and `ee-delta` variants
   (both already validated).
2. `colcon build` inside the container so the new executable + `_produce_action` refactor
   compile; run existing ROS2/unit tests to confirm the extract-method refactor didn't change
   inference behavior.
3. `EE_MODE=true DATASET_PATH=<ee-delta dataset> docker compose -f
   docker-compose.fake-hardware.yml --profile replay up --build` alongside `mock-robot`. First
   pass `DRY_RUN=true` (logs rows, publishes nothing), then a real publish pass.
4. `ros2 bag record /ee_pose_{arm} /commanded_ee_{arm}` during the run. Because the mock echoes
   commands back as observations and the delta publish loop composes against them, a correct
   ee_delta dataset should make the echoed `/ee_pose_{arm}` track the dataset's recorded
   `observation.state` trajectory. Confirm visually/in logs (pose continuity, publish
   FPS ≈ control_freq, no deque-overflow row drops). Repeat for `ee-abs` (deque-passthrough).
5. Regression: full `uv run pytest` after the `gt_replay.py` deletion (deleted test gone,
   nothing else references it).

## Accepted awkwardness (documented, not hidden)
- The replayer runs `_obs_update`'s model-input EE conversions every tick and discards them.
  Harmless (pure math on the observation dict), kept to avoid forking `_obs_update`'s head.
- Replay rate = `control_freq` (one row per publish tick), not the dataset's native fps — which
  is *correct*: it matches real deployment cadence and the real ee_delta per-tick composition
  rate. To replay at recorded wall-clock rate, set `control_frequency` to the dataset fps. No
  separate replay-rate timer (that would duplicate timer machinery).
