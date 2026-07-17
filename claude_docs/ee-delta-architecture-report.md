# EE Delta-Flow Pipeline — Architecture & Design Reference

**Branch:** `patrick/implement-ee-space` · **Generated:** 2026-07-16
**Scope:** the seven items built on this branch — terminology rename, `ee_delta_forward`/`ee_delta_inverse`, mcap_converter convert-time baking, `EEDeltaTransform`/stats, the decoupled delta-mode publish loop, the fake-hardware EE extension, and the GT-replay tool.

## How to read this document

This is a deep-dive reference, not a status update. It is organized shallow → deep:

- **Level 1** — one paragraph per feature: what/why/where.
- **Level 2** — one execution trace per user-facing workflow (data conversion, training, inference, offline validation), naming every file:function touched in call order.
- **Level 3** — file-by-file, function-by-function detail in the order established by Level 2, with exact current line numbers, callers/callees, and non-obvious design decisions.
- **Terminology rename audit**, **deviations from the plan**, and **known bugs/rough edges** are each consolidated into their own section at the end, pulling together findings scattered across all four workflows.

Every file referenced was re-read in full from the current working tree — which includes **both** commits already on this branch **and** uncommitted changes sitting on top of them (`git diff main` was used throughout, not `git diff <merge-base>`, since a large second layer of edits exists uncommitted). Line numbers are cited against that current state, not against the plan document's own (now stale) citations. Where the plan and the code disagree, both versions are stated — the deviation is never silently normalized away.

Two planning documents are referenced throughout:
- `claude_docs/ee-delta-flow-plan.md` — the design plan (rotation math, the six-item plan, terminology lock-in proposal).
- `claude_docs/ee-space-libero-vs-production-diagnosis.md` — the failure diagnosis that motivated this branch (n-0 chunk-wise relativization + normalization-range compression as primary cause).

A third doc, `docs/relative_ee_failure_analysis.md`, predates both — it's an earlier, since-superseded diagnosis (different primary-cause ranking) that this branch touched only to add a two-line "historical record" header note; its actual conclusions were left unreconciled with the later diagnosis (see Level 3, anvil_eval section).

---

# Level 1 — Feature Overview

**1. Terminology rename (`ee_rel` → `ee_relative`).** The pre-existing n-0 (chunk-start-anchor) mechanism was renamed end-to-end — `ee_rel_forward`/`ee_rel_inverse` → `ee_relative_forward`/`ee_relative_inverse`, `EERelTransform` → `EERelativeTransform`, `_delta_ref_state` → `_relative_anchor_state`, etc. — because the old name "delta" collided with the new Delta(n-(n-1)) concept this branch introduces, and "rel" no longer means "any non-absolute representation" once a second non-absolute representation exists. It exists purely to keep the two mechanisms unambiguous in code, config, and logs going forward, and it touches every workflow, since `action_type` is the one string that flows through data conversion (not directly, but as a sibling concept), training, inference, and offline eval.

**2. `ee_delta_forward` / `ee_delta_inverse`.** A new pair of SE(3) transform functions in `anvil_shared/ee_transform.py` implementing world-frame, per-frame single-step delta: `delta_xyz = action_xyz - state_xyz` (plain subtraction, no rotation) and `R_delta = R_action @ R_state.T` (world/extrinsic composition), with the inverse `abs = state + delta` / `R_abs = R_delta @ R_state`. These are new functions, not wrappers around the existing body-frame `ee_relative_*` pair, because the target composition (verified against robosuite 1.4.0's OSC controller source) uses the opposite frame convention and multiplication order. This is the mathematical core everything else in the branch is built on top of; it belongs to no single workflow but is *used by* data conversion (baking) and inference (restoring), and is *validated by* GT-replay.

**3. mcap_converter convert-time baking.** Rather than computing the delta live at training time, `mcap_converter` now bakes `ee_delta_forward` directly into the on-disk `action` column when a config sets `ee_action_encoding: "delta"` (new `DataConfig` field, default `"absolute"`). `observation.state` stays absolute in both modes. This is a static, independently-inspectable value — GT-replay only has to check "is this on-disk number correct," not "does re-invoking this code produce the same result every time." Belongs to the **data conversion** workflow (`mcap-convert` CLI).

**4. `EEDeltaTransform` / `_compute_ee_delta_stats`.** The training-side counterpart: a per-sample dataset transform that converts `observation.state` from quaternion(8n) to rot6d(10n) layout (mirroring `EEAbsTransform`'s obs-only path, not `EERelativeTransform`'s relativizing path) while leaving `action` completely untouched — the delta was already baked at convert time. `_compute_ee_delta_stats` computes normalization stats by reading mean/std/min/max straight off the static baked column (no live replay), replicating the existing epsilon-floor and rot6d-identity-clamp guards. Belongs to the **training** workflow (`anvil-trainer --action-type=ee_delta`).

**5. Decoupled delta-mode publish loop.** On the ROS2 inference side, the action queue now stores raw model-output deltas (not pre-restored absolutes) for `ee_delta` mode. A separate publish-loop timer, independent of the observation-sampling timer, reads the freshest real observation at each tick and composes `absolute_target = obs_pose ∘ delta` fresh, every tick, via the new `ee_delta_restore_step` (`ee_runtime.py`). This mirrors robosuite's per-execution-step anchoring (what LIBERO's validated conditions actually did) instead of restoring against one fixed chunk-start anchor. Belongs to the **inference** workflow (ROS2 `inference_node.py` launch).

**6. Fake-hardware EE extension (Item 2b).** The existing joint-space-only mock hardware node (`fake_hardware_node.py`) gained a `CommandedEEPose` publisher (mocking `/ee_pose_<arm>`) and a subscriber on `/commanded_ee_<arm>` that echoes the received command back as the next published observation — i.e. `next_obs ≈ last_received_command`. This is required specifically to exercise the delta-mode publish loop's self-correction feedback (`obs_pose ∘ delta`) in a multi-container CycloneDDS environment without real hardware. Belongs to the **inference** workflow, as an intermediate validation rung between unit tests and real hardware.

**7. GT-replay tool (`anvil-gt-replay`).** A brand-new, model-free CLI (`packages/anvil_eval/src/anvil_eval/gt_replay.py`) that validates the transform math in isolation: for `--encoding absolute` datasets it round-trips `ee_delta_forward`→`ee_delta_inverse` against consecutive ground-truth poses; for `--encoding delta` datasets it recomputes the expected baked delta from the raw absolute pose sequence and diffs it against what `mcap_converter` actually wrote on disk. No model, no checkpoint. Exits non-zero on any episode failing a machine-precision tolerance. Belongs to the **offline validation** workflow, and is the first gate every converted delta dataset must pass before training compute is spent on it.

---

# Level 2 — End-to-end flow per workflow

## Workflow A — Data conversion (`mcap-convert`)

Entry point: `packages/mcap_converter/src/mcap_converter/cli/convert.py:main` (line 584).

1. `main()` parses argv (611-686) — **no CLI flag exists for `ee_action_encoding`**; it is config-file-only.
2. Loads config: `ConfigLoader.from_yaml(args.config)` (`loader.py:60`) → `load_yaml` (`loader.py:52`) → `ConfigLoader.from_dict` (`loader.py:141-193`). `from_dict` reads `ee_action_encoding` (loader.py:150-156, defaults `"absolute"`, validated inline against `("absolute","delta")`) and constructs `DataConfig(...)`.
3. `validate_config(config)` (`validators.py:44-127`) re-validates `ee_action_encoding` (88-92) and rejects `ee_action_encoding != "absolute"` when `data_space != "ee"` (93-98).
4. `main()` line 710: `space_suffix = "ee-delta-space" if config.is_ee_delta else f"{config.data_space}-space"` — `is_ee_delta` (`schema.py:152-155`) is the single property every consumer checks; this is the only place delta encoding changes the **output path**.
5. `LeRobotWriter.create_dataset(...)` → `_define_features` (`writer.py:180`) — does **not** branch on `is_ee_delta` (only on `is_ee`, writer.py:217); absolute and delta EE datasets get byte-identical schemas.
6. Per MCAP file: `extractor.extract_frames(mcap_path, task)` (`extractor.py:647-895`) — the generator that does per-frame work. It threads a `prev_ee_state` local (line 748, `None` at episode start — genuine self-anchor per episode, no cross-episode leakage) through `_align_frame_at_cursor` (897-967, `prev_ee_state` param added at 906) into `_align_ee_signals`.
7. `_align_ee_signals(ee_buffers, target_ts, prev_state=None)` (`extractor.py:1357-1424`) — **this is where the branch happens**: builds `state_abs`/`action_abs` unconditionally absolute (1390-1409), then `if self.config.is_ee_delta:` (1411) lazy-imports `ee_delta_forward` (anvil_shared.ee_transform), sets `anchor = state_abs if prev_state is None else prev_state` (1416, the first-frame self-anchor convention), and computes `action_out = ee_delta_forward(action_abs, anchor)` (1417) — called **once** over the full concatenated multi-arm array, not per-arm (per-arm looping happens inside `ee_delta_forward` itself via `n_arms_from_dims`). `observation.state` is returned unchanged (always `state_abs`).
8. Control returns up through `extract_frames` → `convert.py`'s per-episode loop → `writer.add_episode(...)` → `writer.finalize(dataset)`.

**Where `ee_action_encoding` is read, end to end:** YAML key → `loader.py:150-156` → `DataConfig.ee_action_encoding` (`schema.py:124`) → `DataConfig.is_ee_delta` property (`schema.py:152-155`) → exactly three call sites: `convert.py:710` (output path), `extractor.py:1411` (branch to baking), `validators.py:88-98` (input validation). No other file reads this field.

**Item 1 (baking) plugs in at step 7.** Item 2 (rotation math) is the library `ee_delta_forward` itself calls into. The terminology rename is orthogonal to this workflow (mcap_converter's own naming — `ee_action_encoding`/`action_type="ee_delta"` — was clean from the start, never used the old "rel"/"delta" vocabulary).

## Workflow B — Training (`anvil-trainer --action-type=ee_delta`)

Entry point: `anvil_trainer.train:main()` (`train.py:273-281`) → `train()` (`train.py:62-133`).

1. `config = TrainingConfig.from_env_and_args()` (`train.py:75` → `config.py:186-476`): `action_type = _pop_argv("action-type") or "joint_abs"` (204), validated against `_VALID_ACTION_TYPES = {"joint_abs","ee_abs","ee_relative","ee_rel","ee_delta"}` (config.py:81, 220-224), then normalized via `anvil_shared.action_types.normalize_action_type` (226) — a no-op for `ee_delta` since it isn't an alias of anything.
2. `config.validate_action_space()` (`train.py:82` → `config.py:515-586`) checks `self.action_type in ("ee_abs","ee_relative")` (543) to decide whether to require EE dataset markers — **`ee_delta` is excluded from this tuple, so no dataset-shape validation runs for it at all** (see Rough Edges).
3. `with patched_lerobot(config):` (`train.py:105` → `patches.py:1181-1213`) builds `TransformRunner(config)` (patches.py:144-165), which registers `EEDeltaTransform()` among five transform instances (patches.py:146-153) — only `EEDeltaTransform.is_enabled(config)` (`transforms.py:303-304`, → `config.is_ee_delta`) returns `True` for an ee_delta run.
4. `runner.apply_metadata_patches()` (patches.py:1199 → 652-655) → `EEDeltaTransform.patch_metadata` (`transforms.py:332-337`) → `_patch_obs_state_shape_8n_to_10n` (transforms.py:174-209) monkey-patches lerobot's `dataset_to_policy_features` so the policy factory sees `observation.state` as 10n-dim (matching what `EEDeltaTransform.apply` will actually produce per-sample).
5. `runner.apply_dataset_patches()` (patches.py:1204 → 657-695) installs `patched_getitem` (675-692), which calls `EEDeltaTransform.apply(item, config)` (`transforms.py:306-330`) on every sample: converts `observation.state` via `ee_obs_abs_forward` (308, 316); **never reads or writes `item["action"]`** — the delta is already baked, so the entire double-transform-avoidance guarantee rests on this method simply not touching that key (no explicit guard, just omission — see Rough Edges).
6. `runner.apply_val_loss_patch()` (patches.py:1205 → 697-846) installs `patched_make_dataset`, which — once per run — dispatches `elif val_state.config.is_ee_delta: _patched_ee_stats = val_state._compute_ee_delta_stats(full_dataset, cfg)` (743-744) → **`TransformRunner._compute_ee_delta_stats`** (patches.py:451-559): reads `actions_np`/`states_np` straight off `full_dataset.hf_dataset` (no live transform replay), computes epsilon-floored mean/std/min/max (498-501), applies `_force_rot6d_identity` (503), converts obs to rot6d via `ee_obs_abs_forward` (518), and logs an explicit `"[ee_delta_stats] COMPLETED and INJECTED (not the dataset-stats fallback)"` line (542-548) before the caller injects the result into `train_dataset.meta.stats["action"]`/`["observation.state"]` (patches.py:826-828) — the actual dataset object lerobot's normalizer reads from.
7. `runner.apply_checkpoint_patch()` (patches.py:1206 → 848-943) writes `anvil_config.json` per checkpoint (`action_type`, `is_ee`, `is_ee_relative`, git provenance) — **`is_ee_delta` is not one of the persisted keys** (see Rough Edges).
8. During the actual training loop, every `__getitem__` call runs `EEDeltaTransform.apply` from step 5.

**Item 4 (EEDeltaTransform/stats) is this entire workflow.** The terminology rename shows up as `EERelativeTransform`/`_compute_ee_relative_stats`/`is_ee_relative` existing alongside the new `EEDeltaTransform`/`_compute_ee_delta_stats`/`is_ee_delta` as siblings in the same `TransformRunner`.

## Workflow C — Inference (ROS2 launch, `ee_delta` mode)

Entry point: `ros2/src/lerobot_control/launch/inference.launch.py` → `inference_node.py:main()` (1370) → `LeRobotInferenceNode.__init__` (56).

**Startup:** `_setup_config()` (176) reads checkpoint metadata and calls `resolve_action_type(meta)` (`ee_runtime.py:54`, the single chokepoint that maps legacy `"ee_rel"` → `"ee_relative"`), setting `self.action_type`/`is_ee`/`is_ee_relative`/`is_ee_abs`/`is_ee_delta` (252-259). `strategy.setup(...)` (65) → `multi_process.py` independently decides EE-vs-joint from the YAML config (`ee_arms = {name: ac for ... if "ee_command_topic" in ac}`) and, for EE arms, subscribes `CommandedEEPose` on `ee_obs_topic` (default `/ee_pose_<arm>`) via `_setup_ee_subscriptions` — **this is a second, independent EE-mode detector from `resolve_action_type`'s, and nothing enforces the two agree** (see Rough Edges). Two independent timers are created on separate callback groups: `_obs_timer → _obs_update` and `_publish_timer → _publish_loop` (133-142) — this decoupled-timer architecture pre-dates this branch; the branch's contribution is what each timer *does* for `ee_delta` mode.

**Per-tick, `_obs_update()` (652), on the obs callback group:**
1. `strategy.get_observation(...)` → `multi_process.py:_build_observation()` — EE branch concatenates each arm's latest `CommandedEEPose` message into `observation["observation.state"]` (quat(8n), absolute — this is the *measured* pose).
2. Because `self.is_ee_delta` (not `is_ee_relative`): `ee_obs_abs_forward` converts quat(8n)→rot6d(10n), **absolute, no relativization** (inference_node.py:702-707).
3. The raw quat-layout obs is stashed into `self._ee_delta_latest_obs_quat` under `self._obs_lock` (728-738) — the cross-thread handoff to `_publish_loop`.
4. `_needs_restore = self.is_ee_relative` → `False` for ee_delta (755) → ee_delta never captures a `_relative_anchor_state`.
5. `model.select_action(observation)` (785) → raw normalized action → `postprocessor.process_action(action)` (804) denormalizes — for ee_delta, **this value IS the delta**, physical units, not yet composed against anything.
6. The raw delta is pushed straight into `self._classic_action_deque` (838), **unrestored**.

**Per-tick, `_publish_loop()` (847), on the independent publish callback group:**
1. `if self.is_ee_delta:` (879) reads `self._ee_delta_latest_obs_quat` under `self._obs_lock` (880-881) — the freshest obs from the *most recent* `_obs_update` tick, generally from a different wall-clock moment than when the popped delta was computed (the deliberate decoupling).
2. If no obs yet, returns **without popping** the queued delta (883-886) — doesn't drop it.
3. `action = ee_delta_restore_step(self._classic_action_deque.popleft(), _latest_obs)` (887) → `ee_runtime.py:147` → `ee_delta_inverse`: `abs_xyz = obs_xyz + delta_xyz`, `R_abs = R_delta @ R_state` (world-frame — exactly the plan's "Rotation math (RESOLVED)" formula).
4. `_publish_action(action)` → `_publish_ee_action` (1038) → per arm: `ee_poses_from_chunk` (`ee_runtime.py:197`) → rot6d→quat pose dict → gripper shift/scale/clamp (1086-1089) → `CommandedEEPose` message published to `/commanded_ee_<arm>`.

**Contrast, `ee_relative` (existing n-0) mode:** `_obs_update` relativizes the *whole* obs window to one fixed anchor, captures `_relative_anchor_state` once per new chunk, and calls `ee_relative_restore_chunk` (`ee_runtime.py:103`, body-frame) **once for the whole chunk** — already-absolute actions go into the same deque; `_publish_loop` just pops, no per-tick composition at all. This is the architectural fork Item 2 introduces: ee_delta composes at *publish* time against fresh obs; ee_relative composes (once) at *chunk-generation* time.

**Fake-hardware mock loop (Item 2b), separately:** `fake_hardware_node.py:MockControllerNode._setup_ee_mode()` (187) creates a `CommandedEEPose` publisher on `/ee_pose_<arm>` and a subscriber on `/commanded_ee_<arm>` → `_ee_command_callback` (241) **overwrites** `self._ee_state[arm]` wholesale with the received command (264-266, instantaneous echo, no simulated dynamics) → a separate timer `publish_ee_poses()` (221, default 100 Hz, independent of `control_freq`) republishes the current state as the next observation. Full loop: `inference_node._publish_loop` → `/commanded_ee_<arm>` → (DDS transport) → `fake_hardware_node._ee_command_callback` → next `publish_ee_poses()` tick → real `inference_node._obs_update` picks it up via `multi_process.py`'s subscription callback — this is what lets the delta-mode self-correction (`obs_pose ∘ delta`) be exercised end-to-end without real hardware.

## Workflow D — Offline validation

Two genuinely separate tools live under this workflow; do not conflate them.

**D1. GT-replay (`anvil-gt-replay`, model-free, the "first gate").** Entry: `gt_replay.py:main()` (239) → `parse_args()` (215). `_detect_encoding(dataset_path)` (107, only when `--encoding auto`) reads `<dataset_root>/conversion_config.yaml`'s `ee_action_encoding` key directly (a second, ad-hoc YAML read — **not** through `mcap_converter`'s own config loader). `main()` reads `<dataset_root>/meta/info.json` directly for `total_episodes` — **does not use `anvil_eval.dataset.EvaluationDataset`** at all, a deliberate deviation from the plan (see Deviations). Per episode: `_load_episode_arrays(dataset_root, episode_idx)` (125) reads parquet files directly via pandas — bypassing `LeRobotDataset`/training transforms "to keep this tool independent of any training-time transform." Two replay functions:
- `_replay_episode_absolute` (155): pure math self-consistency — `ee_delta_forward` then `ee_delta_inverse` against consecutive ground-truth poses, no external reference (a documented limitation: this mode can't detect corruption, only broken math — see the test suite's own docstring).
- `_replay_episode_delta` (171): checks the first-frame self-anchor invariant against real converter output, then reconstructs `action_abs = ee_obs_abs_forward(states)`, computes `expected = ee_delta_forward(action_abs[1:], states[:-1])`, and diffs against the **actual on-disk baked column** — this is the "is the on-disk number correct" check the plan's Item 3 GT-replay implication calls for.

Errors reduce to two scalars per episode (max position error in m, max rotation error in deg, via `_max_pos_rot_err`/`_rot6d_angle_diff_deg`, 92/80) against machine-precision-strict defaults (`atol_pos=1e-6`, `atol_rot_deg=1e-4`) — no smoke-scale relaxation anywhere, matching the plan's "this bar does NOT relax" requirement. No JSON/CSV output — PASS/FAIL is logged to stdout and signaled via process exit code only.

**D2. Model-based offline eval (`anvil-eval`, pre-existing workflow, extended for EE but NOT for `ee_delta`).** Entry: `anvil_eval/cli.py:main()` (91). Loads `anvil_config.json` → `EvaluationDataset` → `EpisodeEvaluator.__init__` (`evaluator.py:62`) normalizes `action_type` and sets `self.is_ee = action_type in ("ee_abs","ee_relative")` — **`ee_delta` excluded**. `evaluate_episode` (93) branches obs-conversion and chunk-restore only on `is_ee_relative`/`is_ee_abs` — **no branch exists for `ee_delta` anywhere**, so an ee_delta checkpoint's obs stays 8-dim quat (wrong — the trained checkpoint expects 10-dim rot6d) and no restore is applied at all. Back in `cli.py`, `if evaluator.is_ee:` (194) gates the EE-aware metric-space conversion; being `False` for `ee_delta`, it falls through to the **generic, non-EE** metrics path in `metrics.py:compute_episode_metrics` (136, gate at 198 also omits `ee_delta`) — producing a numerically meaningless MAE/MSE that mixes metres, unitless rot6d components, and gripper units. `plotting.py`'s `show_delta`/`_is_ee` gates (48, 140, 148) have the same omission, so an ee_delta run gets no EE-aware plot at all. **This is the plan's Item 5 ("downgraded bar" offline eval) simply not implemented for `ee_delta`** — see Deviations.

`anvil_eval_ros/cli.py` (the separate ROS-in-the-loop MCAP-replay tool) has the same gap: its `is_ee = action_type in ("ee_abs","ee_relative","ee_rel")` check also excludes `ee_delta`.

---

# Level 3 — File-by-file, function-by-function detail

## A. Data conversion pipeline

### `packages/mcap_converter/src/mcap_converter/cli/convert.py`
Role: CLI entry point orchestrating config load/validate, writer/extractor construction, per-episode loop, final report.

- `main(args=None)` (584-…): argparse (611-686, no `--ee-action-encoding` flag — config-file-only, unlike joint mode's `--act-from-obs` CLI override — a deliberate omission per the file's own design, not an oversight: delta encoding is more consequential and better pinned in a versioned config). Loads+validates config (690-702). Computes `space_suffix`/output dir (704-717, delta gets `ee-delta-space/`, line 710). Prints a startup banner (776-803) showing `config.data_space` but **never surfacing `ee_action_encoding`** — a delta run's console output looks identical to an absolute run except for the output-path suffix.
- Conversion-loop body (~200-500): `LeRobotWriter` construction (241-249); EE mode uses empty joint-name dict (252-259, EE has fixed per-arm dims); `conversion_config.yaml` re-serialization (303-349) — **drops `ee_action_encoding`** when no `--config` path is given (the field survives only because the normal path copies the source YAML verbatim); zero EE debug-plot support, absolute or delta (518-530, pre-existing gap, unchanged).

### `packages/mcap_converter/src/mcap_converter/config/schema.py`
Role: dataclasses for the unified joint+EE converter config.

- `DataConfig` (102-194): `ee_action_encoding: str = "absolute"` (124), with a 9-line rationale comment (117-123) — a scalar flag, not a new `data_space` value, so `is_ee` (148-150) and every existing EE branch site stay untouched.
- `is_ee_delta` (152-155, new): `data_space == "ee" and ee_action_encoding == "delta"` — the single property every consumer checks; nothing else checks the raw field value directly except the loader/validator's own parse-time reads.

### `packages/mcap_converter/src/mcap_converter/config/loader.py`
- `from_dict(config_dict)` (141-193): reads+validates `ee_action_encoding` (150-156) *before* delegating to `validate_config` — raises a bare `ValueError` here; `validators.py` raises a `ConfigurationError` for the same constraint. Because `from_dict` always runs first in the real call chain, the `validators.py` branch for this specific check is dead code in practice (only reachable if some other caller constructs `DataConfig` directly and calls `validate_config` without going through `ConfigLoader` — which some unit tests do deliberately, to test the loader path specifically).

### `packages/mcap_converter/src/mcap_converter/config/validators.py`
- `validate_config(config)` (44-127): EE branch (82-98) — `action_topics` must be empty in EE mode (83-87, pre-existing, unrelated to delta); new: `ee_action_encoding not in ("absolute","delta")` is an error (88-92); new: `ee_action_encoding != "absolute"` while `data_space != "ee"` is an error (93-98) — setting delta encoding on a joint config is explicitly rejected, not silently ignored.

### `packages/mcap_converter/src/mcap_converter/core/extractor.py`
Role: the actual per-frame extraction/alignment generator — where baking happens.

- `extract_frames(mcap_path, task)` (647-895): `prev_ee_state: Optional[np.ndarray] = None` local (748), reset every call (i.e. every episode gets a genuine self-anchor, no cross-episode leakage since this function runs once per MCAP/episode). Updated at **both** yield sites — the main streaming loop (809) and the buffer-flush tail (850) — a function with two yield paths that both needed the same patch; easy to miss one (didn't happen here, worth re-checking if this function is touched again).
- `_align_frame_at_cursor(..., prev_ee_state=None)` (897-967): pure plumbing, threads `prev_ee_state` to `_align_ee_signals` (954) when `config.is_ee`.
- `_align_ee_signals(ee_buffers, target_ts, prev_state=None)` (1357-1424) — **the core baking function**. Per-arm loop (1392-1406) builds `state_slices`/`action_slices` unconditionally absolute — the delta transform runs *after* concatenation (1408-1409), not inside the per-arm loop (a structural deviation from the plan's phrasing — see Deviations). `if self.config.is_ee_delta:` (1411) lazy-imports `ee_delta_forward` (mirrors the file's existing lazy-import convention for `anvil_shared.rotation` at 1388, keeping mcap_converter free of a hard `anvil_shared` transform dependency except when actually used); `anchor = state_abs if prev_state is None else prev_state` (1416, first-frame self-anchor, exactly matching the plan and the pre-existing `ee_delta_forward` identity-delta unit test precedent); `action_out = ee_delta_forward(action_abs, anchor)` (1417, single call over the full multi-arm concatenated arrays). Returns `observation.state` always absolute, `action` branching absolute/delta.

### `packages/anvil_shared/src/anvil_shared/rotation.py`
Role: low-level SO(3)/rot6d/quaternion primitives shared by both `ee_relative_*` and `ee_delta_*`. Not new to this branch's delta work specifically (reused, not authored fresh), but its batch variants (`quats_to_matrices`, `matrices_to_rot6d`, `rot6ds_to_matrices`, `matrices_to_quats`) are what let `ee_delta_forward`/`ee_delta_inverse` operate uniformly on a single per-arm slice or a `(T, ...)` batch via plain numpy broadcasting, with no shape-branching in the caller.
- **Rough edge:** `rot6ds_to_matrices` (219-241, batch) *clamps* near-zero-norm columns to `1e-10` instead of raising; the non-batch `rot6d_to_matrix` (90-115) *raises* `ValueError` on the same condition. The batch function's docstring justifies this as being "to avoid masking bugs in downstream code" — which reads backwards, since silently clamping is what typically *masks* a bug, while the scalar function raises for exactly that reason. Predates this branch, but directly relevant to auditing `ee_delta_forward`/`ee_delta_inverse`'s behavior on degenerate input, since both call the batch primitives.

### `packages/anvil_shared/src/anvil_shared/ee_transform.py`
Role: the `ee_relative_*` (renamed) and `ee_delta_*` (new) SE(3) transform pair, plus obs-only helpers and layout converters.

- `n_arms_from_dims(state_dim, action_dim)` (65-86): unchanged validation helper (state dim must be a positive multiple of 8, action dim must be `10*n`); called by every top-level transform function to determine the per-arm loop bound.
- `ee_relative_forward`/`ee_relative_inverse` (89-228): **renamed** from `ee_rel_forward`/`ee_rel_inverse` — logic unchanged. BODY-frame: `R_state.T @ world_delta` (translation), `R_state.T @ R_action` (rotation), with a `per_sample_state` branch (130, 200) switching between a single reference state and per-timestep states (used when computing dataset-wide stats, where every frame needs its own anchor). This is the mechanism the module docstring (28-34) now explicitly labels "the diagnosed root cause of the real-hardware jitter failure" — kept only for the existing `ee_relative`/legacy-`ee_rel` action_type.
- **`ee_delta_forward(action_abs, state)` (231-309, new):** WORLD-frame forward. Translation (298): `act_xyz - state_xyz`, deliberately **no** rotation by state (unlike the relative pair) — the docstring (241-244) states this is verified against robosuite 1.4.0's OSC composition (`goal_position = current_position + delta`), not an arbitrary choice. Rotation (300-306): `R_delta = R_action @ R_state.T` — the **opposite multiplication order** from `ee_relative_forward`'s `R_state.T @ R_action`; getting this backwards would silently produce a body-frame-equivalent result under the wrong name, which is exactly what `test_ee_transform.py:712-735` guards against (asserts `ee_delta_forward`'s output does NOT equal `ee_relative_forward`'s for the same non-trivial input). No `per_sample_state` branching is needed here (270-272) — a genuine simplification, not just fewer lines: the relative pair's branching exists because its body-frame translation term is computed differently for 1-D vs batched state, while the world-frame translation term is a plain elementwise subtract in both cases.
- **`ee_delta_inverse(delta, state)` (312-380, new):** exact algebraic inverse — `state_xyz + delta_xyz` (370), `R_abs = R_delta @ R_state` (372-375, same order/side as robosuite's own `goal_orientation = delta_rotation @ current_orientation`). Docstring (332-335) spells out the algebraic proof (`(R_action @ R_state.T) @ R_state = R_action` since `R_state` is orthonormal) explicitly, rather than leaving it for the reader to re-derive. This is the function the decoupled publish loop calls at inference time (via `ee_runtime.ee_delta_restore_step`).
- `ee_obs_relative_forward` (383-443): renamed from `ee_obs_rel_forward`, unchanged logic.
- `ee_obs_abs_forward` (446-487): unchanged function, but newly load-bearing for the delta path — both `EEDeltaTransform.apply` (training) and `gt_replay.py`/`test_ee_encoding.py` (validation) use it to reconstruct `action_abs` from `observation.state` alone, since obs stays absolute regardless of encoding.
- `ee_rot6d_to_quat_layout`, `ee_quat_layout_names`, `ee_action_to_poses` (490-599): unrelated to EE-delta; layout helpers used by `anvil_eval`'s CLI/ROS publish paths.

### `packages/anvil_shared/src/anvil_shared/action_types.py` (new file)
Role: intended single source of truth for the legacy `action_type` alias.
- `ACTION_TYPE_ALIASES = {"ee_rel": "ee_relative"}` (25-27).
- `VALID_ACTION_TYPES = frozenset({"joint_abs","ee_abs","ee_relative","ee_rel"})` (30) — **does not include `"ee_delta"`**, despite the module's own docstring framing itself as shared across `anvil_trainer`/`anvil_eval`/`anvil_eval_ros`/the ROS2 node. In practice `anvil_trainer.config` maintains its *own*, separate `_VALID_ACTION_TYPES` (config.py:81) that does include `ee_delta` and never imports this frozenset — so this module is not actually the single source of truth its docstring claims (see the consolidated Rough Edges section).
- `normalize_action_type(action_type)` (33-42): idempotent alias resolution, the pattern every new action_type addition should follow — `ee_delta` doesn't need an alias (nothing legacy to map from), which is why its absence from `VALID_ACTION_TYPES` specifically (a different concern from the alias dict) is the actual gap.

### `packages/anvil_shared/src/anvil_shared/__init__.py`
- Imports `ACTION_TYPE_ALIASES`/`VALID_ACTION_TYPES`/`normalize_action_type` and rotation primitives into the package namespace (2-18), but **never imports anything from `anvil_shared.ee_transform`**, while `__all__` (22-35) still lists `"ee_obs_abs_forward"` at line 34. This is a real, verifiable bug: `from anvil_shared import ee_obs_abs_forward` (or `import *`) raises `AttributeError`/`ImportError`. Currently latent/unexercised because every actual call site in the repo imports `from anvil_shared.ee_transform import ee_obs_abs_forward` directly (confirmed via repo-wide grep — zero call sites use the package-root form).

### Configs and fixtures
`configs/mcap_converter/openarm_ee_bimanual.yaml`, `openarm_ee_bimanual_16x9.yaml`, `openarm_ee_left.yaml` (all EE, all default `ee_action_encoding` to `"absolute"` by omission), `openarm_joint_bimanual.yaml` (unrelated — unified-config-format migration of `openarm_bimanual_quest.yaml`), `tests/smoke/fixtures/configs/mcap-converter-smoke-test-ee.yaml`, `tests/smoke/fixtures/ee-session/*` (5-episode fixture set), `tests/smoke/fixtures/scripts/generate_ee_fixtures.py`. **None of these — nor any other YAML in the repo — ever sets `ee_action_encoding: "delta"`** (see Deviations, this is the single most concrete "not yet wired up" finding across the whole branch).

`tests/unit/mcap_converter/test_ee_encoding.py` exercises the delta path exclusively by constructing `DataConfig(..., ee_action_encoding="delta")` directly in Python — never through an actual CLI/YAML invocation.

---

## B. Training pipeline

### `packages/anvil_trainer/src/anvil_trainer/train.py`
Role: thin CLI entry point.
- `train(config)` (62-133): wraps `lerobot_train.init_logging` in a self-restoring closure (112-131) purely to emit a resume-summary log line — cosmetic, unrelated to ee_delta.
- `_ANVIL_HELP` (141-251): **already fully documents `ee_delta`** (161-164 example command, 191-197 flag description) — this text is up to date, unlike `docs/training.md` (see Deviations).

### `packages/anvil_trainer/src/anvil_trainer/config.py`
Role: `TrainingConfig` dataclass + argv/env parsing + validation.
- `_VALID_ACTION_TYPES = {"joint_abs","ee_abs","ee_relative","ee_rel","ee_delta"}` (81) — a **second, separate** "valid types" set from `anvil_shared.action_types.VALID_ACTION_TYPES`, which lacks `ee_delta` (see Rough Edges — two sources of truth that can drift).
- `is_ee` (170-172): `action_type in ("ee_abs","ee_relative","ee_delta")` — correctly includes `ee_delta`.
- `is_ee_relative` (174-176), `is_ee_abs` (178-180), `is_ee_delta` (182-184, new): simple equality checks.
- `from_env_and_args()` (186-476): `data_space = "ee" if action_type in ("ee_abs","ee_relative") else "joint"` (343) — **`ee_delta` is excluded**, so an ee_delta run's checkpoints are computed into `model_zoo/joint-space/<dataset>/<run>/` instead of an EE-space directory (real bug, unfixed, not called out anywhere in code or the plan).
- `validate_action_space()` (515-586): `self.action_type in ("ee_abs","ee_relative")` (543) gates whether EE dataset markers are required — **`ee_delta` excluded**, so this function silently validates nothing for `ee_delta` (falls through both branches). Looks like an oversight given the plan's own emphasis on gating dataset/action-type mismatches for the other EE types.

### `packages/anvil_trainer/src/anvil_trainer/patches.py`
Role: all lerobot monkey-patches, installed/torn down via `TransformRunner`.
- `_force_rot6d_identity(min_arr, max_arr, n_arms, dim_per_arm=10)` (61-77): in-place clamps rot6d dims (index 3-8 per arm) to ±1 — shared by all three `_compute_ee_*_stats` methods; exists because under MIN_MAX normalization forcing the range to exactly `[-1,1]` makes rot6d pass through unchanged (rot6d values are already unit-vector-bounded, not because the real data happens to span that range).
- `TransformRunner.__init__` (144-165): registers `EEDeltaTransform()` alongside the other four transforms (151, new this branch).
- `_get_transform_details` (267-289): `elif isinstance(transform, EEDeltaTransform): return "Delta(n-(n-1)): world-frame, baked on disk by mcap_converter — action untouched here"` (287-288).
- `_compute_ee_relative_stats` (291-449) — pre-existing n-0 mechanism's stats method, renamed from `_compute_ee_rel_stats`; live-replays `ee_relative_forward` over every `action_delta_indices` offset, pooled across the whole horizon, episode-boundary-masked. Untouched in mechanism, kept as-is for `ee_relative`.
- **`_compute_ee_delta_stats(self, full_dataset, cfg)` (451-559, new):** guard `if not self.config.is_ee_delta: return None` (474-475); reads `actions_np`/`states_np` straight from `full_dataset.hf_dataset` (484-486) — **no live transform replay**, since the action column is already static; `n_arms = n_arms_from_dims(...)` (491); action stats via plain `mean()`/`std()`/`min()`/`max()` (498-501), epsilon-floored (`np.where(std<1e-6, 1e-6, std)`, 499 — replicates the plan's required guard verbatim); `_force_rot6d_identity(act_min, act_max, n_arms)` (503); obs stats via `ee_obs_abs_forward(states_np)` (518, matching the "mirror EEAbsTransform's obs treatment" decision); explicit completion log (542-548) directly satisfying the plan's "must prove this ran, not the fallback" requirement; `except Exception` falls back with equally explicit `"[ee_delta_stats] FAILED"` wording (552-559) — a `DataIntegrityError` is re-raised, not swallowed (550-551), so only genuinely unexpected failures trigger silent fallback.
- `apply_val_loss_patch`/`patched_make_dataset` (697-846): dispatch chain `is_ee_relative → _compute_ee_relative_stats; is_ee_abs → _compute_ee_abs_stats; is_ee_delta → _compute_ee_delta_stats; else None` (739-746); stats injected into `train_dataset.meta.stats["action"]`/`["observation.state"]` (825-828) — **the mutation that actually matters**, since the earlier in-place mutation of `full_dataset.meta.stats` inside `_compute_ee_delta_stats` itself (512, 534) operates on a *different* dataset object (`full_dataset` vs the filtered `train_dataset`) and is dead-effect work once `full_dataset` goes out of scope (see Rough Edges).
- `apply_checkpoint_patch` (848-943): `anvil_cfg_base` (859-864) persists `action_type`, `is_ee`, `is_ee_relative` — **no `is_ee_delta` key**, despite the property existing and being directly analogous to `is_ee_relative` (see Rough Edges).

### `packages/anvil_trainer/src/anvil_trainer/transforms.py`
- `EEAbsTransform` (217-266): the pre-existing pattern `EEDeltaTransform` mirrors — obs quat(8n)→rot6d(10n) conversion, action passthrough; `_first_apply` gates a one-time log line only (confirmed no numerical difference, matching the plan's "settled, not a real risk" conclusion).
- **`EEDeltaTransform` (274-337, new):** `is_enabled` (303-304) → `config.is_ee_delta`. `apply(item, config)` (306-330): guards on `"observation.state" not in item` (310-311); converts obs via `ee_obs_abs_forward` (308, 316); **never reads or writes `item["action"]`** (318-319, explicit comment) — the double-transform-avoidance guarantee is by omission, not an explicit assertion; the only thing that would catch a future regression here is the dedicated test `test_ee_delta_transform.py::test_action_completely_unchanged_no_double_transform`. `patch_metadata` (332-337) delegates to the shared `_patch_obs_state_shape_8n_to_10n` (174-209). Docstring (274-294) explicitly documents the mirror-`EEAbsTransform`-not-`EERelativeTransform` design decision.
- `EERelativeTransform` (345-425): pre-existing n-0 mechanism, renamed from `EERelTransform`; mechanism untouched.

### `packages/anvil_trainer/src/anvil_trainer/__init__.py`
- `EERelativeTransform` is exported at package level (22, 35); **`EEAbsTransform` and `EEDeltaTransform` are not** — reachable only via `anvil_trainer.transforms.EEDeltaTransform`, not `from anvil_trainer import EEDeltaTransform`. Asymmetric for three otherwise structurally parallel classes.

### Adjacent, not part of the EE-delta mechanism
- `packages/anvil_trainer/src/anvil_trainer/ema.py` (new): a from-scratch `EMAModel` (ported from UMI's diffusion_policy EMA) plus `--no-ema`/`--ema-power`/etc CLI flags and checkpoint plumbing — a direct response to the diagnosis doc's open question ("was EMA active on the failing checkpoint? undetermined"), closing that gap for all future checkpoints going forward. Applies uniformly regardless of `action_type`; bundled into this branch but orthogonal to the delta-flow representation work.
- `tests/unit/anvil_trainer/test_umi_features.py` (new, 642 lines): tests for the EMA/DDPM-IP/DDIM-default bundle above, not the ee_delta mechanism.
- `scripts/training_metrics.sh` (new): a generic ACT/Diffusion/Pi0.5 training-speed benchmark script, unrelated to ee_delta; its own internal usage banner still calls itself `benchmark_training.sh` (a leftover rename mismatch).

---

## C. Inference pipeline (ROS2)

### `ros2/src/anvil_msgs/` (new package)
`CommandedEEPose.msg`: `std_msgs/Header header; geometry_msgs/Pose pose; float64 gripper` — one message type used for both the outbound command (`/commanded_ee_<arm>`) and the inbound observation (`/ee_pose_<arm>`); direction is purely topic-name convention. `CMakeLists.txt`/`package.xml` are standard `rosidl_generate_interfaces` boilerplate.

### `ros2/src/lerobot_control/lerobot_control/ee_runtime.py` (new file, 224 lines)
Role: the only place inference-side EE math primitives are wrapped for ROS consumption — a thin adapter over `anvil_shared.ee_transform`, kept separate from `inference_node.py` so the numeric logic stays importable/testable without `rclpy`. Effectively replaces the deleted `delta_restore.py` as the "runtime utilities" module, though for a different feature (EE math, not the old joint-space delta mechanism `delta_restore.py` used to hold).

- `_ensure_anvil_shared()` (39): lazily inserts `packages/anvil_shared/src` onto `sys.path`, called at the top of every public function here (not at import time) so import overhead is paid only when actually used.
- `resolve_action_type(cfg)` (54): normalizes `cfg.get("action_type","joint_abs")` through `anvil_shared.action_types.normalize_action_type` — the single chokepoint on the ROS2 side for the legacy `"ee_rel"` alias.
- `read_checkpoint_anvil_config(model_path)` (69): resolves bare/`pretrained_model/`/HF-cache checkpoint layouts and reads `anvil_config.json` — **duplicates** (acknowledged in its own docstring, line 72: "mirrors the path resolution in `inference_node._read_checkpoint_metadata`") the same logic in `inference_node.py`, rather than sharing it; used only by `inference_monitor_node.py`.
- `ee_relative_restore_chunk(chunk_np, obs_t)` (103): thin wrapper on `ee_relative_inverse` — **body-frame** composition (`R_abs = R_state @ rot6ds_to_matrices(delta_rot6d)`), unchanged math for the existing n-0 mechanism. Accepts 1-D or 2-D `obs_t` (2-D → last row used).
- **`ee_delta_restore_step(delta, obs_t)` (147, new):** docstring explicitly contrasts with the chunk-restore function above: "composes a single delta against the FRESHEST observed pose... intended to be called once per publish tick, every tick" (154-159). Per-arm: `abs_xyz = obs_xyz + delta_xyz`, `R_abs = R_delta @ R_state` — **world-frame/extrinsic**, opposite composition order from the chunk-restore function; letter-for-letter the plan's resolved formula. Accepts and returns 1-D (single step) — the shape actually used by `_publish_loop`'s per-tick call site.
- `ee_poses_from_chunk(chunk_np, n_arms=None)` (197): thin wrapper on `ee_action_to_poses`, used for **every** EE mode (ee_abs/ee_relative/ee_delta) since by the time it's called the action is already absolute.
- **Design gap, not enforced anywhere:** nothing in this module's types distinguishes "chunk" from "single-step" — `ee_delta_restore_step` could be called with a batch and it would compose every row against the *same* `obs_t`, silently reproducing n-0-style staleness and defeating the "fresh every tick" design if misused. The convention is caller discipline only.

### `ros2/src/lerobot_control/lerobot_control/strategies/multi_process.py`
Role: builds the per-tick `observation` dict; owns the EE-vs-joint subscription decision — **load-bearing for EE mode, not boilerplate** (this file was initially scoped as "check only" during this document's research pass but turned out central enough to warrant full treatment).
- `setup(...)`: `ee_arms = {name: ac for name, ac in arms_config.items() if "ee_command_topic" in ac}` branches to `_setup_ee_subscriptions(ee_arms)` (new method) or the pre-existing `_setup_joint_subscription`. **This is a second, independent EE-mode detector** from `inference_node.py`'s checkpoint-based `resolve_action_type` — the two must agree by convention (a correctly-authored EE YAML always pairs `ee_command_topic` with an EE checkpoint) but nothing in code enforces it; a mismatch would silently misbehave rather than fail loudly (`inference_ee.yaml`'s header comment flags this as a known trap, but only as documentation).
- `_setup_ee_subscriptions(ee_arms)` (new): subscribes `CommandedEEPose` per arm with RELIABLE/KEEP_LAST(10) QoS; uses a `_make_cb(name)` closure factory to correctly bind `name` per-arm in the loop (avoiding the classic Python late-binding-closure bug).
- `get_observation(...)`: EE readiness check is `if not self._ee_state_by_arm: return None` — only checks that **at least one** arm has published at least once, not that *every* configured arm has. In a bimanual setup with one arm's publisher down, `_build_observation` would silently fill that arm's slot with `[0.0]*8` forever — a physically nonsensical zero pose — rather than surfacing the missing publisher.
- `_build_observation(images)`: EE branch concatenates `self._ee_state_by_arm.get(arm_name, [0.0]*8)` per arm and returns immediately — a clean either/or split from the joint-mode observation-building code, no shared path.

### `ros2/src/lerobot_control/lerobot_control/inference_node.py`
Role: the inference orchestrator. See Level 2 for the full per-tick trace; functions not already covered there:
- `_setup_config()` (176): sets `is_ee`/`is_ee_relative`/`is_ee_abs`/`is_ee_delta` (252-259) and `ee_abs_uses_rot6d_obs` (263-265, a data-driven `obs_state_dim % 10 == 0` heuristic distinguishing old quat-obs `ee_abs` checkpoints from new rot6d-obs ones, rather than a stored flag).
- `_read_checkpoint_metadata()` (291): duplicates `ee_runtime.read_checkpoint_anvil_config`'s path-resolution logic but also reads `config.json` (image shape, model type) — can't trivially delegate.
- `_obs_update()`/`_publish_loop()` (652/847): see Level 2. Notable inline decision: the `_will_run_forward` heuristic (740-748) peeks at the model's internal action-queue length to decide whether this tick will invoke the model — used only for latency bookkeeping, introspecting two different lerobot-internal attribute names (`_action_queue` for ACT, `_queues["action"]` for Diffusion) since lerobot's classes aren't unified here.
- `_publish_ee_action(action)` (1038): shared by ee_abs/ee_relative/ee_delta alike (action is always absolute by the time this runs) — builds `CommandedEEPose`, applies gripper shift/scale/clamp (1086-1089).
- `_publish_hold_position()` (1312): explicitly skipped for all EE modes (1321-1323) — the anvil-workcell controller retains the last commanded pose autonomously; sending a zero `Float64MultiArray` would be misinterpreted as a joint command.
- **Confirmed likely bug — `self._obs_lock`:** created only inside `_setup_vla_inference()` (551), called only `if self._is_vla` (411-413). But `_obs_update` (737) and `_publish_loop` (880) both do `with self._obs_lock:` unconditionally whenever `self.is_ee_delta` — and `ee_delta`'s designated target architecture per the plan is **Diffusion**, which is never `_is_vla`. This would raise `AttributeError: 'LeRobotInferenceNode' object has no attribute '_obs_lock'` on the very first `_obs_update` tick for any classic (non-VLA) `ee_delta` model. **Not verified by actually running an ee_delta checkpoint through this node during this research pass — flagged as the single highest-priority thing to check before trusting the decoupled publish loop works at all.**

### `ros2/src/lerobot_control/lerobot_control/test/fake_hardware/fake_hardware_node.py` (Item 2b)
Role: standalone integration-test double for the full ROS2 topic surface. Single-threaded (`rclpy.spin`, no executor concurrency) — no locking needed anywhere even though `self._ee_state` is written from a subscription callback and read from a timer callback, since both run on the same thread.
- `_setup_ee_mode()` (187): seeds `self._ee_state` per arm with an arbitrary starting pose (explicit comment: "not physically meaningful; this is a software timing/plumbing smoke test, not a dynamics simulator").
- `publish_ee_poses()` (221): timer at `ee_pose_fps` (default 100 Hz, independent of the real node's `control_freq`) — publishes the current state as-is.
- `_ee_command_callback(arm, msg)` (241): validates all 8 values finite (raises `SystemExit(1)` otherwise, same fail-hard contract as joint mode); **overwrites** `self._ee_state[arm]` wholesale (264-266) — a literal instantaneous echo, not a weighted blend or physically-integrated step, explicitly documented as out-of-scope-for-realism.
- Joint-mode code is unchanged; the "joint topics are not started in EE mode" claim is enforced structurally by an `if self._ee_mode:` branch in `__init__`, not a per-topic runtime check.
- **Worth restating plainly:** because the echo is a full state overwrite with zero simulated latency, this mock cannot exercise "what if the real arm hasn't caught up to the last command yet" — a real limitation given real hardware *will* have exactly this lag (the plan's own Item 6 concern about "bounded trailing lag"). Documented as a scope boundary, not a bug — but a reader should know the mock validates the self-correction loop "only in the zero-latency limit."

### `ros2/src/lerobot_control/lerobot_control/inference_monitor_node.py`
- `__init__` (43): resolves `action_type` via `read_checkpoint_anvil_config` → `resolve_action_type` (self-detect from the checkpoint, robust to whether the launcher correctly threaded an env var through) or falls back to the ROS param — both paths funnel through `resolve_action_type`, so the CSV's `# action_type:` header is always canonical. **No `ee_delta`-specific handling exists anywhere in this file** (not a crash risk, just an untested gap — see Deviations).

### `ros2/src/lerobot_control/lerobot_control/eval_recorder_node.py`
- `self._is_ee = self._action_type in ("ee_abs","ee_relative","ee_rel")` — checks the raw ROS param string directly against a 3-tuple **instead of calling `resolve_action_type`**, the only place in the ROS2 tree that re-implements the alias check rather than using the shared chokepoint. **`ee_delta` is not in this tuple** — an ee_delta checkpoint run through this node would silently fall into the joint-mode branch, subscribing `Float64MultiArray` on the wrong topics, recording nothing meaningful, without erroring.
- `_on_gt_ee`/`_on_pred_ee` (new): `CommandedEEPose` → flat `[x,y,z,qx,qy,qz,qw,gripper]` list.
- `_compute_raw_ground_truth` (the old joint-space delta GT reconstruction) is **deleted entirely** — part of a broader removal described below.

### `ros2/src/lerobot_control/lerobot_control/action_limiter.py`
Joint-space-only safety limiter (fully skipped for any EE mode). Changes here are purely removal of the old `delta_exclude_joints` deadband logic — no EE-specific logic was ever added, since EE mode bypasses this class entirely.

### Configs, scripts, docs (supporting)
- `configs/lerobot_control/inference_ee.yaml` (new): documents both EE-detection mechanisms directly in its header comment; per-arm `gripper_factor`/`gripper_min`/`gripper_max` tunables.
- `docker-compose.fake-hardware.yml`: adds `EE_MODE`/`EE_ARMS`/`EE_POSE_FPS` env vars threaded to the mock's ROS params.
- `tests/smoke/fixtures/configs/inference-eval-smoke-test-ee.yaml` (new): explicitly scopes itself to exercising "the `ee_rel` legacy alias for `action_type=ee_relative`" — **no `ee_delta` scenario exists in this fixture** (consistent with the `eval_recorder_node.py` gap: ee_delta's ROS-side publish loop has zero smoke-test coverage in this suite, only unit-tested math).
- `docs/inference.md`: adds an "EE-mode fake hardware" section for the `EE_MODE=true` workflow.

### Beyond-plan scope creep in this workflow, not called out in the plan
Wholesale removal of the old joint-space `delta_obs_t`/`delta_sequential` action-type family from the ROS2 side: `delta_restore.py` deleted, `eval_recorder_node.py`'s `_compute_raw_ground_truth` deleted, `action_limiter.py`'s `delta_exclude_joints` removed, `docker-compose.eval.yml`'s `EVAL_USE_DELTA_ACTIONS`/`EVAL_DELTA_EXCLUDE_JOINTS` env vars removed, `plot_monitor_csv.py`'s legacy-CSV compat branch removed. This is a real, substantial feature removal the plan never mentions deciding to do — it reads as bundled housekeeping (the plan's "delta collision" concern was about `_delta_ref_state`, but `delta_obs_t`/`delta_sequential` was a third, apparently-unused, entirely separate joint-space mechanism swept away in the same pass).

### Genuinely unrelated to this branch's EE-delta work (verified, one line each)
`image_worker.py` (full-episode JPEG recording feature), `mcap_player_node.py` (dead-code deletion, no EE connection), `docker/inference/Dockerfile`/`entrypoint.sh` (video-recording feature plumbing + DDS env hygiene).

---

## D. Offline validation

### `packages/anvil_eval/src/anvil_eval/gt_replay.py` (new, untracked)
Role: standalone, model-free CLI gate — the plan's Item 3, "first gate, strict bar." Lazily imports `anvil_shared.ee_transform`/`anvil_shared.rotation`/`pandas`/`yaml`/`json` inside functions, not at module top (keeps module import cheap, avoids a hard pandas dependency at CLI-help time).

- `_pos_slices`/`_rot_slices` (70, 75): per-arm `(start,end)` slices derived from a **hardcoded** `_ACTION_DIM_PER_ARM = 10` (54) — no validation that `action_dim % 10 == 0`; a wrong-shaped dataset would silently misslice via integer floor division.
- `_rot6d_angle_diff_deg(r6d_a, r6d_b)` (80): rot6d→matrix→`Rdiff = Ra @ Rb.T`→quat→`angle = 2*arccos(clip(|qw|,0,1))` — using `|qw|` is a deliberate quaternion double-cover fix, matching the immunity argument in the diagnosis doc §1.3.
- `_max_pos_rot_err` (92): reduces an episode to **max** (not mean) position/rotation error across all arms/frames — appropriate for a strict gate, where one bad frame must fail even if the average is fine.
- `_detect_encoding(dataset_path)` (107): reads `conversion_config.yaml` directly — a second, ad-hoc read of a key `mcap_converter` itself writes, not routed through mcap_converter's own config schema/loader.
- `_load_episode_arrays(dataset_root, episode_idx)` (125): reads parquet directly via pandas, explicitly bypassing `LeRobotDataset`/training transforms ("to keep this tool independent of any training-time transform," docstring 127-130) — a deliberate choice, but it duplicates dataset-loading logic `EvaluationDataset` already implements for the model-based workflow, with no shared code between them; a bug fix in one's episode-boundary logic wouldn't propagate to the other.
- `_replay_episode_absolute` (155): anchor=`states[:-1]`, gt=`actions[1:]`; `delta = ee_delta_forward(gt, anchor)`; `recon = ee_delta_inverse(delta, anchor)`; error = recon vs gt. Requires `T>=2`.
- `_replay_episode_delta` (171): checks the first-frame self-anchor invariant against actual converter output (184-188, the **only** place this convention is tested end-to-end against real conversion, not just the transform function in isolation); for `T>=2`, reconstructs `action_abs = ee_obs_abs_forward(states)`, computes `expected = ee_delta_forward(action_abs[1:], states[:-1])`, diffs against the real on-disk `actions[1:]` — the "is the baked column correct" check, not a round-trip.
- `main()` (239): orchestrates; `if n_total == 0: ... sys.exit(1)` (323-325) — a dataset where every episode is skipped counts as a hard FAIL, not a silent no-op, a deliberate strict-gate choice.
- **Rough edge — unit mismatch:** the first-frame tolerance (`main()`, line 296) uses `args.atol_rot_deg * 1e-2` as the threshold for a unitless rot6d L2 error — multiplying a degrees-tolerance by `1e-2` to get a unitless-L2 tolerance is a numeric convenience, not a principled derivation (the comment at 296-298 acknowledges this). Would be clearer as its own named constant.
- **Rough edge:** no JSON/CSV output at all — PASS/FAIL is stdout logging plus process exit code, inconsistent with `anvil-eval`'s own convention of writing a structured summary.

### `tests/unit/anvil_eval/test_gt_replay.py` (new, untracked)
Fully synthetic data, no real mcap/parquet fixtures — so these tests validate the *replay logic's* self-consistency, not an integration bug in `mcap_converter`'s actual `_align_ee_signals` output; no end-to-end fixture-based test exists bridging the two. `TestReplayEpisodeAbsolute::test_round_trip_exactness_is_anchor_content_independent` (76) explicitly documents, in its own docstring, that "absolute" mode passes even on a corrupted action array (no external reference) — "by design, not a gate weakness," deferring corruption detection to "delta" mode's `test_corrupted_baked_column_fails` (124).

### `packages/anvil_eval/src/anvil_eval/cli.py`
`main()` (91): `if evaluator.is_ee:` (194, 229) gates `ee_rot6d_to_quat_layout` conversion before calling `compute_episode_metrics` — unreachable for `ee_delta` since `evaluator.is_ee` never includes it.

### `packages/anvil_eval/src/anvil_eval/evaluator.py`
- `EpisodeEvaluator.__init__` (62): `self.action_type = normalize_action_type(...)` (84); `is_ee`/`is_ee_relative`/`is_ee_abs` flags (85-87) drive every branch in `evaluate_episode` — **no `is_ee_delta` flag exists**.
- `evaluate_episode` (93): `_relative_anchor_state` (91, reset per-episode at 113) — this is the **renamed pre-existing n-0 mechanism**, reused here for the evaluator's own chunk-restore logic, a separate concern from mcap_converter's baked delta; don't confuse the two when reading this file. `_abs_shadow_queue` (115) deliberately shadows the model's own action queue rather than mutating it in place. `_is_new_chunk` detection (142-146) relies on lerobot's private `_queues` attribute name — fragile, no adapter layer. Obs conversion happens before `self.preprocessor` runs (160-181) because "dataset stores 8-dim quat obs; checkpoint normaliser stats are 10-dim rot6d" — hand-reimplementing, in miniature, what `EEAbsTransform`/`EERelativeTransform` do at train time; exactly the kind of obs-conversion an `ee_delta`-aware branch would also need but doesn't have.

### `packages/anvil_eval/src/anvil_eval/metrics.py`
- `EEMetrics` (12): `position_pass`/`orientation_pass` hardcode `0.02` m / `0.0873` rad directly in the class — the **same** thresholds are independently re-declared in `compute_summary_metrics` (282) and in `reporting.py`'s `_compute_aggregate` (82, in a *third* unit convention — `5.0` degrees) — three copies of the same magic numbers, currently consistent but a drift risk on any future tuning.
- `compute_ee_metrics` (65): expects 8-dim quaternion layout (not the 10-dim rot6d action layout `gt_replay.py` works in) — the "metric space" the CLI explicitly converts into first. Orientation error uses a matrix-trace geodesic formula (`arccos((trace(rel)-1)/2)`) — mathematically equivalent to, but independently implemented from, `gt_replay.py`'s quaternion-based `2*arccos(|qw|)` formula.
- `compute_episode_metrics` (136): the `action_type in ("ee_abs","ee_relative","ee_rel")` gate (198) **omits `ee_delta`** — the Item-5 gap. When EE mode does trigger, all generic scalar metrics are deliberately set to NaN/empty (204-217) rather than computed-and-ignored, specifically to avoid a misleading unit-mixing MAE — which makes it worse, not better, that `ee_delta` falls through to the generic path instead: it produces a number that *looks* like a valid metric but isn't.

### `packages/anvil_eval/src/anvil_eval/plotting.py`
- `plot_episode_joints` (20): `show_delta = action_type in ("ee_relative","ee_rel") and ...` (48) — no `ee_delta` branch, so an ee_delta run gets no diagnostic bottom-block plot even though the analogous baked-delta-vs-delta-GT comparison would be equally informative.
- `plot_monitor_signals` (115): same gate pattern (140, 148); `_EE_DIM_NAMES`/`_EE_DIM_UNITS` (146-147) hardcode the 10-dims-per-arm layout — a third independent hardcoding of that layout convention alongside `gt_replay.py`'s `_ACTION_DIM_PER_ARM` and `anvil_eval_ros/cli.py`'s `_EE_DIMS_PER_ARM`, none sharing a constant from `anvil_shared`.

### `packages/anvil_eval/src/anvil_eval/reporting.py`
`_compute_aggregate` (53): re-declares the PASS/FAIL thresholds a third/fourth time (82, in degrees not radians); since `ee` is never populated for `ee_delta` (per the gap above), this silently produces no `"ee"` key in an ee_delta run's summary JSON — a plain generic-metrics report with no visible error, making the gap easy to miss without inspecting output closely.

### `packages/anvil_eval_ros/src/anvil_eval_ros/cli.py`
Rewritten this branch to branch EE vs joint for MCAP-replay eval-config generation. `_detect_arms_from_conversion_config`: EE branch reads arm names from `observation_topics` keys (EE configs use that instead of `action_topics`). `_EE_DIMS_PER_ARM = 10` (new module constant — the fourth independent hardcoding of this layout convention). `generate_inference_config`: `is_ee = action_type in ("ee_abs","ee_relative","ee_rel")` (raw, unnormalized) — **omits `ee_delta`**; an ee_delta checkpoint takes the joint branch here, likely producing a wrong-shaped generated config. Fallback EE arm order hardcodes `["right"]` only — under-tested for a hypothetical bimanual-EE-without-conversion-config case. `main()`: the old `use_delta_actions`/`delta_exclude_joints` env-var surface (`EVAL_USE_DELTA_ACTIONS`, `EVAL_DELTA_EXCLUDE_JOINTS`) is removed entirely, replaced by a single `action_type` string read from `anvil_config.json` — a real breaking simplification of the eval-ros env-var contract.

### `docs/relative_ee_failure_analysis.md`
This is the **original** root-cause diagnosis (predates this branch), concluding the primary cause was an obs/action anchor mismatch (H1) with mixed world/body-frame representation as a secondary factor (H2) — a **different ranking** from the later, more rigorous `claude_docs/ee-space-libero-vs-production-diagnosis.md`, which puts n-0 stats-pooling/normalization-range compression as primary and downgrades anchor-mismatch to "(c) not comparable." This branch only added a two-line header note clarifying it's a historical record under the old `ee_rel` name — the actual conclusions were **not reconciled** with the newer diagnosis, so a reader relying on this file alone would get an outdated causal story.

---

# Terminology rename — consolidated audit

**Overall status: essentially complete, with a permanent legacy alias, and one small inconsistency.**

Done and verified by repo-wide grep (zero remaining old-name call sites):
- `ee_rel_forward`/`ee_rel_inverse`/`ee_obs_rel_forward` → `ee_relative_forward`/`ee_relative_inverse`/`ee_obs_relative_forward` (`anvil_shared/ee_transform.py`).
- `EERelTransform` → `EERelativeTransform`; `_compute_ee_rel_stats` → `_compute_ee_relative_stats`; `is_ee_rel` → `is_ee_relative` (`anvil_trainer`).
- `_delta_ref_state` → `_relative_anchor_state`; `ee_rel_restore_chunk` → `ee_relative_restore_chunk` (ROS2 `inference_node.py`/`ee_runtime.py`).
- Public `action_type` token: `"ee_relative"` is canonical; `"ee_rel"` is accepted everywhere as a **permanent** alias via `anvil_shared.action_types.ACTION_TYPE_ALIASES`/`normalize_action_type`, exercised by tests loading a legacy string and asserting the canonical result — this satisfies the plan's "must keep reading the legacy token forever" requirement for the 31 existing on-disk checkpoints.
- `gt_replay.py` and `evaluator.py` use only the new vocabulary throughout, correctly, since `gt_replay.py` is purpose-built for the new mechanism and `evaluator.py` was fully migrated.

The one inconsistency: `eval_recorder_node.py` re-implements the alias check inline (`action_type in ("ee_abs","ee_relative","ee_rel")`) instead of calling `resolve_action_type`/`normalize_action_type` — functionally correct today, but it's the one place that also happens to omit `ee_delta` (see below), which is not a coincidence: a single un-DRY reimplementation is exactly where an omission like this slips through.

A related but distinct concern: `patches.py`'s `_compute_ee_relative_stats` log line says `n_offset_steps=%d` (the plan's proposed rename target) but the underlying variable is still named `action_delta_indices` — the rename reached the log wording, not the code identifier, in that one spot.

---

# Deviations from `claude_docs/ee-delta-flow-plan.md`

1. **mcap_converter's per-arm branching structure** (`extractor.py:1357-1424`) differs from the plan's literal phrasing ("only the `action_slices.append(...)` computation branches to the delta formula"): the actual delta call happens once, after the per-arm loop, on the fully-concatenated array — functionally equivalent (since `ee_delta_forward` loops per-arm internally) but structurally different from what the plan describes, if you're line-referencing the plan while reading the diff.
2. **No shipped config ever sets `ee_action_encoding: "delta"`.** The plan's Item 4 calls for converting a real session with the new flag; empirically, zero YAML files in the repo (shipped configs or test fixtures) set it — the delta path is exercised only from Python unit tests constructing `DataConfig` directly. **Item 4 (the actual convert+train step) has not been run/wired up on this branch.** This is the single most concrete "what's actually left to do" signal in the whole codebase.
3. **`docs/data-conversion.md` and `docs/training.md` do not document the new mechanism.** The mcap_converter docs have zero mention of `ee_action_encoding`/delta baking; `docs/training.md`'s action-type table lists `joint_abs`/`ee_abs`/`ee_relative` but omits `ee_delta` entirely — even though `train.py`'s own `_ANVIL_HELP` text fully documents it. An incomplete pass, not a deliberate cut (nothing marks it as pending).
4. **GT-replay (Item 3) does not use `EvaluationDataset`**, despite the plan's explicit "reuse `anvil_shared.ee_transform` + `anvil_eval.dataset.EvaluationDataset`" instruction — it reads parquet/`info.json` directly by hand instead, a deliberate choice (documented reasoning: independence from training-time transforms) but a real deviation that leaves dataset-loading logic duplicated in two places.
5. **Item 5 (offline evaluation, "downgraded bar") is not implemented for `ee_delta` at all** — the single most consequential deviation found. `evaluator.py`, `metrics.py`, `plotting.py`, and `anvil_eval_ros/cli.py` all check EE-mode via tuples that omit `"ee_delta"`. The plan's bar was "runs and produces output, FAIL is expected and fine" — what actually happens is worse than a FAIL: an `ee_delta` checkpoint silently takes the **generic, non-EE metrics path**, producing a numerically meaningless MAE/MSE that mixes metres, unitless rot6d, and gripper units, with no warning anywhere that this happened. Nothing in the code acknowledges this as a known gap.
6. **Beyond-plan scope creep, undocumented as a decision:** the wholesale removal of the old joint-space `delta_obs_t`/`delta_sequential` mechanism (`delta_restore.py` deletion, `eval_recorder_node.py`'s GT-reconstruction deletion, `action_limiter.py`/`docker-compose.eval.yml`/`plot_monitor_csv.py` cleanup) — real, substantial, and never mentioned as a decision anywhere in the plan.
7. **Matches the plan exactly, confirmed by direct read** (called out explicitly since the user asked not to silently normalize *either* direction): the world-frame rotation-math formulas in `ee_delta_forward`/`ee_delta_inverse`/`ee_delta_restore_step`; the first-frame self-anchor convention end-to-end (converter → training stats → GT-replay); the epsilon-floor and rot6d-identity-clamp stats guards; the explicit COMPLETED/FAILED stats logging; Item 2b's fake-hardware echo/integrate behavior; and GT-replay's machine-precision strictness with no smoke-scale relaxation.

---

# Known bugs, gaps, and rough edges (consolidated, roughly severity-ordered)

**Likely crash / correctness bugs:**
1. **`self._obs_lock` is never initialized for classic (non-VLA) models** (`inference_node.py`) — created only inside `_setup_vla_inference()`, called only when `self._is_vla`. But `_obs_update`/`_publish_loop` do `with self._obs_lock:` unconditionally whenever `self.is_ee_delta` — and Diffusion (ee_delta's target architecture) is never `_is_vla`. Almost certainly an `AttributeError` on the first `_obs_update` tick for any real ee_delta checkpoint. **Not verified by actually running one — this is the top item to check before trusting any of this works end-to-end.**
2. **`ee_delta` is silently misrouted or dropped in at least five places**, each independently: `config.py:343` (training output-dir routing → lands in `model_zoo/joint-space/` instead of EE-space), `config.py:543` (`validate_action_space` — no dataset-shape validation at all), `eval_recorder_node.py` (falls into joint-mode topic subscription), `evaluator.py`/`metrics.py`/`plotting.py` (falls into meaningless generic metrics), `anvil_eval_ros/cli.py` (falls into joint-mode eval-config generation). None of these five raise an error or log a warning — they all silently do the wrong thing.
3. **No shipped config exercises the delta-baking path at all** — Item 4 was never actually run through the real CLI; only unit-tested via direct `DataConfig` construction.

**Real but lower-severity gaps:**
4. `anvil_config.json` omits `is_ee_delta` (checkpoint metadata is inconsistent with the analogous `is_ee_relative` key).
5. Two independent "valid action types" registries that can drift: `anvil_shared.action_types.VALID_ACTION_TYPES` (no `ee_delta`) vs `anvil_trainer.config._VALID_ACTION_TYPES` (has it) — the former is documented as the shared source of truth but isn't actually imported for that purpose by the latter.
6. `anvil_shared/__init__.py`'s `__all__` lists `ee_obs_abs_forward` without importing it — `AttributeError` if ever imported via the package root (currently latent, zero real call sites hit it).
7. Two independent, unsynchronized EE-mode detectors on the inference side (`inference_node.py`'s checkpoint-based `resolve_action_type` vs `multi_process.py`'s config-based `ee_command_topic` presence check) — a mismatched checkpoint/config pairing fails silently, not loudly.
8. `multi_process.py`'s EE readiness check only requires *one* arm to have reported, not all configured arms — a down publisher in a bimanual setup silently defaults that arm's observation to a zero vector forever.
9. 4 stale/broken legacy `configs/mcap_converter/*.yaml` files (empirically verified to raise `ValueError` or silently produce empty topics under the current loader) — unrelated to EE-delta itself, but real accumulated cruft from the same branch's broader unified-config rewrite.
10. `gt_replay.py`'s first-frame tolerance mixes units (a degrees-based CLI flag scaled by `1e-2` to threshold a unitless L2 quantity).
11. `gt_replay.py` produces no structured (JSON/CSV) output, unlike every other tool in the package — PASS/FAIL is stdout + exit code only.

**Duplication / DRY violations (correctness-neutral today, drift risk tomorrow):**
12. The "10 dims per arm, `[x,y,z,r0..r5,grip]`" action-layout constant is hardcoded independently in at least four places (`gt_replay.py`, `plotting.py`, `anvil_eval_ros/cli.py`, implicitly in `metrics.py`) with none importing a shared constant from `anvil_shared`.
13. The geodesic rotation-error formula is implemented two different ways in two files (`gt_replay.py`'s quaternion-based `2*arccos(|qw|)` vs `metrics.py`'s matrix-trace-based `arccos((trace-1)/2)`) — mathematically equivalent, never shared.
14. PASS/FAIL thresholds (`0.02` m / `0.0873` rad ≈ 5°) are copy-pasted 3-4 times across `metrics.py` (twice) and `reporting.py` (in a third unit convention).
15. Checkpoint-path resolution logic is duplicated between `inference_node.py` and `ee_runtime.py` (acknowledged in the latter's own docstring as a deliberate-but-unshared mirror).
16. `ee_action_encoding` validation is duplicated between `loader.py` (raises `ValueError`) and `validators.py` (raises `ConfigurationError`) — the latter is dead code in the actual call chain today.
17. Three copies of near-identical `setup_logging()` boilerplate across `gt_replay.py`/`anvil_eval/cli.py`/`anvil_eval_ros/cli.py`.
18. Redundant stats mutation in `patches.py`: `_compute_ee_delta_stats` mutates `full_dataset.meta.stats` in place (lines 512, 534), but the object that actually matters for training is `train_dataset` (a different, filtered instance) — the caller's later mutation (patches.py:826-828) is what's load-bearing; the first mutation is dead-effect work that reads as load-bearing but isn't.

**Naming / API-shape nits (cosmetic, but genuinely confusing on a cold read):**
19. `_classic_action_deque` now holds semantically different payloads depending on mode — raw deltas for `ee_delta`, already-restored absolutes for everything else — with nothing in the name or type signaling which.
20. `ee_relative_restore_chunk`/`ee_delta_restore_step` are conceptually parallel ("restore an encoding to absolute given a reference") but share no base class/interface, and nothing prevents calling the single-step function with a batch composed against one shared reference (silently reproducing n-0-style staleness if misused).
21. `EEAbsTransform`/`EEDeltaTransform` are not exported at `anvil_trainer` package level, unlike the structurally-parallel `EERelativeTransform`.
22. `rotation.py`'s batch `rot6ds_to_matrices` clamps degenerate input while the scalar `rot6d_to_matrix` raises on the same condition, with a docstring rationale that reads backwards (clamping is what typically *masks* a bug, not what avoids masking one) — predates this branch but is directly relevant to auditing `ee_delta_*`'s numerical behavior.
23. `scripts/training_metrics.sh`'s internal usage banner still calls itself `benchmark_training.sh` — a leftover rename mismatch (this script is unrelated to EE-delta regardless).

---

# Appendix — files touched on this branch that are NOT part of the seven EE-delta items

Verified during research and excluded from the deep-dive above; listed here so a full-branch audit doesn't waste time re-investigating them as EE-delta work:

- `ema.py`, `test_umi_features.py`, `scripts/training_metrics.sh` — EMA/DDPM-IP/DDIM-default training-quality bundle, a direct response to the diagnosis doc's open EMA question, orthogonal to the delta representation.
- `image_worker.py`, the ffmpeg-batch-convert additions in `run_inference.sh`, `docker/inference/Dockerfile`/`entrypoint.sh` — full-episode JPEG/video recording feature for the inference monitor.
- `mcap_player_node.py` — dead-code deletion, no EE connection.
- `packages/mcap_converter/src/mcap_converter/cli/inspect.py` — standalone MCAP topic/schema inspection CLI, no EE-transform reference.
- `configs/mcap_converter/openarm_bimanual_quest.yaml`, `openarm_bimanual.yaml`, `openarm_single_quest.yaml`, `openarm_single_quest_afo.yaml` — stale legacy-format configs, empirically broken under the current loader; unrelated cruft from the branch's broader unified-config rewrite.
- `configs/cyclonedds/*` renames (`two_pc_gpu.xml`→`gpu_pc.xml` etc.) and the `docs/inference.md` "Distributed Inference Architecture" section rewrite — unrelated docs/infra cleanup that happens to touch some of the same files.
- The large block of `D` (deleted-relative-to-main) files under `packages/mcap_converter/.../{dataset_viz.py, mcap_valid.py, quality.py, schema_inspect.py, viz/}` and their tests — this branch's working tree simply predates work that exists on `main`; not something this branch deleted as part of EE-delta work (branch-divergence, not a design decision).
