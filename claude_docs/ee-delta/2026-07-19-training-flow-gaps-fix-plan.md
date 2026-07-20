# Fix Plan — ee_delta Training Unblock (Phase A) + Config-Loader Unification (Phase B)

Date: 2026-07-19 (revised; supersedes the 2026-07-18 version of this file)
Branch: `patrick/implement-ee-space`

## Status: Phase A COMPLETE (2026-07-19), including an unscoped-but-necessary
## extension — see "Phase A+ " below. Phase B superseded — see below.

**2026-07-20: Phase B (below) is superseded by
`claude_docs/config-architecture/2026-07-20-dataset-config-consolidation-plan.md`.**
That doc is the actual approved design (Option C: slim resolved-descriptor
reader in `anvil_shared`, `mcap_converter/config/` untouched) — worked out
with the user after presenting Phase B's sketch alongside two other
alternatives. The Phase B section below is kept only as the historical first
draft that prompted that follow-up design discussion; refer to the new doc
for anything implementable.

Implemented gaps 1–5 as planned, then **discovered and fixed a deeper, related
bug during end-to-end verification**: `anvil_shared/ee_transform.py`'s entire
math library (`n_arms_from_dims`, `ee_relative_forward/inverse`,
`ee_delta_forward/inverse`, `ee_obs_relative_forward`, `ee_obs_abs_forward`)
hardcoded quaternion's 8-dim/arm state layout, silently degrading or
corrupting training for any `rot6d`/`axis_angle`-encoded dataset — including
the exact dataset gap 1 reconverts. User-approved scope expansion: fixed the
whole module + its training-path callers (`anvil_trainer/config.py`,
`patches.py`, `transforms.py`). Eval/inference-side consumers (`anvil_eval`,
ROS `inference_node.py`/`ee_runtime.py`) have the same underlying gap but
require new checkpoint-metadata plumbing (`observation_encoding` isn't
written to `anvil_config.json` yet) — deferred, see the appendix.

**Verification, all green:**
- Full unit suite: 642 passed, 2 skipped (baseline was 572/2; net +70 new
  tests across `test_ee_validation.py` and `test_ee_transform.py`).
- Real end-to-end training on the reconverted bimanual `rot6d` dataset
  (`data/debug/ee-delta-v2/ee-delta/ee-space-testing`) now logs
  `[ee_delta_stats] COMPLETED and INJECTED` (previously
  `FAILED: ... falling back to raw dataset stats`) and
  `[ee_delta] active — 2 arm(s), obs (rot6d) → (10n rot6d, absolute)`
  (previously silently mislabeled `8n quat`). Checkpoint correctly routed to
  `model_zoo/ee-space/ee-space-testing/...` (gap 2 confirmed).
- Smoke test (`--scenario ee_abs,ee_rel,ee_delta --select 1,2`): 6/6 steps
  pass — confirms the quaternion-default path (backward compatibility) is
  unaffected by the encoding-aware rewrite.
- **2026-07-19, later**: added a dedicated `ee_delta_rot6d` smoke scenario
  (`SCENARIOS` in `pipeline_smoke_test.py`, new fixture
  `mcap-converter-smoke-test-ee-delta-rot6d.yaml` with
  `observation_encoding: "rot6d"`) so the smoke suite itself — not just an ad
  hoc manual run — exercises both encodings through the real `mcap-convert` →
  `anvil-trainer` path. `--scenario ee_delta,ee_delta_rot6d --select 1,2`:
  4/4 steps pass. Verified the two converted datasets are genuinely distinct
  on disk (`ee_delta`: `observation.state` shape `[8]`; `ee_delta_rot6d`:
  shape `[10]`, `action` shape `[10]` in both — action is always rot6d
  regardless of `observation_encoding`), and that training logs the correct
  per-scenario behavior: `ee_delta_rot6d` logs
  `[ee_delta] active — 1 arm(s), obs (rot6d) → (10n rot6d, absolute)` and
  `[ee_delta_stats] COMPLETED and INJECTED` (no fallback), matching the
  quaternion scenario's equivalent log lines. Full unit suite re-confirmed
  green (642 passed, 2 skipped) after these smoke-test additions.

## Context

Two intertwined goals, deliberately sequenced:

1. **Unblock ee_delta training now (Phase A).** `ee_delta` (per-frame
   Delta(n→n+1) EE action space, baked on disk by `mcap_converter` with
   `action_encoding: delta`) is code-complete and unit-tested in
   `anvil_trainer` + `mcap_converter`, but a readiness audit found gaps that
   block or degrade the convert→train path. Patrick wants to start testing
   training now, in parallel with a separate agent's GT-replay work.

2. **Unify config-file loading (Phase B, follow-up).** Investigation
   (2026-07-18/19, two exploration passes) established the real picture:
   - Only `mcap_converter` has a real typed, migration-aware YAML loader
     (`config/loader.py`'s `ConfigLoader` + `versioning.py` + `schema.py`).
   - `anvil_eval` / `anvil_trainer` build their config from argv/env + loose
     `json.load`; neither has a YAML file loader. `TrainingConfig.from_yaml`
     (`config.py:482`) exists but is **dead code** (zero callers).
   - The **actual duplication** is hand-rolled lenient `yaml.safe_load`
     readers of a dataset's `conversion_config.yaml`, re-implemented in
     `ros2/.../dataset_reader.py:69` (+ its delegate in
     `dataset_gt_replayer_node.py:159`) and
     `anvil_eval_ros/cli.py:210,316,433`.
   - **Hard constraint:** those readers deliberately avoid importing
     `mcap_converter` because the inference Docker image doesn't ship it —
     only `anvil_shared` is available there (via each node's
     `_ensure_anvil_shared` `sys.path` hack). So a shared loader **must live
     in `anvil_shared`**, and the ROS readers only need a *lenient* read
     (current-schema v1.1 assumed), not the migration machinery.

   Decisions taken (2026-07-18, user-confirmed): **full framework + adapter**
   extraction (not just a thin lenient reader, not a JSON-reads sweep too);
   **unblock training first**, do the loader refactor as a separate
   follow-up. Guiding principle, per user direction: **loaders/shared-contract
   go in `anvil_shared`; each package keeps its own schema.**

`ee_delta` is genuinely distinct from `ee_relative`/`ee_rel` (per-frame
world-frame delta vs. chunk-anchor body-frame relative) — not a rename. Its
**observation** handling mirrors `ee_abs` (quat→rot6d, absolute); only the
**action** column is delta-baked at conversion time.

`anvil_shared` is intentionally dependency-free today (`dependencies = []`,
per its own pyproject docstring: "no ML deps... importable from any context").
Both phases add exactly one lightweight dependency, `pyyaml` — already used by
every consumer, including at ROS inference time.

---

## Phase A — Unblock ee_delta training (execute now)

### A1. `anvil_shared` seeds (small, needed by gap 3, reused whole by Phase B)

These are the non-throwaway foundation gap 3 needs; Phase B builds on top of
them rather than replacing them.

- **Move the EE encoding-layout table** from
  `packages/mcap_converter/src/mcap_converter/config/encodings.py` →
  `packages/anvil_shared/src/anvil_shared/ee_encodings.py` (verbatim:
  `VALID_ACTION_ENCODINGS`, `IMPLEMENTED_ACTION_ENCODINGS`,
  `VALID_OBSERVATION_ENCODINGS`, `OBSERVATION_ROTATION_LAYOUTS`,
  `observation_state_dim_per_arm`, `observation_state_names_per_arm`,
  `encode_rotation`). Rationale: this is the shared on-disk layout contract
  consumed by the trainer (validation), eval, and ROS inference — not
  mcap-converter-private schema. `encode_rotation`'s lazy
  `from anvil_shared.rotation import ...` (previously lazy only to avoid a
  hard cross-package dependency) becomes a plain top-level import now that
  it's in the same package as `rotation.py`.
  - Update the 3 mcap_converter import sites to `anvil_shared.ee_encodings`:
    `config/schema.py:28`, `core/writer.py:339`, `core/extractor.py:1587`.
    Delete `mcap_converter/config/encodings.py` outright (prefer a clean
    delete + fixed imports over a re-export shim).
- **Add `packages/anvil_shared/src/anvil_shared/dataset_config.py`** — a
  lenient reader for a converted dataset's `conversion_config.yaml`,
  generalizing what `ros2/.../dataset_reader.py:58-90` already does:
  - `read_conversion_config(dataset_root) -> dict` (returns `{}` if the file
    is absent; optional logger param mirroring `dataset_reader`'s signature).
  - `resolve_data_space(cfg) -> str` (default `"joint"`),
    `resolve_observation_encoding(cfg) -> str` (default `"quaternion"`),
    `resolve_action_type(cfg) -> str` (joint→`joint_abs`; ee+delta→`ee_delta`;
    ee+absolute→`ee_abs`) — matching `dataset_reader.resolve_action_type`'s
    semantics exactly so Phase B can delete the ROS copy in favor of this.
  - Implemented with `yaml.safe_load`; requires `pyyaml`.
- **`anvil_shared` packaging:** add `"pyyaml~=6.0"` to
  `packages/anvil_shared/pyproject.toml`'s `dependencies`; add the two new
  modules' public names to `anvil_shared/__init__.py`'s import +
  `__all__` block (same pattern as the existing
  `action_types`/`rotation`/`splits` re-exports).

### A2. Gap — data_space routing
`packages/anvil_trainer/src/anvil_trainer/config.py:343`
```python
# before
data_space = "ee" if action_type in ("ee_abs", "ee_relative") else "joint"
# after
data_space = "ee" if action_type in ("ee_abs", "ee_relative", "ee_delta") else "joint"
```
Update the adjacent comment (`:341-342`) to list `ee_delta`. This is a
classmethod-local `action_type` string (no `self`), so keep the existing
literal-tuple pattern rather than reaching for a property.

### A3. Gap — validate_action_space (broadened: fixes a pre-existing bug too)

`packages/anvil_trainer/src/anvil_trainer/config.py:519-574`. On
re-investigation this is more than a missing `ee_delta` entry: the current
check hardcodes **quaternion** as the only possible `observation.state`
encoding, so it silently misclassifies `rot6d`- and `axis_angle`-encoded EE
datasets as joint-space — a pre-existing bug independent of `ee_delta`.
Confirmed against the on-disk
`data/debug/ee-delta/ee-space-testing/conversion_config.yaml` (records
`observation_encoding: rot6d`), which this function would misclassify today.

Per-arm state dims (from the A1 table): quaternion=8, rot6d=10, axis_angle=7
(= 3 xyz + rotation dim + 1 gripper). **Action is always rot6d = 10/arm
regardless of `observation_encoding`** (confirmed via `schema.py:225-231`), so
`n_arms` should be derived from `action_dim // 10`, not from `state_dim`.

Rewrite:
```python
if self.is_ee:   # ee_abs | ee_relative | ee_delta — all need this check
    cfg = read_conversion_config(self.dataset_root)          # A1 shared reader
    obs_encoding = resolve_observation_encoding(cfg)          # default quaternion
    # Prefer the on-disk config's own data_space when available; fall back
    # to the suffix heuristic only for legacy datasets lacking
    # conversion_config.yaml.
    is_ee_dataset = (resolve_data_space(cfg) == "ee") if cfg else (has_ee_state and has_ee_action)
    if not is_ee_dataset:
        raise DataIntegrityError(...)   # unchanged message
    if action_dim % 10 != 0 or action_dim == 0:
        raise DataIntegrityError(
            f"EE dataset has unexpected action dim {action_dim} "
            "(expected positive multiple of 10 — action is always rot6d)."
        )
    n_arms = action_dim // 10
    expected_state_dim = n_arms * observation_state_dim_per_arm(obs_encoding)
    if state_dim != expected_state_dim:
        raise DataIntegrityError(
            f"EE dataset observation.state dim {state_dim} != "
            f"{n_arms} arms * {observation_state_dim_per_arm(obs_encoding)} "
            f"({obs_encoding} per-arm dim) = {expected_state_dim}."
        )
```
Also broaden `_EE_STATE_MARKER_SUFFIXES` (`config.py:85`) to
`{"qx","qy","qz","qw","r0","r1","r2","r3","r4","r5","ax","ay","az"}` so the
**fallback** path (no `conversion_config.yaml`) detects `rot6d`/`axis_angle`
state too, not just quaternion. (`r0..r5` overlaps with
`_EE_ACTION_MARKER_SUFFIXES` — expected, since action is always rot6d too;
the two marker sets check independent feature-name lists.)

### A4. Gap — delete dead VALID_ACTION_TYPES

`packages/anvil_shared/src/anvil_shared/action_types.py:30` is dead
(re-verified 2026-07-18: only its own definition + the re-export in
`__init__.py:4,24`; no importer anywhere, no test). **Delete** the frozenset
and its two `__init__.py` re-export lines — strictly better than adding
`ee_delta` to a set nothing reads, since it removes the second source of
truth entirely (trainer's own `_VALID_ACTION_TYPES` at `config.py:81`, which
already includes `ee_delta`, remains the sole validator). Keep
`normalize_action_type`/`ACTION_TYPE_ALIASES` untouched — they are actively
load-bearing (`config.py`, `evaluator.py`, `ee_runtime.py`) and the module's
own docstring documents the `ee_rel` alias as **permanent**: existing
checkpoints persist `action_type="ee_rel"` and must keep loading forever.

### A5. Gap — smoke-test ee_delta scenario (convert + train only)

1. New fixture
   `tests/smoke/fixtures/configs/mcap-converter-smoke-test-ee-delta.yaml` —
   a copy of `mcap-converter-smoke-test-ee.yaml` with one added line,
   `action_encoding: "delta"` (default is `absolute`).
2. New `SCENARIOS["ee_delta"]` entry in
   `tests/smoke/scripts/pipeline_smoke_test.py` (after the `ee_rel` block,
   ~L207):
   - `action_type="ee_delta"`
   - `convert_config=` the new delta fixture
   - `dataset_dir = OUTPUTS/"datasets"/"ee_delta"/"ee-delta"/_EE_MCAP_NAME`
     — its own dataset dir; delta is baked differently from abs/rel so it
     cannot share `ee_abs`'s converted dataset (extractor appends
     `<data_space>-<encoding>` = `ee-delta` to the output path).
   - `train_out` / `eval_out` / `eval_ros_out` under `.../ee_delta/...`
   - `inference_config=_EE_INFERENCE_CFG` (kept for later, once eval lands)
   - Verified: `_build_train_cmd` already appends `--action-type=ee_delta`
     automatically for any non-`joint_abs` scenario (L337-338) — no
     train-command change needed.
3. Update the module docstring (L4-18) to mention the `ee_delta` scenario and
   that steps 3–4 are gated on the deferred eval-side branch.

Scope note: eval-side `ee_delta` support (`anvil_eval`
evaluator/metrics/plotting/cli, `anvil_eval_ros/cli`) and the GT-replay
validation gate remain out of scope here, so the new scenario is wired for
**steps 1–2 only** (`--select 1,2`) until that lands.

### A6. Gap — reconvert stale dataset (execution, not code)

`data/debug/ee-delta/ee-space-testing` was baked at commit `9611d14`, before
the Delta(n-(n-1)) → Delta(n→n+1) redesign. Raw source data is confirmed
present at `data/raw/ee-space-testing`. Reconvert to a **new** directory
(non-destructive):
```bash
uv run mcap-convert \
  -i data/raw/ee-space-testing \
  -o data/debug/ee-delta-v2 \
  --config configs/mcap_converter/v1.1/openarm_ee_bimanual_debug.yaml
```

---

## Phase A+ — ee_transform.py encoding audit (discovered mid-verification, user-approved)

Running the A6-reconverted dataset (bimanual, `observation_encoding: rot6d`)
through end-to-end training surfaced a real, independent bug:
`patches.py`'s `_compute_ee_delta_stats` logged
`FAILED: EE observation.state dim 20 is not a positive multiple of 8 ...
falling back to raw dataset stats`. Root cause traced to
`anvil_shared/ee_transform.py`: every function touching
`state`/`anchor`/`obs_abs` hardcoded `EE_STATE_DIM_PER_ARM = 8`
(quaternion-only), including the internal rotation-matrix decode
(`state[..., s0+3:s0+7]` assumed to always be a quaternion). Two of these
functions (`ee_obs_relative_forward`, `ee_obs_abs_forward`) had **no
dimension validation at all** — for inputs where `dim // 8` coincidentally
matched the true arm count (e.g. bimanual rot6d: `20 // 8 == 2`), they would
have silently produced wrong output with no error whatsoever, not even the
"FAILED, falling back" warning. Confirmed via a direct regression test
(`test_ee_obs_abs_forward_bimanual_rot6d_without_encoding_now_raises`).

Given this affects `ee_abs`/`ee_relative`/`ee_delta` alike (not just the
`ee_delta` gap originally in scope) and was already silently degrading real
training, user chose "full fix now" over deferring or a narrow patch.

**Changes:**
1. `anvil_shared/ee_transform.py` — every function that reads a
   state/anchor/obs array gained an `observation_encoding: str = "quaternion"`
   keyword (default preserves behavior for existing callers). Per-arm state
   dim and the rotation slice width are now derived from
   `anvil_shared.ee_encodings.observation_state_dim_per_arm`/
   `OBSERVATION_ROTATION_LAYOUTS` instead of the hardcoded constant; the
   rotation-matrix decode dispatches to `quat_to_matrix`/`rot6d_to_matrix`/
   `axis_angle_to_matrix` (and their batched counterparts) based on the
   encoding. `ee_obs_relative_forward`/`ee_obs_abs_forward` gained the
   dimension validation they previously lacked entirely. Action-side-only
   functions (`ee_rot6d_to_quat_layout`, `ee_quat_layout_names`,
   `ee_action_to_poses`) were left untouched — action is always rot6d
   regardless of `observation_encoding`, confirmed via exhaustive caller audit.
2. `anvil_trainer/config.py` — `TrainingConfig` gained an
   `observation_encoding: str = "quaternion"` field, NOT a CLI flag — resolved
   once inside `validate_action_space()` (which already reads
   `conversion_config.yaml` for gap 3) and cached on `self`, so it's available
   by the time `patches.py`/`transforms.py` need it (confirmed via `train.py`'s
   call order: `validate_action_space()` runs immediately after config
   construction, well before `TransformRunner`/patches install).
3. `anvil_trainer/patches.py` — `_compute_ee_relative_stats`,
   `_compute_ee_delta_stats`, `_compute_ee_abs_stats` now pass
   `self.config.observation_encoding` into every `n_arms_from_dims`/
   `ee_relative_forward`/`ee_obs_relative_forward`/`ee_obs_abs_forward` call.
4. `anvil_trainer/transforms.py` — `EEAbsTransform`, `EEDeltaTransform`,
   `EERelativeTransform`'s `apply()` methods pass `config.observation_encoding`
   through similarly. Also fixed `_patch_obs_state_shape_8n_to_10n` (the
   policy-input-shape patch shared by all three transforms), which had the
   identical hardcoded-8 bug — for `axis_angle` (7/arm) datasets it would have
   silently failed to patch the policy's input shape at all, causing a shape
   mismatch against the transform's actual (10/arm) output at train time.
5. Added ~21 new tests to `tests/unit/anvil_shared/test_ee_transform.py`
   (`TestMultiEncodingRoundTrip`): forward/inverse round-trips for all three
   encodings (single-arm and bimanual), `ee_obs_*_forward` identity/dim
   checks, `n_arms_from_dims` cross-encoding rejection, an explicit
   backward-compatibility check (omitting `observation_encoding` behaves
   identically to passing `"quaternion"`), and the direct regression test for
   the exact silent-corruption scenario described above.

**Deliberately NOT done in this pass** (confirmed via exhaustive caller audit
— see the appendix): `anvil_eval/evaluator.py` and ROS
`inference_node.py`/`ee_runtime.py` have the identical underlying gap, but
fixing them requires new plumbing — `observation_encoding` isn't written to
checkpoint `anvil_config.json` at all today, so these consumers have no way
to learn it (ROS `ee_runtime.py`'s bare numpy functions don't even receive a
config object). This is the same eval-side work already deferred earlier in
Phase A's own scope (see appendix) — extending checkpoint metadata plus
teaching eval/inference to read it is a cross-layer task for a dedicated pass,
not a natural extension of "fix ee_transform.py's callers."

---

## Phase B — Config-loader framework unification (follow-up PR, after Phase A)

Goal: one generic config-loading mechanism lives in `anvil_shared`; each
package keeps its own schema. Phase B touches files a concurrent agent is
actively editing (`dataset_reader.py`, `dataset_gt_replayer_node.py`) —
coordinate before starting B3.

### B1. Extract the generic skeleton into `anvil_shared`

Split `mcap_converter/config/` along the generic/schema line identified
during investigation:
- New `anvil_shared/config_migration.py` — the migration **engine** only:
  `FieldMigration`/`VersionMigration` dataclasses, a registry-driven
  `migrate_to_current(cfg, migrations, current_version)` and
  `detected_version(cfg, ...)` dispatch (adding a migration = append one
  registry entry, never touch the dispatcher). Generic, no mcap fields.
- New `anvil_shared/config_loader.py` — the loading **pipeline shell**:
  `load_yaml` / `to_yaml` I/O, `reject_unknown_keys(cfg, recognized_keys)`,
  and a `load(dict_or_path, *, recognized_keys, migrations, current_version,
  hydrate, strict)` that runs migrate → (if strict) unknown-key check →
  `hydrate(dict) -> DataConfig`. Every schema-specific piece is injected by
  the caller.
- Fold the A1 `dataset_config.py` lenient reader in as the
  no-schema/no-migration fast path of the same module (or keep it standalone
  if that reads more clearly — a call during implementation, not now).

### B2. mcap_converter's `ConfigLoader` becomes a thin adapter

`mcap_converter/config/loader.py` keeps `ConfigLoader` as its public API but
delegates to `anvil_shared.config_loader`, injecting what **stays** in
mcap_converter: `DataConfig`, `RECOGNIZED_YAML_KEYS`, `CURRENT_SCHEMA_VERSION`
(`schema.py`), the concrete `MIGRATIONS` registry +
`_migrate_legacy_v1_0_shape` (`versioning.py`), and the `_parse_*` field
hydrators. Behavior must stay byte-identical (see Verification). `schema.py`
and `validators.py` keep their current ownership and responsibilities.

### B3. Switch hand-rolled readers to the shared lenient path

Replace the duplicated `yaml.safe_load` of `conversion_config.yaml` with
`anvil_shared.dataset_config` calls in:
- `ros2/.../dataset_reader.py:58-90` (and drop the "avoids ConfigLoader"
  rationale comment in `dataset_gt_replayer_node.py`, since the reason for
  hand-rolling disappears once the shared reader exists in `anvil_shared`,
  which these nodes already import via `_ensure_anvil_shared`).
- `anvil_eval_ros/cli.py:210,316,433`.

### B4. Leave argv/env configs alone

`anvil_eval`'s `EvalConfig` and `anvil_trainer`'s
`TrainingConfig.from_env_and_args` are argv/env builders, not file loaders —
out of scope for this unification. Optionally delete the dead
`TrainingConfig.from_yaml` (`config.py:482-498`) while touching this file.

---

## Verification

### Phase A
1. **Unit suite stays green** (baseline 572 passed / 2 skipped):
   `uv run python -m pytest tests/unit/anvil_trainer tests/unit/anvil_shared tests/unit/mcap_converter -q`.
   Special attention: mcap_converter's encoding tests still pass after the
   `encodings.py` → `anvil_shared/ee_encodings.py` move (3 import sites
   updated: `schema.py`, `writer.py`, `extractor.py`).
2. **Gap 3 (the real payoff)** — add unit cases covering all three
   `observation_encoding` values (quaternion/rot6d/axis_angle) crossed with
   `ee_abs`/`ee_relative`/`ee_delta`, plus the no-`conversion_config.yaml`
   fallback path; and a regression assertion that the existing rot6d
   `data/debug/ee-delta/ee-space-testing/` is now classified as EE, not
   joint. Assert a joint dataset + `--action-type=ee_delta` still raises
   `DataIntegrityError`.
3. **Gap 2 routing** — confirm an `ee_delta` run's output directory lands
   under `model_zoo/ee-space/...`, not `joint-space/`.
4. **Gap 5 smoke** (reconvert-independent):
   ```bash
   uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario ee_delta --select 1,2 --force
   ```
   → convert + train both PASS; checkpoint written under
   `outputs/model_zoo/ee_delta/smoke/checkpoints/`.
5. **Gap 1 reconvert** — run the A6 command; sanity-check `meta/stats.json`
   (xyz delta means ~1e-4, rot6d dims near identity) and
   `conversion_config.yaml` records `action_encoding: delta` with the current
   `code_commit`.
6. **End-to-end training smoke** on the reconverted data (adjust flags per
   `anvil-trainer --help`):
   ```bash
   uv run anvil-trainer --action-type=ee_delta \
     --dataset.root=data/debug/ee-delta-v2/ee-delta/ee-space-testing \
     --dataset.repo_id=local --policy.type=diffusion --policy.push_to_hub=false \
     --split-ratio=8,1,1 --steps=10 --save_freq=10 --batch_size=1 \
     --num_workers=0 --eval_freq=0 --log_freq=5
   ```
   Confirm it completes and writes a checkpoint under `model_zoo/ee-space/...`.

### Phase B
1. **Byte-identical config round-trip** — before/after introducing the
   `ConfigLoader` adapter, running `ConfigLoader.from_yaml` on every config
   under `configs/mcap_converter/v1.1/*.yaml` produces an identical
   `DataConfig`, and `to_yaml` produces identical output. Diff-check both.
2. **Migration parity** — `tests/unit/mcap_converter/test_versioning.py` and
   `test_migrate_config.py` pass unchanged against the extracted engine.
3. **ROS reader parity** — `resolve_action_type` / `resolve_observation_encoding`
   return identical values via `anvil_shared.dataset_config` as the old
   hand-rolled path, across joint / ee_abs / ee_delta sample datasets.
4. Full `uv run python -m pytest tests/unit -q` green.

## Notes

- All work stays in worktree `.worktrees/implement-ee-space`
  (branch `patrick/implement-ee-space`); commit there per the repo's
  git-worktree workflow. Phase A and Phase B land as separate
  commits/PRs — Phase A unblocks training immediately, Phase B is the
  broader refactor.
- Cosmetic stale docs (`configs/mcap_converter/v1.1/README.md`, the debug
  config's own header comment referencing deleted sibling files) are out of
  scope for both phases — leave for a later cleanup pass.
- Phase B touches files a concurrent agent is actively working on
  (`dataset_reader.py`, `dataset_gt_replayer_node.py`) — check in before
  starting B3 to avoid stepping on that work.

## Appendix — deferred eval-side gap inventory (for a later pass, unchanged from prior investigation)

Every action-type branch below predates `ee_delta` and currently omits it.
Not touched in Phase A or B; recorded here so a future pass has exact
locations.

### Newly confirmed (2026-07-19, Phase A+): eval/inference-side observation_encoding gap

Beyond the `ee_delta`-specific branches below, `anvil_eval/evaluator.py` and
ROS `inference_node.py`/`ee_runtime.py` share the SAME observation_encoding
gap that Phase A+ just fixed for the training path — confirmed via
exhaustive call-site audit, not touched in this pass because it needs new
plumbing, not just a parameter thread-through:

- `anvil_eval/evaluator.py`'s `EpisodeEvaluator` calls `ee_obs_relative_forward`,
  `ee_obs_abs_forward`, `ee_relative_forward`, `ee_relative_inverse` — but only
  ever receives `anvil_cfg` (checkpoint `anvil_config.json`), which today has
  no `observation_encoding` key at all (confirmed: `patches.py`'s
  `anvil_cfg_base` writes `action_type`/`is_ee`/`is_ee_relative`/
  `task_description`/`note`/git provenance, nothing about encoding).
- ROS `inference_node.py` (`LeRobotInferenceNode`) has the identical gap — its
  own `self.ee_abs_uses_rot6d_obs` heuristic (`obs_state_dim % 10 == 0`) is
  itself evidence this field is missing; it's a workaround, not a fix.
- ROS `ee_runtime.py`'s functions (`ee_relative_restore_chunk`,
  `ee_delta_restore_step`) are bare `(np.ndarray, np.ndarray) -> np.ndarray`
  utilities with no config parameter whatsoever — the hardest layer to fix,
  since encoding would need threading all the way from `inference_node.py`,
  which doesn't have it either.
- `dataset_reader.py:211` (`load_episode_observations_quat`) is already a
  best-practice example of doing this right — it calls
  `resolve_observation_encoding(dataset_root)` and gates on it explicitly.
- `dataset_gt_replayer_node.py` is the one "easy" case (has `self.dataset_path`
  and already imports the sibling `resolve_action_type` helper) — but it's
  actively being edited by a concurrent agent, so left untouched here.

Fix requires: (1) `patches.py`'s `anvil_cfg_base` writes
`observation_encoding` into `anvil_config.json` at train time; (2)
`EpisodeEvaluator`/`inference_node.py` read it from there (or, for `cli.py`,
resolve it directly from the dataset's own `conversion_config.yaml` since
`dataset_path` is already a local variable there); (3) thread it down into
`ee_runtime.py`'s bare functions. Cross-layer, multi-file — a dedicated pass,
not a small addition to Phase A+.

`packages/anvil_eval/src/anvil_eval/evaluator.py`
- `L85` `self.is_ee = self.action_type in ("ee_abs", "ee_relative")` — omits
  `ee_delta`; no `self.is_ee_delta` property exists yet.
- `L141` `_needs_restore = self.is_ee_relative` — `ee_delta` also needs action
  restore, via `ee_delta_inverse` (`ee_transform.py:314`), currently unhandled.
- `L160`/`L169` — obs-forward branches on `is_ee_relative` /
  `is_ee_abs`; `ee_delta`'s obs handling should mirror the `is_ee_abs` branch
  (`ee_obs_abs_forward`), not `is_ee_relative` — confirmed via
  `anvil_trainer/transforms.py:325` and `patches.py:465,515` docstrings
  ("obs stays absolute for ee_delta"). There is no `ee_obs_delta_forward`.
- `L201` raw-GT computation branches on `is_ee_relative` only; `ee_delta`
  falls through to a plain GT append today.

`packages/anvil_eval/src/anvil_eval/metrics.py`
- `L198` `if action_type in ("ee_abs", "ee_relative", "ee_rel") and predicted.shape[1] % 8 == 0:`
  — gates the entire EE-Cartesian metrics path; `ee_delta` episodes fall to
  generic (meaningless) metrics today.

`packages/anvil_eval/src/anvil_eval/plotting.py`
- `L48`, `L140` `show_delta = action_type in ("ee_relative", "ee_rel") and ...`
  — whether to show a delta panel for `ee_delta` is a design choice to make
  in the follow-up pass.
- `L148` `_is_ee = action_type in ("ee_abs", "ee_relative", "ee_rel")` — gates
  EE axis labels/scaling.

`packages/anvil_eval/src/anvil_eval/cli.py`
- `L194`, `L229` `if evaluator.is_ee:` — auto-corrects once `evaluator.is_ee`
  (evaluator.py:85) includes `ee_delta`.

`packages/anvil_eval_ros/src/anvil_eval_ros/cli.py`
- `L363` `is_ee = action_type in ("ee_abs", "ee_relative", "ee_rel")` — master
  gate for arm-info / `dims_per_arm=10` / EE command topics; all downstream
  `is_ee` uses (L369, L374, L378, L384, L401, L416, L428, L451, L479,
  L495, L502-504) auto-correct once this tuple includes `ee_delta`.
- `L357`, `L603` — docstring/log-string mentions of `ee_abs, ee_relative`
  should be updated to include `ee_delta`.
