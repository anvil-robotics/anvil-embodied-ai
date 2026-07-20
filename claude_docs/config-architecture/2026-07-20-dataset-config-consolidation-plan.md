# Plan — Consolidate dataset-config reading into one canonical `anvil_shared` API (Option C)

Date: 2026-07-20
Branch: `patrick/implement-ee-space`
Status: **SUPERSEDED** by
`claude_docs/config-architecture/2026-07-20-config-schema-behavior-split-plan.md`.
After this doc was approved but before implementation started, the user asked
for a stricter architecture: split *schema* (pure dataclass fields) from
*behavior* (loading/validation/CLI) within `anvil_shared`, rather than this
doc's "slim resolved-descriptor" approach (which still mixed some derived
logic into the shared reader). The new doc is the one to implement; this one
is kept for historical context only.

## Context

`packages/mcap_converter/src/mcap_converter/config/` (`DataConfig`, `ConfigLoader`,
the version-migration engine) is the only complete, typed definition of what a
converted dataset's `conversion_config.yaml` means — but `mcap_converter` is not
shipped in the ROS2 inference Docker image, so every ROS2/eval consumer that
needs to know a dataset's `data_space`/`action_encoding`/`observation_encoding`
has ended up hand-rolling its own `yaml.safe_load` + ad hoc defaulting/fallback
logic instead: `ros2/.../dataset_reader.py`, `anvil_eval_ros/cli.py` (3 call
sites), and `anvil_trainer/config.py` (which also independently reinvented a
feature-name-suffix fallback for legacy datasets lacking the file). This plan
converges all of that onto **one** shared, canonical reader.

Two decisions made explicitly with the user (2026-07-20), after presenting 3
alternatives (move the whole schema+loader engine verbatim; also fold in
`anvil_trainer/config.py`'s `TrainingConfig`; or a slim resolved-descriptor
reader) and one rejected alternative (a `Protocol`-only interface, no code
movement — doesn't solve the actual Docker-availability problem):

1. **Approach: slim resolved-descriptor reader** (not moving the whole schema
   engine). `mcap_converter/config/` stays exactly as-is — untouched,
   package-private, authoring-time-only (strict validation, legacy v1.0
   migration, joint-mode fields) — none of that is relevant to a *reader*.
   `anvil_shared` gains a complete, self-contained, read-only API for
   "resolved facts about an already-converted dataset," built by expanding
   the `anvil_shared/dataset_config.py` + `ee_encodings.py` modules already
   started for the ee_delta gap-3 fix. Two related types (mcap_converter's
   authoring-time `DataConfig`, and the read-only resolved descriptor) instead
   of one literal shared class — a deliberate trade-off: it keeps
   `anvil_shared` free of authoring-time complexity nothing but the converter
   itself needs.
2. **`anvil_trainer/config.py`'s `TrainingConfig` stays in `anvil_trainer`,
   not folded in.** It's training-CLI orchestration (`sys.argv` injection for
   lerobot/draccus, EMA/DDPM-IP/wandb/resume logic) — a fundamentally
   different kind of "config" than a dataset schema. Nothing outside
   `anvil_trainer` has ever needed it (confirmed: `anvil_eval` has its own
   separate `EvalConfig`; ROS2 only ever reads a training run's *output*,
   `anvil_config.json`, never `TrainingConfig` itself). Moving it would not
   solve the ROS2-Docker problem and would cost `anvil_shared` its
   "dependency-light, importable anywhere" guarantee.

Verified before writing this plan: `anvil_shared.dataset_config` currently has
exactly one consumer (`anvil_trainer/config.py`) — full freedom to redesign its
API without a migration/back-compat shim.

Related prior work this plan builds on: `claude_docs/ee-delta/2026-07-19-training-flow-gaps-fix-plan.md`
(the `anvil_shared/ee_encodings.py` + `dataset_config.py` modules were first
created there, for the ee_delta gap-3 dataset-shape-validation fix — this plan
expands them into the full canonical reader that Phase B of that doc had only
sketched). Independent of `claude_docs/ee-delta/2026-07-20-anvil-eval-gaps-fix-plan.md` (the two touch
`anvil_eval_ros/cli.py` at different, non-overlapping lines).

## Current state (verified line-by-line, 2026-07-20)

- `anvil_shared/ee_encodings.py` — has `OBSERVATION_ROTATION_LAYOUTS`,
  `observation_state_dim_per_arm`, `observation_state_names_per_arm`,
  `encode_rotation`. Missing: any EE-detection-from-feature-names logic.
- `anvil_shared/dataset_config.py` — has `read_conversion_config`,
  `resolve_data_space`, `resolve_action_encoding`, `resolve_observation_encoding`,
  `resolve_action_type` — all operate on an already-loaded `dict`, all default
  gracefully when the file/keys are absent. Missing: anything that reads
  `meta/info.json`, computes `n_arms`, or does the suffix-based fallback
  inference.
- `anvil_trainer/config.py` — `_EE_STATE_MARKER_SUFFIXES`,
  `_EE_ACTION_MARKER_SUFFIXES`, `_has_ee_markers`,
  `_OBSERVATION_ENCODING_STATE_MARKERS`, `_infer_observation_encoding`
  (~60 lines, config.py:89-134) are **local, duplicated** logic that
  conceptually belongs in the shared module, not here. `validate_action_space`
  (config.py:558-649) is the sole real caller of all of this.
- `ros2/.../dataset_reader.py` — `_load_conversion_config`, `resolve_action_type`,
  `resolve_observation_encoding` (lines ~58-89) are a **near-exact duplicate**
  of `anvil_shared.dataset_config`'s equivalents (they were originally designed
  to match semantics exactly, anticipating this exact consolidation).
- `anvil_eval_ros/cli.py` — 3 independent `yaml.safe_load(config_path.read_text())`
  calls on `conversion_config.yaml` (lines ~210, ~316, ~433), each hand-rolled.
- `mcap_converter/config/` — untouched by this plan entirely.

## Changes

### 1. `anvil_shared/ee_encodings.py` — add EE-detection-from-names logic
Move (not duplicate) the marker-suffix detection from `anvil_trainer/config.py`
here, generalized and made public:
```python
EE_STATE_MARKER_SUFFIXES = {"qx","qy","qz","qw","r0","r1","r2","r3","r4","r5","ax","ay","az"}
EE_ACTION_MARKER_SUFFIXES = {"r0","r1","r2","r3","r4","r5"}

def has_ee_markers(names: list[str], markers: set[str]) -> bool: ...
def infer_observation_encoding_from_names(state_names: list[str]) -> str: ...
```
Semantics identical to today's `anvil_trainer/config.py:89-134` (bare or
arm-prefixed name matching; per-encoding suffix table; defaults to
`"quaternion"` if nothing matches).

### 2. `anvil_shared/dataset_config.py` — the new canonical entry point
Add:
```python
@dataclass
class ResolvedDatasetConfig:
    data_space: str                 # "joint" | "ee"
    action_encoding: str             # "absolute" | "delta" | "relative"
    observation_encoding: str        # "quaternion" | "rot6d" | "axis_angle"
    action_type: str                 # "joint_abs" | "ee_abs" | "ee_delta"
    is_ee: bool
    n_arms: int
    state_dim_per_arm: int
    action_dim_per_arm: int          # always 10 (rot6d), present for convenience
    camera_names: list[str]
    from_conversion_config: bool     # True = read from conversion_config.yaml
                                      # (ground truth); False = suffix-inference
                                      # fallback (legacy dataset, no such file)

def resolve_dataset_config(
    dataset_root: str | Path, *, logger: Any = None
) -> ResolvedDatasetConfig:
    """Self-contained: reads conversion_config.yaml AND meta/info.json from
    dataset_root itself. Prefers conversion_config.yaml as ground truth;
    falls back to feature-name-suffix inference (via ee_encodings' new
    helpers) against meta/info.json's state/action names when that file is
    missing. n_arms is always derived from action_dim // 10 (action is always
    rot6d regardless of observation_encoding — see ee_encodings.py)."""
```
Keep the existing small functions (`read_conversion_config`, `resolve_data_space`,
etc.) as-is — still useful low-level primitives, and `resolve_dataset_config`
is built on top of them, not a replacement.

### 3. `anvil_trainer/config.py` — consume the shared resolver
Delete `_EE_STATE_MARKER_SUFFIXES`, `_EE_ACTION_MARKER_SUFFIXES`,
`_has_ee_markers`, `_OBSERVATION_ENCODING_STATE_MARKERS`,
`_infer_observation_encoding` (config.py:89-134) entirely. Rewrite
`validate_action_space` (config.py:558-649) to call
`resolve_dataset_config(self.dataset_root, logger=log)` once, use its fields
directly for the is_ee/dimension checks, and set
`self.observation_encoding = resolved.observation_encoding`. This must be a
**behavior-preserving refactor** — same raises, same messages, same log lines
— verified by the existing `test_ee_validation.py` suite passing unchanged
(see Verification).

### 4. `anvil_eval_ros/cli.py` — dedupe the 3 read sites
Replace each `yaml.safe_load(config_path.read_text())` (lines ~210, ~316, ~433)
with `anvil_shared.dataset_config.read_conversion_config(dataset_root)`.
Mechanical, same dict shape returned — no change to each call site's own
downstream interpretation logic (arm defaulting, `action_from_observation`
detection, etc.), only the file-read mechanism. (The separate `is_ee` tuple
fix for `ee_delta` at cli.py:363 belongs to
`claude_docs/ee-delta/2026-07-20-anvil-eval-gaps-fix-plan.md`, not this plan — independent, both
touch this file, no conflict.)

### 5. ROS2 `dataset_reader.py` / `dataset_gt_replayer_node.py` — same fix, gated
Delete `dataset_reader.py`'s local `_load_conversion_config`/`resolve_action_type`/
`resolve_observation_encoding` (~lines 58-89); import the equivalents directly
from `anvil_shared.dataset_config` (already on this file's import path via its
existing `_ensure_anvil_shared()` sys.path shim — no new plumbing needed).
**Before touching this file**: re-check `git status` for whether the
concurrent GT-replay agent is still actively editing it (it was mid-edit as of
2026-07-19, investigating an unrelated anchor-staleness bug in
`inference_node.py`) — if still active, defer this file specifically and land
items 1-4 alone first.

## Verification

1. **Unit tests, new**: `tests/unit/anvil_shared/test_ee_encodings.py` (new
   file, or extend if one exists) for `has_ee_markers`/
   `infer_observation_encoding_from_names`; `tests/unit/anvil_shared/test_dataset_config.py`
   (new) for `resolve_dataset_config` — cover: conversion_config.yaml present
   × all 3 observation_encodings × single/bimanual; conversion_config.yaml
   absent (fallback path) × all 3 encodings via feature-name suffixes; both
   files missing (degenerate default).
2. **Regression, the important one**: `tests/unit/anvil_trainer/test_ee_validation.py`
   (21 tests, added during the ee_delta gap-3 fix) must pass **unchanged,
   verbatim** after the `validate_action_space` refactor — this is the
   concrete proof the consolidation is behavior-preserving, not just "looks
   equivalent."
3. Full `uv run python -m pytest tests/unit -q` stays green (baseline: 642
   passed, 2 skipped).
4. Re-run the `ee_delta`/`ee_delta_rot6d` smoke scenarios
   (`--scenario ee_delta,ee_delta_rot6d --select 1,2 --force`) to confirm the
   real convert→train path is unaffected by the `anvil_trainer/config.py`
   refactor.
5. If item 4 (`anvil_eval_ros/cli.py`) lands: run
   `tests/unit/anvil_eval_ros/test_ros_eval.py` (existing suite) to confirm
   the 3 swapped read sites behave identically.

## Notes
- All work in worktree `.worktrees/implement-ee-space`
  (branch `patrick/implement-ee-space`); commit per the git-worktree rule.
- `mcap_converter/config/` (schema.py, loader.py, versioning.py, validators.py)
  is explicitly **out of scope** — zero changes, zero risk to the converter's
  authoring-time correctness.
- This plan is independent of, and can land before or after,
  `claude_docs/ee-delta/2026-07-20-anvil-eval-gaps-fix-plan.md` — the two touch `anvil_eval_ros/cli.py`
  at different, non-overlapping lines (this plan: the 3 yaml-read call sites;
  that plan: the `is_ee` tuple at line 363).
- Item 5 (ROS2 `dataset_reader.py`) is conditionally scoped on the concurrent
  GT-replay agent's status at execution time — check `git status` fresh
  before touching it.
