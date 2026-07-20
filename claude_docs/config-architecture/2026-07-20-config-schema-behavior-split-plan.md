# Plan — Split anvil_shared config into schema (data shape) vs. behavior (loading/validation/CLI)

Date: 2026-07-20
Branch: `patrick/implement-ee-space`
Status: Investigation complete, plan ready for approval. **No code written yet.**
Supersedes `claude_docs/config-architecture/2026-07-20-dataset-config-consolidation-plan.md`.

## Context

Read first: `claude_docs/ee-delta/2026-07-19-training-flow-gaps-fix-plan.md` and
`claude_docs/ee-delta/2026-07-20-anvil-eval-gaps-fix-plan.md`.

We previously considered three approaches to the "dataset-config duplication"
problem (A: move mcap_converter's whole schema+loader engine to `anvil_shared`
verbatim; B: A + also move `TrainingConfig`; C: a slim resolved-descriptor
reader) — all three are superseded by the approach below. The core problem
those docs identified stands: `packages/mcap_converter/src/mcap_converter/config/`
(`DataConfig`, `ConfigLoader`, the migration engine) is the only complete,
typed definition of a converted dataset's `conversion_config.yaml`, but
`mcap_converter` isn't shipped in the ROS2 inference Docker image — so ROS2
nodes, `anvil_eval_ros`, and `anvil_trainer` each hand-rolled their own
`yaml.safe_load` + ad hoc defaulting/fallback logic instead of sharing one
definition.

The new instruction sharpens *how* to fix this: don't just relocate reading
logic — enforce a hard architectural boundary. `anvil_shared/config/` may
contain **only** dataclass fields, type hints, and trivial defaults; all
loading/validation/migration/CLI-parsing behavior stays in each consuming
package, importing the shared dataclass. This is a stricter, more durable rule
than "Option C"'s slim-reader approach, which still let some derived-value
logic live in the shared module without a clear line.

## Phase 0 — Investigation (all findings cite file:line; nothing assumed)

### 0.1 Import-dependency safety check for `mcap_converter/config/`

Audited all four files' complete import lists:

- **`schema.py`** (`config/schema.py:25-32`): only `dataclasses`, `typing`
  [STDLIB], and `from anvil_shared.ee_encodings import (...)` [ANVIL_SHARED,
  already relocated there in the prior ee_delta work]. Zero `mcap_converter.core`
  or third-party imports.
- **`validators.py`** (`config/validators.py:10,12`): `typing` [STDLIB] +
  `from .schema import ConfigurationError, DataConfig` [same-package].
  `validate_topics_exist(config: DataConfig, available_topics: List[str])`
  (`validators.py:17`) takes `available_topics` as a plain caller-supplied
  parameter — confirmed it does **not** reach into `mcap_converter.core` for
  topic discovery. Fully decoupled already.
- **`versioning.py`** (`config/versioning.py:48-54`): stdlib only
  (`logging`, `dataclasses`, `typing`) + `from .schema import
  CURRENT_SCHEMA_VERSION, ConfigurationError` [same-package].
- **`loader.py`** (`config/loader.py:45-58`): stdlib (`pathlib`, `typing`) +
  `import yaml` [THIRD_PARTY: pyyaml, already an `anvil_shared` dependency
  since the Phase A ee_delta work] + `from .schema import (...)` +
  `from .versioning import migrate_to_current` [same-package].
- **Dependency direction confirmed one-way**: `core/writer.py:9`,
  `core/extractor.py:9`, `core/aligner.py:7` import `DataConfig` FROM
  `config.schema`; nothing in `config/{schema,validators,versioning}.py`
  imports anything from `core/` (which is where all the heavy deps — `mcap`,
  `opencv-python`, `lerobot`, `numpy` — actually live, per
  `mcap_converter/pyproject.toml:27-38`).

**Verdict: `DataConfig` is safe to move as a pure dataclass with zero heavy-dependency drag.**

### 0.2 `DataConfig`'s actual method/property inventory — the real complication

Read `schema.py` in full (203-462). `DataConfig` is not pure data today — it
has 7 properties and 2 methods. Categorized against the hard rule:

| Member | Raises? | I/O? | Verdict |
|---|---|---|---|
| `is_ee` (schema.py:255-257) | no | no | **KEEP** — trivial 1-field comparison |
| `is_action_delta` (259-265) | no | no | **KEEP** — trivial 2-field comparison |
| `output_subdir` (267-281) | no | no | **KEEP** — dict lookup + string format, no raise |
| `arms` (283-286) | no | no | **KEEP** — `list(dict.keys())` |
| `action_command_topics` (307-320) | no | no | **KEEP** — dict inversion, no raise |
| `robot_state_topic` (288-305) | **yes, `ValueError`** | no | **EXTRACT** — validation-shaped, raises |
| `validate()` (327-429) | **yes, `ConfigurationError`** | no | **EXTRACT** — explicitly validation |
| `to_dict()` (431-458) | no | no | **EXTRACT** — real serialization logic tied to YAML writing (`loader.py`'s `to_yaml` is its only caller); not "trivial defaults" |

Also: `validate_joint_name_pattern` (82-98) and `validate_feature_mapping`
(101-111) are already free functions (not `DataConfig` methods) called only
from `validate()` — these stay in mcap_converter unchanged, just relocate
alongside the extracted `validate()`.

### 0.3 Call-site blast radius for the 3 extracted members (grepped, not estimated)

- **`.validate()`**: 1 production call site (`cli/convert.py:888`) + 18 test
  call sites (`test_ee_encoding.py` ×6, `test_versioning.py` ×4,
  `test_joint_ordering.py` ×8) — all as `cfg.validate()`/`config.validate()`
  method calls. **19 total**, all mechanical `cfg.validate()` →
  `validate_data_config(cfg)` renames.
- **`.to_dict()`**: exactly **1** call site (`config/loader.py:94`, inside
  `ConfigLoader.to_yaml`). (The other `.to_dict()` grep hit,
  `cli/mcap_valid.py:550`, is on an unrelated quality-report class — not
  `DataConfig`.)
- **`.robot_state_topic`**: 7 call sites — `cli/convert.py:189`,
  `core/extractor.py:193,218,729,780,922`, plus the internal use inside
  `validate()` itself (schema.py:349, moves together since `validate()` moves
  too). All read-only property access → free-function-call rewrites.
- **`.action_command_topics`** (kept as a property, not extracted): 15 call
  sites (`convert.py` ×5, `extractor.py` ×8, tests ×2) — **zero changes
  needed**, since it stays a `DataConfig` property.

This call-site audit is the concrete "Phase 0 precondition" the task asked
for: the move is NOT a pure rename — `validate()`/`robot_state_topic`/`to_dict()`
extraction touches ~27 call sites across `convert.py`, `extractor.py`,
`loader.py`, and 3 test files. All mechanical (method access → function call,
identical semantics), but real.

### 0.4 `anvil_config.json` complete field inventory (checkpoint metadata)

Read `patches.py:860-875` directly:
```python
anvil_cfg_base: dict = {
    "action_type": self.config.action_type,
    "is_ee": self.config.is_ee,
    "is_ee_relative": self.config.is_ee_relative,
    **git_provenance(),
}
if self.config.task_override:
    anvil_cfg_base["task_description"] = self.config.task_override
if self.config.note:
    anvil_cfg_base["note"] = self.config.note
```
`git_provenance()` (`anvil_shared/provenance.py:14-38`) returns 0, 1, or 2 of
`code_commit`/`code_tag` (all subprocess errors swallowed — never raises).

**Complete field set**: `action_type` (str), `is_ee` (bool), `is_ee_relative`
(bool), `code_commit` (optional str), `code_tag` (optional str),
`task_description` (optional str), `note` (optional str). Plus the field this
whole redesign needs to ADD: `observation_encoding` (the root cause identified
in `ee-delta-anvil-eval-gaps-fix-plan.md` finding #1 — never persisted today).

**Readers** (from this session's prior audits, re-confirmed consistent):
`anvil_eval_ros/cli.py:592-607` (action_type only), ROS `inference_node.py:382-388`
(action_type, task_description), ROS `ee_runtime.py:60,75` (action_type, via
`resolve_action_type`/`read_checkpoint_anvil_config`), `anvil_eval/cli.py:100-124`
(task_description + whole dict passed to `resolve_splits`), `anvil_trainer/config.py:361,680`
(action_type, note — `--resume` inheritance).

### 0.5 `EvalConfig` consumer check

Grepped the entire repo for `EvalConfig`: all 4 non-definition references
(`anvil_eval/cli.py:11,110`, `anvil_eval/dataset.py:15,126`) are inside
`anvil_eval` itself. Confirmed `anvil_eval_ros/` and `anvil_trainer/` contain
zero references to `anvil_eval` at all (`grep -r "anvil_eval\b"` empty in
both). **`EvalConfig` is genuinely single-consumer — per the task's own
instruction, skip building `anvil_shared/config/eval/` for `EvalConfig`.**

### 0.6 `metrics_summary.json` / results-schema consumer check

Not single-consumer, but the actual duplication is much smaller than the
dataset-config or checkpoint-metadata cases:
- **Writers**: `anvil_eval/cli.py:222-223` (via `anvil_eval.reporting`) and
  ROS2's `eval_recorder_node.py:647-650`, which imports `anvil_eval.metrics`/
  `.reporting`/`.plotting` **directly as Python functions** (via a
  `_ensure_anvil_eval_importable()` sys.path shim, `eval_recorder_node.py:47-59`)
  — i.e. this ROS2 node already reuses the real implementation, not a
  reimplementation. `anvil_eval.metrics`/`.reporting` have no torch/lerobot
  imports (confirmed: plain `numpy`/`dataclasses`/`typing`), so this works
  without heavy deps in the ROS2 context already, unprompted by this plan.
- **Readers of the JSON shape** (not the Python API): `anvil_eval_ros/cli.py:810-825`
  and `tests/smoke/scripts/pipeline_smoke_test.py:558-561` both do
  `json.loads(...)["overall"]["mean_mae"]`-style ad hoc parsing.
- Smoke-test step 3 (`anvil-eval`) and step 4 (`anvil-eval-ros`) write to
  **separate directories** and never read each other's output
  (`pipeline_smoke_test.py:159-250` shows distinct `eval_out`/`eval_ros_out`
  paths).

**Verdict: skip `anvil_shared/config/eval/` for now.** The real sharing point
(metrics/reporting logic) is already correctly consolidated via direct Python
import, not duplicated. The remaining raw-JSON-field-access at 2 call sites is
minor — worth a one-line note, not a new shared module, per the task's "don't
fill folders for symmetry" instruction. A `results schema` dataclass is a
plausible future addition if this duplication grows, not a need today.

### 0.7 `TrainingConfig` consumer check

Already established across this session's prior work (dataset-config-consolidation-plan.md's
own investigation): zero consumers outside `anvil_trainer`. **Skip `anvil_shared/config/train/`.**

## Scope

**Subfolders actually built**: `anvil_shared/config/datasets/` and
`anvil_shared/config/inference/`. `train/` and `eval/` are **not** built
(confirmed empty per 0.5-0.7).

**In scope**:
- `anvil_shared/config/` (new subpackage): `datasets/` (DataConfig + siblings,
  pure), `inference/` (new `AnvilCheckpointConfig`, pure).
- `mcap_converter/config/schema.py` — shrinks to: re-export `DataConfig` (+
  nested types + constants) from `anvil_shared`, keep `ConfigurationError`,
  `validate_joint_name_pattern`, `validate_feature_mapping`, and the 3
  extracted behaviors (`validate_data_config`, `robot_state_topic`, `to_dict`
  as free functions), plus `DEFAULT_DATA_CONFIG`.
- `mcap_converter/config/versioning.py`, `loader.py`, `validators.py`,
  `__init__.py` — import updates only (where each symbol now lives), no
  logic changes.
- `mcap_converter/cli/convert.py`, `core/extractor.py` — ~27 call-site
  rewrites (0.3) from method access to free-function calls.
- `anvil_shared/dataset_config.py` — redesigned to return a real `DataConfig`
  instance (constructed leniently from the YAML dict, filtered to known
  fields) instead of a bare `dict` + ad hoc `resolve_*` functions; add a new
  `action_type` derived property directly on `DataConfig` (mirrors the
  existing `is_ee`/`is_action_delta` precedent) so `resolve_action_type` can
  be retired in favor of `cfg.action_type`. Suffix-inference fallback (for
  legacy datasets lacking `conversion_config.yaml`) is preserved, now
  producing a best-effort `DataConfig` instead of a dict.
- `anvil_trainer/config.py` — `validate_action_space` consumes the real
  `DataConfig` type instead of its own local marker-suffix constants +
  `_infer_observation_encoding` (deleted, logic now lives in
  `anvil_shared/dataset_config.py`'s fallback path).
- `anvil_eval_ros/cli.py` — the 3 hand-rolled `yaml.safe_load` reads
  (lines ~210, 316, 433) switch to constructing/reading the shared
  `DataConfig` via `anvil_shared.dataset_config`.
- `anvil_trainer/patches.py` — `anvil_cfg_base` becomes an
  `AnvilCheckpointConfig` instance (adds the missing `observation_encoding`
  field), serialized via a trivial `dataclasses.asdict`-style call.

**Deferred / explicitly NOT touched this pass** (see Notes for why):
- ROS2 `dataset_reader.py` / `dataset_gt_replayer_node.py` (dataset-config
  reading) and `ee_runtime.py` / `inference_node.py` (checkpoint-config
  reading) — both are files the concurrent GT-replay agent has been actively
  editing (confirmed in `ee-delta-anvil-eval-gaps-fix-plan.md`'s own notes).
  Re-check `git status` before touching; land everything else first.
- `anvil_eval/evaluator.py`'s consumption of the new `AnvilCheckpointConfig`
  — see the sequencing conflict below.
- Building `anvil_shared/config/train/` or `config/eval/` — confirmed
  unneeded (0.5-0.7).

## Changes

### 1. `anvil_shared/config/datasets/` (new)
`anvil_shared/config/datasets/schema.py` (or `data_config.py`) — pure
dataclasses, moved verbatim from `mcap_converter/config/schema.py`:
- `DataConfig` (all 14 fields, schema.py:203-249) plus the 5 **non-raising**
  properties (`is_ee`, `is_action_delta`, `output_subdir`, `arms`,
  `action_command_topics`) kept as-is.
- **New property**: `action_type` — derives `"joint_abs"`/`"ee_abs"`/`"ee_delta"`
  from `data_space`/`action_encoding`, folding in what
  `anvil_shared.dataset_config.resolve_action_type` does today as a free
  function on a dict.
- Sibling dataclasses: `JointNamePattern`, `ActionTopicConfig`, `ActionTopicSpec`,
  `FeatureMapping` (all pure per 0.2, unchanged).
- Constants: `CURRENT_SCHEMA_VERSION`, `RECOGNIZED_YAML_KEYS`,
  `_ACTION_ENCODING_SUBDIR_ABBREV` (needed by the kept `output_subdir` property).
- **Not included** (extracted to mcap_converter instead, per 0.2):
  `validate()`, `robot_state_topic` (the raising property — replaced by a
  free function of the same name in mcap_converter), `to_dict()`.
- **Judgment call, flagging for your input**: `RECOGNIZED_YAML_KEYS` is
  loader-strict-mode-specific data (only consumed by `loader.py`'s
  unknown-key rejection) but is pure, logic-free data (a frozenset of
  strings) — proposing to keep it with the schema for cohesion (it's
  literally "what keys does DataConfig recognize"), but it could equally
  live in mcap_converter if you'd rather keep anything loader-shaped there.

### 2. `mcap_converter/config/schema.py` (shrinks, doesn't disappear)
```python
from anvil_shared.config.datasets.schema import (
    DataConfig, JointNamePattern, ActionTopicConfig, ActionTopicSpec,
    FeatureMapping, CURRENT_SCHEMA_VERSION, RECOGNIZED_YAML_KEYS,
)

class ConfigurationError(Exception): ...   # unchanged, stays here (only mcap_converter raises it)

def validate_joint_name_pattern(pattern): ...   # unchanged, moved verbatim (still free functions)
def validate_feature_mapping(mapping, name): ...

def validate_data_config(config: DataConfig) -> None:
    """Extracted from DataConfig.validate() verbatim — same body, same errors,
    same ConfigurationError. Only the `self` -> `config` parameter changes."""
    ...

def robot_state_topic(config: DataConfig) -> str:
    """Extracted from the raising DataConfig.robot_state_topic property."""
    ...

def to_dict(config: DataConfig) -> Dict[str, Any]:
    """Extracted from DataConfig.to_dict() verbatim."""
    ...

DEFAULT_DATA_CONFIG = DataConfig()
```
Update the 27 call sites (0.3): `convert.py:888` `config.validate()` →
`validate_data_config(config)`; `convert.py:189` + `extractor.py:193,218,729,780,922`
`config.robot_state_topic` → `robot_state_topic(config)`; `loader.py:94`
`config.to_dict()` → `to_dict(config)`; 18 test call sites `cfg.validate()` →
`validate_data_config(cfg)`.

### 3. `mcap_converter/config/versioning.py`, `loader.py`, `validators.py`, `__init__.py`
Import-only updates:
- `versioning.py:54`: `from .schema import CURRENT_SCHEMA_VERSION, ConfigurationError`
  → `from anvil_shared.config.datasets.schema import CURRENT_SCHEMA_VERSION` +
  `from .schema import ConfigurationError` (split across two sources).
- `loader.py:50-58`: `ActionTopicSpec, DataConfig, FeatureMapping,
  JointNamePattern, RECOGNIZED_YAML_KEYS` now come from
  `anvil_shared.config.datasets.schema`; `ConfigurationError` still from
  `.schema` (mcap_converter's). `loader.py`'s own logic (`ConfigLoader.from_dict`,
  `_parse_*` hydrators, `to_yaml`) is unchanged — it's genuinely "loading
  behavior," stays exactly where it is.
- `validators.py:12`: `DataConfig` from `anvil_shared`, `ConfigurationError`
  from `.schema`. `validate_topics_exist` body unchanged (already decoupled,
  0.1).
- `config/__init__.py:3-13`: re-export list updated to source `DataConfig`/
  nested types from the new location transparently — external code doing
  `from mcap_converter.config import DataConfig` keeps working unchanged.

### 4. `anvil_shared/config/inference/` (new)
`anvil_shared/config/inference/checkpoint.py`:
```python
@dataclass
class AnvilCheckpointConfig:
    action_type: str = "joint_abs"
    is_ee: bool = False
    is_ee_relative: bool = False
    observation_encoding: str = "quaternion"   # the fix — never persisted before
    code_commit: str | None = None
    code_tag: str | None = None
    task_description: str | None = None
    note: str | None = None
```
Every current field from `anvil_cfg_base` (0.4) is covered, plus the missing
`observation_encoding`. Pure dataclass — no methods beyond perhaps a
`from_dict`/`to_dict` classmethod pair that ONLY filters/maps dict keys (no
raising, no I/O) — **flagging as a judgment call**: strictly, a classmethod
isn't a "trivial default," but it's the only way to make this usable for
forward/backward-compatible reads (old checkpoints lack `observation_encoding`;
future checkpoints might add fields an older `anvil_shared` doesn't know
about) without scattering key-filtering logic at every read site. Alternative:
leave `from_dict`/`to_dict` out of `anvil_shared/config/` entirely and define
them in the new `anvil_shared/checkpoint_config.py` (below) instead, keeping
the dataclass in `config/inference/` truly members-only. Your call — I lean
towards the latter (stricter reading of the hard rule).

`anvil_shared/checkpoint_config.py` (new, sibling to the existing
`dataset_config.py` — this is where the "read anvil_config.json, construct a
typed object" behavior lives, since it does file I/O and thus cannot go under
`config/`):
```python
def read_checkpoint_config(checkpoint_dir: str | Path) -> AnvilCheckpointConfig:
    """Read pretrained_model/anvil_config.json leniently — unknown keys ignored,
    missing keys use AnvilCheckpointConfig's field defaults (so old checkpoints
    without observation_encoding load fine, defaulting to "quaternion")."""
```

### 5. `anvil_trainer/patches.py` — write via the new type
`anvil_cfg_base` (864-873) becomes:
```python
anvil_cfg = AnvilCheckpointConfig(
    action_type=self.config.action_type,
    is_ee=self.config.is_ee,
    is_ee_relative=self.config.is_ee_relative,
    observation_encoding=self.config.observation_encoding,   # the actual fix
    task_description=self.config.task_override or None,
    note=self.config.note or None,
    **git_provenance(),
)
```
Written via `json.dumps(dataclasses.asdict(anvil_cfg), indent=2)` at the
existing write site (patches.py:937).

### 6. `anvil_shared/dataset_config.py` — redesigned around real `DataConfig`
Replace the bare-dict-returning `read_conversion_config`/`resolve_*` functions
with a lenient `DataConfig` constructor: read the YAML, filter to
`DataConfig`'s known field names, construct `DataConfig(**filtered)`. Keep
the suffix-inference fallback (from the ee_delta gap-3 work) for datasets
lacking `conversion_config.yaml` entirely, now producing a best-effort
`DataConfig` rather than a dict. `resolve_action_type` retires in favor of
the new `DataConfig.action_type` property (item 1).

### 7. `anvil_trainer/config.py` — consume real `DataConfig`
`validate_action_space` (currently config.py:558-649, per the
`training-flow-gaps-fix-plan.md`'s Phase A+ state) drops its local
`_EE_STATE_MARKER_SUFFIXES`/`_has_ee_markers`/`_infer_observation_encoding`
(that logic now lives inside `anvil_shared/dataset_config.py`'s fallback
path) and instead calls the redesigned reader, using `cfg.is_ee`,
`cfg.observation_encoding`, `cfg.action_type` directly.

### 8. `anvil_eval_ros/cli.py` — dataset-config read sites
Lines ~210, 316, 433: replace raw `yaml.safe_load(config_path.read_text())`
with the redesigned `anvil_shared.dataset_config` reader, getting a real
`DataConfig` back instead of a dict. Downstream arm-defaulting/topic logic at
each site is otherwise unchanged (same as the superseded plan's item 4).

## Sequencing conflicts with the two in-flight plans — needs your decision

1. **`anvil_eval/evaluator.py` vs. `ee-delta-anvil-eval-gaps-fix-plan.md`.**
   That plan (not yet implemented) specifies
   `self.observation_encoding: str = anvil_cfg.get("observation_encoding", "quaternion")`
   — written assuming `anvil_cfg` stays a raw `dict`. This plan's item 4/5
   introduce a typed `AnvilCheckpointConfig` instead. Since **neither plan has
   been implemented in code yet**, I recommend: land this plan's `inference/`
   + `patches.py` write-side (items 4-5) first, then implement the anvil-eval
   plan's `evaluator.py` changes directly against `AnvilCheckpointConfig` from
   the start (`self.observation_encoding = anvil_cfg.observation_encoding`,
   simpler than the dict-`.get()` form) — avoiding writing dict-based code now
   just to retype it later. This means `EpisodeEvaluator.__init__`'s
   `anvil_cfg` parameter type changes from `dict` to `AnvilCheckpointConfig`;
   `anvil_eval/cli.py:100-124`'s construction site changes from
   `json.loads(...)` to `read_checkpoint_config(...)` accordingly (small
   ripple, same file already in that plan's scope).
2. **ROS2 files under concurrent edit.** Both this plan (items for
   `dataset_reader.py`/`ee_runtime.py`/`inference_node.py`, listed as
   deferred above) and the anvil-eval plan (finding #6, also deferred) touch
   files the concurrent GT-replay agent has live debug scaffolding in. Same
   mitigation as that plan: re-check `git status` immediately before touching
   any of the three ROS2 files; land everything else first regardless of
   that agent's status.

## Verification

1. **`test_ee_validation.py` (21 tests, added in the ee_delta gap-3 fix) must
   pass unchanged, verbatim** — the concrete proof `anvil_trainer/config.py`'s
   refactor (item 7) is behavior-preserving.
2. **mcap_converter's own test suite** (`test_ee_encoding.py`,
   `test_versioning.py`, `test_joint_ordering.py` — the 18 `.validate()`
   call sites) — after the mechanical `cfg.validate()` → `validate_data_config(cfg)`
   rename, all must still pass with identical pass/fail behavior per test.
3. **New unit tests**: `tests/unit/anvil_shared/test_config_datasets.py` for
   the moved `DataConfig` (construction, the 5 kept properties, the new
   `action_type` property) and `tests/unit/anvil_shared/test_config_inference.py`
   for `AnvilCheckpointConfig` (round-trip through `read_checkpoint_config`,
   old-checkpoint-missing-`observation_encoding` defaulting).
4. **Byte-identical YAML round-trip**: `ConfigLoader.from_yaml` on every
   config under `configs/mcap_converter/v1.1/*.yaml`, before/after, produces
   an identical `DataConfig`; `to_yaml` (now calling the extracted `to_dict`
   free function) produces identical output.
5. Full `uv run python -m pytest tests/unit -q` green (baseline: 642 passed,
   2 skipped).
6. Re-run the `ee_delta`/`ee_delta_rot6d` smoke scenarios
   (`--scenario ee_delta,ee_delta_rot6d --select 1,2 --force`) to confirm the
   real convert→train path is unaffected end-to-end.
7. If item 5 (`patches.py`) lands: train one checkpoint, confirm
   `anvil_config.json` now contains `observation_encoding`.

## Notes
- All work in worktree `.worktrees/implement-ee-space`
  (branch `patrick/implement-ee-space`); commit per the git-worktree rule.
- `mcap_converter/config/loader.py`'s `ConfigLoader` (hydration, migration
  dispatch, strict unknown-key rejection) and `versioning.py`'s migration
  engine are **untouched in logic** — only their imports change to source
  types from `anvil_shared`. No behavior change, no new risk to the
  converter's authoring-time correctness.
- Two explicit judgment calls flagged above for your input:
  `RECOGNIZED_YAML_KEYS`'s home (with the schema vs. with the loader), and
  whether `AnvilCheckpointConfig` gets a `from_dict`/`to_dict` classmethod
  pair or stays members-only with all dict-mapping logic in the sibling
  `checkpoint_config.py`.
