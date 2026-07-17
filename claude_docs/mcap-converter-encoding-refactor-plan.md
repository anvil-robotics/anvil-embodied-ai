# Plan — mcap_converter encoding cleanup: `action_encoding` rename + multi-mode `observation_encoding`

## Status

**Design only — nothing in this document has been implemented.** Written in response to
three pieces of feedback on the current `ee_action_encoding`/`is_ee_delta` design (see
`claude_docs/ee-delta-architecture-report.md`, "Known bugs, gaps, and rough edges" #12,
#16, #19 for the rough edges this plan directly addresses). Scope for the implementation
pass, as decided:

1. Rename `ee_action_encoding` → `action_encoding`; add a reserved (not-yet-implemented)
   `"relative"` value.
2. Consolidate mcap_converter's own `is_ee`/`is_ee_delta`/`space_suffix` branching into a
   single source of truth **within `packages/mcap_converter/` only** — `anvil_trainer`,
   the ROS2 inference stack, and `anvil_eval` are explicitly OUT of scope for this pass
   and keep their own `is_ee_relative`/`is_ee_abs`/`is_ee_delta` boolean soup as-is (that is
   a separate, larger, cross-package follow-up, not attempted here).
3. Make `observation.state`'s rotation representation configurable: `quaternion` (current
   default, byte-identical), `rot6d`, or `axis_angle` — all three actually implemented this
   pass, not just plumbed-and-stubbed.
4. Re-anchor the authority relationship between `DataConfig`, `ConfigLoader`, and
   `validators.py`: `DataConfig` becomes the self-validating primary entity; `ConfigLoader`
   is demoted to pure YAML-shape hydration plus unrecognized-key rejection.
   **Included in this pass:** `anvil_eval/gt_replay.py` and
   `mcap_converter/utils/debug_plot.py` — both currently hand-parse
   `conversion_config.yaml` with ad hoc `yaml.safe_load` + `.get(...)` calls, bypassing
   `ConfigLoader`/`DataConfig` entirely — are rewritten to reconstruct a real `DataConfig`
   via `ConfigLoader.from_yaml()`. **Explicitly deferred, NOT touched this pass:**
   `anvil_eval_ros/src/anvil_eval_ros/cli.py`'s `_find_conversion_config`/
   `_detect_arms_from_conversion_config` — a third, much larger ad hoc
   `conversion_config.yaml` reader (arm detection, `action_topics`/`observation_topics`
   inspection, `action_from_observation` detection) that substantially overlaps with what
   `ConfigLoader.from_dict` already does, but rewriting it means restructuring
   `anvil_eval_ros/cli.py` to consume a `DataConfig` object instead of a raw dict — real
   work, and squarely inside the cross-package follow-up already deferred by decision #2
   above. Recorded as a memory (`project_mcap_converter_config_refactor`, see the auto-memory
   system) so this isn't lost before that follow-up pass happens.
5. **Explicit schema versioning for `DataConfig`/`conversion_config.yaml`** (Part 0b) —
   triggered by the `action_encoding` rename's own backward-compatibility gap (see Part 0b's
   opening), but designed as a durable, general-purpose mechanism: a version-tagged
   migration registry, a serializer to write configs back out at the current version, and
   a new `dataset-config-migrate` CLI. **This supersedes Part 0's earlier "rejected: a
   schema_version field" conclusion** — that conclusion was correct for the narrower
   strict/lenient-only problem it was written against, but a real gap surfaced afterward
   (§ Part 0b) that strict/lenient alone cannot fix, and this is the resolution.

Every fact below (line numbers, function signatures, current behavior) was re-verified
against the current working tree at the time of writing, not assumed from memory.

---

## Part 0 — `DataConfig` as the primary, self-validating entity

### Current state (verified)

`ConfigLoader.from_dict()` (loader.py:141-193) does per-field `config_dict.get(key,
default)` extraction with no check at all for keys in the YAML that aren't recognized by any
field — **an unknown or misspelled top-level key is silently dropped, not rejected.** This
is not hypothetical: it is the confirmed root cause of one of the four broken legacy configs
found during the architecture-report audit. `configs/mcap_converter/openarm_bimanual.yaml`
uses the pre-unification legacy schema (singular `robot_state_topic` field, topic-keyed
`action_topics`); today it "loads successfully" — every legacy key is silently ignored,
`observation_topics`/`action_topics` both collapse to their empty-dict defaults — and only
fails two steps later, at `validate_config`'s generic "observation_topics cannot be empty"
check, which gives no hint that the actual problem is an unrecognized/legacy key. Strict
unknown-key rejection at load time would have surfaced this immediately, with the actual
offending key name in the error message, instead of a confusing secondary symptom.

Validation logic is currently split across two files with two different exception types:
`loader.py:145-156` inline-validates `data_space`/`ee_action_encoding` and raises bare
`ValueError`; `validators.py:validate_config()` (44-127) re-validates the same two fields
(among others) and raises `ConfigurationError`. Every `DataConfig` field already has a
default today (verified — every field in schema.py:113-142 has either a literal default or
a `field(default_factory=...)`), so that part of the ask is already satisfied; no schema
change needed there.

### Design

**`ConfigLoader` is demoted to hydration + gatekeeping only:**
- Parses YAML → dict → typed field values (turning nested dicts into `ActionTopicSpec`/
  `JointNamePattern`/`FeatureMapping` instances) — this part is unchanged, it's genuinely a
  parsing concern.
- **New:** rejects any top-level YAML key not in an explicit, maintained allowlist of
  recognized keys. The allowlist must be YAML-key-level, not `dataclasses.fields(DataConfig)`
  names directly — some fields accept more than one YAML spelling today (`joint_names` is
  accepted as an alias for the `joint_name_pattern` field, loader.py:166), so the allowlist
  has to enumerate every accepted YAML key including aliases, not just derive one
  mechanically from the dataclass.
- No longer validates *values* (no more inline `data_space`/`ee_action_encoding` checks) —
  only structural/type errors belong here (e.g. "`action_topics` must be a mapping",
  "`topic` must be a non-empty string") — these are about malformed YAML *shape*, which is
  this layer's actual job; whether a given (well-shaped) value is a *legal* value is
  `DataConfig`'s job, not the loader's.

**`DataConfig` gains a `validate()` method, absorbing all of `validators.py:validate_config()`'s
logic** (`data_space` legality, `ee_action_encoding`/`action_encoding` legality and its EE-only
restriction, `action_topics` emptiness in EE mode, camera topic/mapping consistency,
`image_resolution` shape, feature-mapping field legality) — one method, one exception type
(`ConfigurationError`), replacing the two-file/two-exception-type split. **Deliberately NOT
auto-invoked from `__post_init__`:** several existing unit tests construct a deliberately
invalid `DataConfig` and then assert that validation catches it — auto-validating at
construction time would break that pattern (the invalid object could never be constructed to
test against in the first place). The call site doesn't change: `convert.py` still calls
`config.validate()` explicitly right after construction, exactly where it calls
`validate_config(config)` today — only the logic's *home* moves, not *when* it runs.

**`validate_topics_exist()` (validators.py:130-156) stays a standalone function, not a
`DataConfig` method.** It checks a config against externally-supplied MCAP runtime topics —
a cross-check against data outside the config, not "is this config internally well-formed."
Different concern, correctly kept separate.

**Implementation note — avoiding a circular import:** `ConfigurationError` is currently
defined in `validators.py` (validators.py:8-9), which imports `DataConfig` FROM `schema.py`.
Once `DataConfig.validate()` needs to raise `ConfigurationError` itself, `schema.py` cannot
import it back from `validators.py` without a cycle. Resolution: `ConfigurationError`, plus
the two small helpers `validate_config()` currently delegates to
(`validate_joint_name_pattern`, `validate_feature_mapping` — both validate sub-objects
`JointNamePattern`/`FeatureMapping` that are already defined in `schema.py`, so this is
their natural home anyway), all move into `schema.py` alongside `DataConfig`. `validators.py`
shrinks to just `validate_topics_exist()`, importing `DataConfig`/`ConfigurationError` FROM
`schema.py` as it already does today for `DataConfig` — no cycle either direction. Flagging
this now so it isn't discovered mid-implementation.

**Downstream consumers of an already-converted dataset must reconstruct a real `DataConfig`
via `ConfigLoader.from_yaml(dataset_root / "conversion_config.yaml")`, not hand-parse the
YAML.** Three files do this today; this pass fixes two of them:
- `anvil_eval/gt_replay.py:_detect_encoding()` (currently: raw `yaml.safe_load` +
  `cfg.get("ee_action_encoding", "absolute")`) → becomes a call through the **lenient**
  loader path (see below), then reads `.action_encoding` off the resulting `DataConfig`
  (the missing-file fallback behavior — default to `"absolute"` with a warning — is
  preserved, just moved to wrap the loader call instead of a raw dict `.get`).
- `mcap_converter/utils/debug_plot.py` — **correction made during implementation**: this
  plan originally assumed this file reads a task-name string relevant to the rename. On
  actually reading it, it reads `action_from_observation_n` — a legacy joint-mode
  act-from-obs debug knob that isn't part of `DataConfig` at all (pre-unification, EE-
  unrelated). Forcing it through `ConfigLoader`/`DataConfig` doesn't fit (that field has no
  home on `DataConfig` to read it from) and isn't needed for this rename. **Left unchanged**
  — flagging the original assumption as wrong rather than silently updating this bullet as
  if it had always said so.
- `anvil_eval_ros/src/anvil_eval_ros/cli.py`'s `_find_conversion_config`/
  `_detect_arms_from_conversion_config` — **explicitly NOT touched this pass** (see the
  Status section decision above and the memory recorded for it). It reads several fields
  (`action_topics`, `observation_topics`, `action_from_observation`) via its own independent
  logic, has its own multi-candidate-path search (`_find_conversion_config`, three fallback
  locations) that has no equivalent in `ConfigLoader` today, and would need real
  restructuring — not a drop-in swap — to consume a `DataConfig` object. Left as-is,
  tracked as follow-up.

### Strict vs. lenient hydration — required to avoid a real regression (added after review)

The unknown-top-level-key rejection above is correct and wanted for `convert.py`'s own
load path — a user actively authoring/editing a config should get an immediate, specific
error for a typo or a stale pre-unification key (this is the exact mechanism that would
have caught `openarm_bimanual.yaml`'s legacy `robot_state_topic` field immediately, instead
of two steps later as a confusing "observation_topics cannot be empty").

But applying that same strictness uniformly to the "read back an already-converted
dataset's frozen `conversion_config.yaml`" path would be a regression, not an improvement:
**a dataset converted at ANY point in this project's history** — potentially using an even
older config shape than today's, long before the unified schema existed — has its
originally-used config copied verbatim into its own `conversion_config.yaml`
(`convert.py:307-309`). Today's ad hoc single-key `yaml.safe_load` reads in `gt_replay.py`/
`debug_plot.py` don't care what shape the rest of that file is in; routing them through a
strict, unknown-key-rejecting loader would make GT-replay/debug-plot *more* fragile against
old datasets than they are today, not less.

**Resolution: two hydration modes, not a schema version number.** `ConfigLoader` gets a
`strict: bool` parameter (default `True`):
- `strict=True` (used by `convert.py`, the only place actively authoring/validating a
  new config): unrecognized top-level keys raise, as designed above.
- `strict=False` (used only by the two "read back a frozen historical file" call sites):
  unrecognized top-level keys are silently ignored during hydration — this is reading a
  historical record to peek at one or two fields, not validating that the file is a fully
  legal, convertible config. **A `DataConfig` built via the lenient path must never be
  passed to `.validate()`** — there is no reason to demand full schema legality from an old
  file just to read `action_encoding` or a task-name string off it, and doing so would
  reintroduce the exact fragility this split exists to avoid.
- Missing keys are unaffected by `strict` either way — a key simply absent from the YAML
  always falls back to `DataConfig`'s own field default, in both modes. This is the
  correct, already-working behavior for "old config predates a newly-added field" and needs
  no detection logic at all: every field addition to `DataConfig` so far (`ee_action_encoding`
  originally, now `action_encoding`/`observation_encoding`) was deliberately designed with a
  default that reproduces prior behavior exactly, so an old file missing the key and a new
  file that explicitly sets the value to the default are, correctly, indistinguishable and
  behave identically.

**Originally considered and rejected here, later reversed — see Part 0b.** This section
initially concluded no `schema_version` field was needed, reasoning that every field
addition so far had a behavior-preserving default so strict/lenient hydration alone was
sufficient. That reasoning missed a real case: **the rename itself** (`ee_action_encoding` →
`action_encoding`) is not "a field with a new default," it's "an existing, meaningful,
already-on-disk VALUE now unreachable under its old key name." Under lenient hydration, a
pre-rename dataset's `conversion_config.yaml` (e.g. one written with
`ee_action_encoding: "delta"`) has that key silently ignored as "unrecognized," and
`action_encoding` falls back to its *default* (`"absolute"`) — not because that's the
correct historical value, but because the loader can no longer find it under the old name.
This is a silent wrong-answer, not a safely-reproduced old behavior, and strict/lenient
hydration cannot distinguish the two cases on its own. Part 0b below is the resolution:
an explicit, versioned migration mechanism, designed as a durable general-purpose system
rather than a one-off patch for this single rename.

---

## Part 0b — Schema versioning mechanism

**Correction made after implementation review (important — read before anything below):**
this section originally, incorrectly described v1.0 as "the schema before
`action_encoding`/`observation_encoding` existed," without checking `main`. That is wrong.
Verified by reading `main`'s actual `configs/mcap_converter/*.yaml` directly: `main` has NO
`data_space`, NO `observation_topics`, NO EE support of any kind — a pure joint-only,
pre-unification format (singular `robot_state_topic`, topic-keyed `action_topics`,
`action_from_observation[_n]`). **v1.0 = `main`'s actual schema. v1.1 = the current
`implement-ee-space` tip** (the unified joint+EE schema, including this session's
`action_encoding`/`observation_encoding` work) — this was the original intent from the very
start of this design conversation and was mis-implemented; now fixed. See
`config/versioning.py`'s module docstring for the exact real sub-shapes found on `main`
(leader-follower, quest+real-command-topics, quest+`action_from_observation`) and how each
is migrated — this required extending the registry with a `custom` transform callable
(§2 below), since restructuring topic-keyed `action_topics` into arm-keyed, and deriving
`observation_topics` from `robot_state_topic` + `joint_names.arms`, aren't expressible as
simple field renames.

### The gap this closes (verified against the actual failure mode, not hypothetical)

Confirmed by tracing it through: after the Part 1 rename, any dataset converted **before**
this change has a `conversion_config.yaml` containing `ee_action_encoding`, not
`action_encoding`. Read via the lenient path from Part 0 (`ConfigLoader.from_yaml(path,
strict=False)`, used by `gt_replay.py`/`debug_plot.py`), the old key is treated as just
another unrecognized key and dropped; `action_encoding` is absent from the dict entirely
under its new name, so it takes the class default `"absolute"`. If that dataset was actually
converted with `ee_action_encoding: "delta"`, `gt_replay.py`'s `--encoding auto` would now
silently pick the wrong validation branch — no error, no warning, just a wrong answer feeding
a tool whose entire job is being a strict correctness gate. This is a defect the rename
itself introduces, not a pre-existing one, and strict/lenient hydration (which only decides
whether to *complain* about unrecognized keys, not what they *mean*) cannot fix it.

### 1 — Where `schema_version` lives; collision check

`schema_version: str = CURRENT_SCHEMA_VERSION` becomes a real field on `DataConfig`
(schema.py), not a side-channel value threaded separately. Confirmed no collision: no field
named `schema_version` or `version` exists anywhere in schema.py today. A freshly
constructed `DataConfig()` (tests, `ConfigLoader.get_default()`) is current-version by
construction with zero special-casing, since the field's default IS
`CURRENT_SCHEMA_VERSION`.

Stored as a bare string, no `"v"` prefix, in the YAML/field value itself (`schema_version:
"1.1"`) — the `"v"` prefix appears only in human-readable messages and backup filenames
(`conversion_config_v1.0.yaml`), matching the convention implied by the request. Versions
are treated as **opaque strings looked up by exact match in a registry chain**, not
numerically compared (no tuple-of-ints, no `packaging.version`) — this is sufficient because
the registry only ever needs "is there a step whose `from_version` equals this string,"
never "is 1.2 newer than 1.1" in the abstract. Flagging this as a deliberate simplification.

New module: `packages/mcap_converter/src/mcap_converter/config/versioning.py` — houses
the migration data structures, the registry, and `migrate_to_current()`. Kept separate
from `schema.py`/`encodings.py` since this is a distinct subsystem (registry +
sequential-application logic), not a value table.

**Architecture-consistency correction (caught in review, fixed):** `CURRENT_SCHEMA_VERSION`
itself is defined in `schema.py`, not here — it's `DataConfig.schema_version`'s own field
default, so `schema.py` (the primary, authoritative entity per Part 0) should own it;
`versioning.py` imports it FROM `schema.py`. This also fully eliminates the
"avoiding a circular import" lazy-import workaround from the original draft of this
section — since `schema.py` no longer needs anything from `versioning.py` at module level,
`versioning.py` can import `ConfigurationError`/`CURRENT_SCHEMA_VERSION` from `schema.py`
directly at the top of the file, no cycle either direction. The same principle applies to
the loader's unknown-key allowlist: it's now `RECOGNIZED_YAML_KEYS`, defined in `schema.py`
right next to `DataConfig` (it's a direct description of that class's own field set), with
`loader.py` importing it rather than maintaining its own copy.

### 2 — The migration registry (satisfies Gaps 5 & 6), and its boundary with strict/lenient (Gap 1)

`FieldMigration` (rename or pure removal) covers simple key-level changes declaratively.
**Extended after implementation review**: real v1.0 → v1.1 migration needs more than that
— `main`'s actual legacy shapes require restructuring (topic-keyed `action_topics` inverted
to arm-keyed; `observation_topics` derived from `robot_state_topic` + `joint_names.arms`;
`action_from_observation` dropped with no field-level replacement) that a declarative
rename/removal pair cannot express. `VersionMigration` gained a `custom` field — an
optional callable applied after `field_migrations`, for exactly this kind of structural
transform:

```python
# config/versioning.py

@dataclass(frozen=True)
class FieldMigration:
    """One simple field-level change: a straight rename or a pure removal.
    new_key set   -> rename (old_key's value moves to new_key)
    new_key=None  -> pure removal (old_key dropped, no replacement)
    """
    old_key: str
    new_key: Optional[str]
    reason: str  # human-readable; always logged when triggered

@dataclass(frozen=True)
class VersionMigration:
    from_version: str
    to_version: str
    field_migrations: Tuple[FieldMigration, ...] = ()
    custom: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None  # for changes
    # that aren't expressible as key rename/removal — see _migrate_legacy_v1_0_shape

MIGRATIONS: Tuple[VersionMigration, ...] = (
    VersionMigration(
        from_version="1.0",
        to_version="1.1",
        field_migrations=(
            FieldMigration(
                old_key="ee_action_encoding",
                new_key="action_encoding",
                reason="renamed for generality (joint-space encoding support planned) — "
                       "see claude_docs/mcap-converter-encoding-refactor-plan.md",
            ),
        ),
        custom=_migrate_legacy_v1_0_shape,  # main's real structural differences — see below
    ),
    # Future: VersionMigration(from_version="1.1", to_version="1.2", field_migrations=(...))
)
```

`_migrate_legacy_v1_0_shape(raw)` is the real payload — it's a no-op unless
`robot_state_topic` is present AND `observation_topics` is absent (i.e., genuinely
`main`-shaped input, not the branch-internal pre-rename case, which already has
`observation_topics`). When it fires: derives `observation_topics` from
`robot_state_topic` + `joint_names.arms.values()`; adds `data_space: "joint"`; inverts
topic-keyed `action_topics` to arm-keyed when the values look topic-keyed (each value is a
dict containing an `arm` key); drops `action_from_observation[_n]` (no field-level
equivalent in the current schema) — and when `action_from_observation: true` co-occurred
with real `action_topics` (an ambiguous case: `main`'s semantics there were "fall back to
observation only if the command topic wasn't actually recorded in this specific MCAP," a
per-recording runtime condition with no static equivalent), logs an explicit warning and
keeps the real `action_topics` (strict behavior) rather than silently guessing. Verified
against all three real, distinct shapes found by reading `main`'s actual config files
directly (leader-follower, quest+real-topics, quest+`action_from_observation`) — see the
test suite additions in Sequencing step 13.

```python
def detected_version(raw: Dict[str, Any]) -> str:
    return str(raw.get("schema_version", "1.0"))  # absence == v1.0, the rule


def migrate_to_current(
    raw: Dict[str, Any],
    *,
    registry: Tuple[VersionMigration, ...] = MIGRATIONS,
    current_version: str = CURRENT_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Applies the registered chain from raw's detected version up to current_version.
    Returns a NEW dict; never mutates raw. A true no-op (only stamps schema_version, no
    field changes) when already current — the while loop below never executes its body."""
    out = dict(raw)
    version = detected_version(out)
    by_from = {m.from_version: m for m in registry}
    while version != current_version:
        step = by_from.get(version)
        if step is None:
            raise ConfigurationError(
                f"No migration registered from schema_version={version!r} to "
                f"{current_version!r}; cannot upgrade this config."
            )
        out = _apply_version_migration(out, step)  # field_migrations, then step.custom
        version = step.to_version
    out["schema_version"] = current_version
    return out


def _apply_field_migrations(raw: Dict[str, Any], step: VersionMigration) -> Dict[str, Any]:
    out = dict(raw)
    for fm in step.field_migrations:
        if fm.old_key not in out:
            continue
        old_value = out.pop(fm.old_key)
        if fm.new_key is None:
            log.warning("[config-migrate] %s->%s: dropping deprecated field %r (%s)",
                        step.from_version, step.to_version, fm.old_key, fm.reason)
            continue
        if fm.new_key in out:
            log.warning(
                "[config-migrate] %s: both legacy %r and current %r present — legacy "
                "value wins (migration-only conflict rule, see Gap 1 boundary below)",
                step.from_version, fm.old_key, fm.new_key,
            )
        else:
            log.warning(
                "[config-migrate] Upgrading from v%s: field %r is deprecated, use %r (%s)",
                step.from_version, fm.old_key, fm.new_key, fm.reason,
            )
        out[fm.new_key] = old_value
    return out
```

**N-step chaining (Gap 5):** the `while` loop applies as many registered steps as needed,
in sequence, transparent to the caller — a config several versions behind would walk
`1.0→1.1→1.2→...` without the caller ever needing to know how many hops occurred. The
`registry`/`current_version` keyword parameters exist specifically so a test can inject a
*synthetic* second step (`VersionMigration(from_version="1.1", to_version="1.2",
field_migrations=(FieldMigration("old_thing", "new_thing", "synthetic test step"),))`) and
assert two-step chaining end-to-end, without ever adding a fake entry to the real, production
`MIGRATIONS` tuple. Required test, not optional: prove chaining works before it's ever
needed for a real second migration.

**The Gap 1 boundary, stated exactly:** the "old key wins on conflict" rule lives *only*
inside `_apply_field_migrations`, which is *only* ever called from `migrate_to_current`'s
`while` loop, which *only* executes when `_detected_version(raw) != current_version`. There
is no separate flag gating this — it falls out of control flow. **If `schema_version`
already equals current, the while loop's condition is false immediately and the function
body never runs.** Any leftover old-named key in a file that claims to be current-version is
therefore never touched by this mechanism — it falls straight through to the strict/lenient
unknown-key check from Part 0, unmodified. The two mechanisms have non-overlapping,
unambiguous trigger conditions: migration owns *version-tracked* key changes; strict/lenient
owns everything else (typos, truly-removed legacy shapes with no migration entry).

### Composition with `strict`/`lenient` — sequencing

`migrate_to_current` is inserted as the first line of `ConfigLoader.from_dict`, before
anything else, **regardless of `strict`**:

```python
@staticmethod
def from_dict(config_dict: Dict[str, Any], strict: bool = True) -> DataConfig:
    config_dict = migrate_to_current(config_dict)   # NEW — always runs, both modes
    defaults = DataConfig()
    if strict:
        _reject_unknown_keys(config_dict)            # Part 0's unknown-key check
    ...
```

Consequence worth stating explicitly: migration applies uniformly to **both** `convert.py`'s
strict path and `gt_replay.py`/`debug_plot.py`'s lenient path. An actively-maintained input
config still using `ee_action_encoding` today converts successfully (with a deprecation
warning logged, not an error) rather than being forced into an immediate manual edit —
matching this codebase's existing "permanent alias, never a breaking change without a
transparent path" philosophy already established for the `ee_rel`→`ee_relative` string
rename. `strict`/`lenient` continues to govern only genuinely-unrecognized, non-versioned
keys; migration owns version-tracked field renames exclusively, in both modes.

**Confirms the fix for the original triggering bug (item 3 of the request):** an old
dataset's `conversion_config.yaml` with no `schema_version` and
`ee_action_encoding: "delta"`, read via `ConfigLoader.from_yaml(path, strict=False)`, now
resolves correctly: `migrate_to_current` detects `"1.0"`, applies the registered
`1.0→1.1` step, renames the key (value `"delta"` preserved), stamps `schema_version="1.1"`
→ `DataConfig.action_encoding == "delta"`, correctly. This is the first, and currently only,
real entry in `MIGRATIONS`.

### 3 — Serialization + migration CLI

**`DataConfig.to_dict()`** (schema.py, new method) — the canonical current-version
YAML-serializable representation, replacing `convert.py`'s ad hoc `config_to_save` dict
(convert.py:311-324) which — confirmed by re-reading it — **never included
`ee_action_encoding` at all**, the exact "conversion_config.yaml re-serialization drops the
field" rough edge from the original architecture report:

```python
def to_dict(self) -> Dict[str, Any]:
    out = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "data_space": self.data_space,
        "observation_topics": dict(self.observation_topics),
        "action_topics": {
            arm_id: {"topic": s.topic, "joint_order": list(s.joint_order)}
            for arm_id, s in self.action_topics.items()
        },
        "action_encoding": self.action_encoding,
        "observation_encoding": self.observation_encoding,
        "camera_topics": list(self.camera_topics),
        "camera_topic_mapping": dict(self.camera_topic_mapping),
        "image_resolution": list(self.image_resolution),
    }
    if not self.is_ee:  # matches convert.py's existing "omit for EE" behavior
        out["joint_names"] = {
            "separator": self.joint_name_pattern.separator,
            "source": self.joint_name_pattern.source,
            "arms": self.joint_name_pattern.arms,
        }
    return out
```

**`ConfigLoader.to_yaml(config, path)`** (loader.py, new method, mirrors `from_yaml`):

```python
@staticmethod
def to_yaml(config: DataConfig, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
```

`convert.py`'s conversion-config-saving branch (the `else` at line ~311) collapses to
`ConfigLoader.to_yaml(config, conversion_config_dest)` — fixing the missing-field rough edge
at the source, using the same code path the migration CLI uses.

**CLI: `dataset-config-migrate`.** Naming follows this package's existing `<domain>-<verb>`
console-script convention (`mcap-convert`, `mcap-inspect`, `dataset-valid`, `hf-upload`,
`mcap-to-video`, `merge-datasets`) — `dataset-` since this operates on an already-converted
dataset's config, not raw MCAP, matching `dataset-valid`'s prefix. Entry point
`mcap_converter.cli.migrate_config:main`, new file
`packages/mcap_converter/src/mcap_converter/cli/migrate_config.py`, new
`pyproject.toml` console-script line.

```
dataset-config-migrate --dataset <path-to-dataset-dir> [--force]
```

(`--dataset` matches `gt_replay.py`'s own flag name for the same kind of path.)

Exact flow:

1. `cfg_path = Path(args.dataset) / "conversion_config.yaml"`. Missing → error, exit 1, no
   file ops.
2. `raw = ConfigLoader.load_yaml(cfg_path)`; `detected = _detected_version(raw)`.
3. **Gap 4 — true no-op:** `if detected == CURRENT_SCHEMA_VERSION:` print `"Already at
   current schema version (v{detected}) — nothing to do."`, exit 0. **Zero file
   operations** — checked before anything else touches the filesystem; no re-serialize, no
   rename, no write, even with identical content (a rewrite would still touch mtime and
   could trigger unrelated downstream staleness logic — explicitly avoided).
4. **Gap 3 — reject-by-default on backup collision:** `backup_path =
   cfg_path.with_name(f"conversion_config_v{detected}.yaml")`. If it exists and `not
   args.force`: print `"{backup_path} already exists — this directory appears to have
   already been migrated. Use --force to overwrite the existing backup."`, exit 1, no file
   ops. Implemented, not left as an open question.
5. **Gap 2 — confirmation, independent of `--force`:** print the exact plan —
   ```
   This will:
     1. Rename {cfg_path} -> {backup_path}
     2. Write the upgraded config (schema v{CURRENT_SCHEMA_VERSION}) to {cfg_path}
   Proceed? [y/N]:
   ```
   via `input()`; only an explicit `y`/`yes` (case-insensitive) proceeds — anything else,
   including bare Enter, aborts with no file ops. **`--force` does NOT skip this prompt** —
   its only effect is step 4's overwrite permission. This is a real, non-obvious design
   choice (an alternative reading of "--force" could mean "skip all friction including the
   prompt") — stated explicitly rather than assumed; revisit if batch/scripted migration of
   many datasets turns out to need a fully-non-interactive mode later.
6. On confirmation: `cfg_path.rename(backup_path)` → `upgraded = ConfigLoader.from_dict(raw,
   strict=False)` (lenient — the migration tool's job is upgrading the version, not
   policing unrelated stray keys; any truly-unknown key in the old file is simply dropped
   from the rewritten output, the desired "clean upgrade" behavior) →
   `ConfigLoader.to_yaml(upgraded, cfg_path)` → print `"Migrated {cfg_path}: v{detected} ->
   v{CURRENT_SCHEMA_VERSION}. Original backed up at {backup_path}."`, exit 0.

Any tool that auto-discovers `conversion_config.yaml` by its canonical name
(`anvil_eval_ros/cli.py`'s `_find_conversion_config`, deferred but unmodified — see the
recorded memory) transparently picks up the upgraded file with zero changes to its own
lookup logic.

### 4 — Confirmed: Gaps 5 and 6 are both satisfiable by the one `FieldMigration`/`VersionMigration` shape above

No restructuring needed for a future pure-removal case (`new_key=None`) or a future N-step
chain (the `while` loop in `migrate_to_current` already walks arbitrarily many registered
steps) — both are already expressible in the registry design as written.

---

## Part 1 — `action_encoding` rename + reserved `"relative"` value

### Current state (verified)

The field is called `ee_action_encoding` and is validated in three independent places,
each with its own hardcoded `("absolute", "delta")` tuple and its own exception type:

- `schema.py:124` — the field itself, `ee_action_encoding: str = "absolute"`.
- `loader.py:150-156` — parses + validates, raises bare `ValueError`.
- `validators.py:88-98` — re-validates + additionally rejects the value when
  `data_space != "ee"`, raises `ConfigurationError`.

Because `loader.py`'s check always runs first in the real call chain (`ConfigLoader.from_yaml`
is called before `validate_config` in `convert.py`), the `validators.py` branch for the
valid-value check is currently unreachable except when a caller constructs `DataConfig`
directly and calls `validate_config` without going through `ConfigLoader` — which some unit
tests do deliberately. This duplication is exactly the kind of thing that drifts silently
if a value is ever added to one tuple and not the other.

Outside `packages/mcap_converter/`, exactly one file reads the YAML key at runtime:
`packages/anvil_eval/src/anvil_eval/gt_replay.py`'s `_detect_encoding()` (reads
`conversion_config.yaml`'s `ee_action_encoding` key directly, for `--encoding auto`). Five
more files reference the string only in prose/docstrings explaining provenance
(`anvil_trainer/config.py:79,119`, `train.py:194`, `transforms.py:280`, `patches.py:457`) —
these do not execute against the field, so they will not break, but their comments will
become stale if left unedited.

### Design

- Rename the field: `ee_action_encoding` → `action_encoding` (schema.py, loader.py,
  validators.py, and the four docstring-only mentions above for consistency — the
  docstring updates are cosmetic but cheap, do them in the same pass to avoid leaving
  stale references pointing at a field name that no longer exists).
- Introduce one canonical list, defined once, imported everywhere it's checked (new module,
  see Part 2): `VALID_ACTION_ENCODINGS = ("absolute", "delta", "relative")` and
  `IMPLEMENTED_ACTION_ENCODINGS = ("absolute", "delta")`. `loader.py` and `validators.py`
  both import these instead of hardcoding the tuple — this alone kills the
  duplicated-validation rough edge from the architecture report.
- Validation becomes two distinct checks with two distinct messages:
  1. `action_encoding not in VALID_ACTION_ENCODINGS` → "unknown value" error (existing
     behavior, just against the shared constant).
  2. `action_encoding in VALID_ACTION_ENCODINGS and action_encoding not in
     IMPLEMENTED_ACTION_ENCODINGS` (i.e. exactly `"relative"` today) → a **distinct** error:
     `"action_encoding='relative' is reserved for future use and is not yet implemented in
     mcap_converter."` This is deliberately louder and more specific than a generic
     "invalid value" message — selecting a reserved-but-unimplemented value is a different
     failure mode from a typo, and should read differently in the terminal.
- **Scope restriction unchanged, decision made explicit rather than left implicit:** the
  existing rule "`action_encoding` must be `\"absolute\"` when `data_space != \"ee\"`"
  (validators.py:93-98) stays as-is. The field is deliberately renamed away from the `ee_`
  prefix specifically so that when joint-space encoding support is actually designed later,
  only this one validator restriction needs loosening — no second rename. Nothing for joint
  mode is implemented in this pass; the rename prepares the name, not the behavior.
- **Required consequential fix, outside the "mcap_converter only" scope but load-bearing:**
  `anvil_eval/gt_replay.py`'s `_detect_encoding()` reads this field from
  `conversion_config.yaml` today, so it must track the rename. Per Part 0's redesign,
  `_detect_encoding` is rewritten to call `ConfigLoader.from_yaml(path, strict=False)` and
  read `.action_encoding` off the resulting `DataConfig`, same as
  `mcap_converter/utils/debug_plot.py`. **This alone is not sufficient** — reading an old
  dataset's `conversion_config.yaml` (which still says `ee_action_encoding`, not
  `action_encoding`) through the lenient loader would silently default to `"absolute"`
  rather than recovering the actual historical value, since a rename isn't a "missing key
  with a safe default," it's an existing value now unreachable under its old name. **Part
  0b's schema-versioning mechanism is the actual fix** — the old key is recognized and
  migrated by version, not silently dropped as an unrecognized key.

### What does NOT change in Part 1

`DataConfig.is_ee` stays as-is (single-purpose, correctly named, not part of the mess).
The `EERelativeTransform`/`ee_relative_forward`/training-side "relative" (n-0) mechanism is
a completely different, already-shipped concept at a different layer (anvil_trainer) — the
new converter-level `action_encoding="relative"` reserved value is NOT assumed to mean the
same thing as that mechanism; if/when it is implemented, its semantics need their own design
pass, not an inherited assumption from the training-side name collision this branch already
had to resolve once.

---

## Part 2 — Consolidating mcap_converter's own `is_ee`/`is_ee_delta`/`space_suffix` soup

### Current state (verified)

Every current touch point, from a full grep of `packages/mcap_converter/src/`:

| Symbol | Defined | Read at |
|---|---|---|
| `DataConfig.is_ee` | schema.py:149-150 | writer.py:217; extractor.py:707,754,808,828,849,870,953; convert.py:252,326,520,792 |
| `DataConfig.is_ee_delta` | schema.py:153-155 | extractor.py:1411; convert.py:710 |
| `space_suffix` | inline ternary, convert.py:710 | convert.py:711+ (output path construction) |

`is_ee` is a clean, single-purpose, correctly-scoped boolean — it is not part of the
complaint and is left untouched. The actual mess is `space_suffix`: it is computed as an
inline ternary in `convert.py` (`"ee-delta-space" if config.is_ee_delta else
f"{config.data_space}-space"`) that re-derives knowledge already expressed by
`is_ee_delta` in a *different file*, and every future `action_encoding` value that needs
its own output subdirectory means editing `convert.py` again, in a location that has
nothing else to do with output-path policy. This is the literal shape of "logic for one
feature living in the wrong layer and growing ad hoc" — the fix is to move the decision to
where the data lives.

Separately, `writer.py:222-227` (state/action feature name suffixes) and
`extractor.py:1400-1406` (per-frame state/action encoding) both hardcode the same "8 dims
quaternion state / 10 dims rot6d action" layout independently — not part of the
`is_ee`/`is_ee_delta` complaint specifically, but the same root problem (one fact, encoded
in two places), and directly relevant to Part 3 below since it's about to need a third
variant (rot6d) and a fourth (axis-angle).

### Design

**New module: `packages/mcap_converter/src/mcap_converter/config/encodings.py`.** Single
home for every encoding-related constant and the one function that knows how to realize an
encoding choice. Nothing here is EE-math (that stays in `anvil_shared.rotation`, see Part
3) — this module is converter-domain policy: what values are legal, what they're called on
disk, what dimension they produce.

```python
# config/encodings.py

VALID_ACTION_ENCODINGS = ("absolute", "delta", "relative")
IMPLEMENTED_ACTION_ENCODINGS = ("absolute", "delta")

VALID_OBSERVATION_ENCODINGS = ("quaternion", "rot6d", "axis_angle")

# One row per encoding: (per-arm feature-name suffixes, dimension). Position (x,y,z) and
# gripper are invariant across all three and are NOT part of this table — this table
# describes only the rotation component.
OBSERVATION_ROTATION_LAYOUTS = {
    "quaternion": (("qx", "qy", "qz", "qw"), 4),
    "rot6d":      (("r0", "r1", "r2", "r3", "r4", "r5"), 6),
    "axis_angle": (("ax", "ay", "az"), 3),
}


def observation_state_dim_per_arm(observation_encoding: str) -> int:
    """3 (xyz) + rotation dim + 1 (gripper), for the given observation_encoding."""
    _, rot_dim = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    return 3 + rot_dim + 1


def observation_state_names_per_arm(observation_encoding: str) -> tuple[str, ...]:
    rot_names, _ = OBSERVATION_ROTATION_LAYOUTS[observation_encoding]
    return ("x", "y", "z", *rot_names, "gripper")
```

`writer.py:_define_features` and `extractor.py:_align_ee_signals` both import from this
module instead of each hardcoding their own copy of the layout — this is the concrete fix
for the "hardcoded independently in N places" pattern the architecture report flagged
repeatedly (not just for this exact layout, but this is the one instance actually in this
pass's scope).

**`DataConfig` gains one new property, replacing `convert.py`'s inline ternary:**

```python
# schema.py

@property
def is_action_delta(self) -> bool:
    """Renamed from is_ee_delta for consistency with the renamed action_encoding field."""
    return self.data_space == "ee" and self.action_encoding == "delta"

@property
def output_subdir(self) -> str:
    """Canonical <space>-space/ output directory name for this config.

    Single source of truth for convert.py's output path. A future action_encoding value
    that needs its own subdirectory is a one-line addition HERE, not a new branch in
    convert.py.
    """
    if self.is_action_delta:
        return "ee-delta-space"
    return f"{self.data_space}-space"
```

`convert.py:710` collapses to `space_suffix = config.output_subdir` — one attribute read,
no re-derivation of `is_ee_delta`'s logic in a second location. `extractor.py:1411`'s
`if self.config.is_ee_delta:` becomes `if self.config.is_action_delta:` (rename only, same
call site, same behavior).

**Explicitly not done in this pass:** no equivalent consolidation for `anvil_trainer`'s
`is_ee`/`is_ee_relative`/`is_ee_abs`/`is_ee_delta`, the ROS2 stack's two independent EE-mode
detectors, or `anvil_eval`'s parallel `is_ee` checks — all of these remain exactly as
documented in the architecture report. Attempting all four packages in one pass was
considered and explicitly rejected (see the scope decision at the top of this document) —
it is real, valuable follow-up work, but it is large enough (and touches enough
already-shipped-checkpoint-facing code) that it deserves its own dedicated design pass,
not a rider on this one.

---

## Part 3 — Configurable `observation.state` rotation encoding (`quaternion` / `rot6d` / `axis_angle`)

### Current state (verified)

`observation.state` is unconditionally quaternion on disk today —
`extractor.py:_align_ee_signals` (1392-1406) always builds `state_slices` as
`[pos, quat, gripper]`, and `writer.py:_define_features` (217-238) always declares 8 names
per arm (`x,y,z,qx,qy,qz,qw,gripper`). There is no config knob for this at all today.

**Critical interaction, verified by reading `anvil_shared.ee_transform.ee_delta_forward`
and its `n_arms_from_dims` helper directly:** `n_arms_from_dims` (ee_transform.py:65-86)
hardcodes `EE_STATE_DIM_PER_ARM = 8` as the *only* accepted state layout — it raises
`ValueError` if `state_dim` is not a positive multiple of 8. Both `ee_delta_forward` and
`ee_relative_forward` call this helper on whatever `state` array they're given. Today,
`extractor.py:1416`'s delta-anchor computation passes `prev_state`/`state_abs` — which is
always the quaternion-encoded `observation.state` — directly into `ee_delta_forward`. If
`observation.state` becomes rot6d- or axis-angle-encoded on disk, passing that same array
as the anchor would either raise `ValueError` (wrong dim: 10 or 7 is not a multiple of 8)
or, worse, silently misinterpret a differently-shaped array as if it were `n_arms` worth of
quaternion state. **This is not a hypothetical edge case — it is the direct, mechanical
consequence of shipping Part 3 without a corresponding change to how the delta anchor is
computed**, since `ee_transform.py` is explicitly out of scope for this pass (Part 2's
scope decision).

### Design

**Resolution of the interaction above: decouple "what anchors the delta math" from "what's
written to disk."** The raw buffered EE-pose sample (`_buffer_ee_pose`, extractor.py:1313)
is always quaternion — that never changes, since it's decoded straight from the ROS
message. `_align_ee_signals` will build **two** state arrays per call instead of one:

- `state_quat` — always quaternion, `(8 * n_arms,)`, used ONLY as the delta-anchor input to
  `ee_delta_forward`. Never written to the dataset directly unless
  `observation_encoding == "quaternion"` (in which case it happens to equal
  `state_encoded`, but that's incidental, not load-bearing).
- `state_encoded` — in whatever `observation_encoding` the config selects, this IS the
  `observation.state` feature written to disk.

This keeps `anvil_shared.ee_transform` completely untouched (satisfies the Part 2 scope
decision) while making `observation_encoding` fully independent of `action_encoding`.

**`_align_ee_signals`'s signature and return shape both change** (this is the one place in
this whole plan where an existing internal contract changes, not just an addition):

```python
def _align_ee_signals(
    self,
    ee_buffers: Dict[str, deque],
    target_ts: float,
    prev_state_quat: Optional[np.ndarray] = None,
) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
    """Returns (frame_dict, state_quat) — state_quat is NOT part of the dataset schema;
    callers thread it back in as next call's prev_state_quat. It is deliberately returned
    as a separate value, not smuggled into frame_dict and stripped out later, so nothing
    downstream can accidentally treat it as a real feature."""
    ...
    for arm_id in self.config.observation_topics:
        ...
        _, pos, quat, gripper = buffer[idx]
        state_quat_slices.append(np.concatenate([pos, quat, [gripper]]))
        rot_encoded = encode_rotation(quat, self.config.observation_encoding)  # Part 3 helper, see below
        state_encoded_slices.append(np.concatenate([pos, rot_encoded, [gripper]]))
        rot6d = matrix_to_rot6d(quat_to_matrix(quat))  # action stays rot6d always — unrelated to observation_encoding
        action_slices.append(np.concatenate([pos, rot6d, [gripper]]))

    state_quat = np.concatenate(state_quat_slices)
    state_encoded = np.concatenate(state_encoded_slices)
    action_abs = np.concatenate(action_slices)

    if self.config.is_action_delta:
        from anvil_shared.ee_transform import ee_delta_forward
        anchor = state_quat if prev_state_quat is None else prev_state_quat  # ALWAYS quaternion
        action_out = ee_delta_forward(action_abs, anchor)
    else:
        action_out = action_abs

    frame = {
        "observation.state": state_encoded.astype(np.float32),
        "action": action_out.astype(np.float32),
    }
    return frame, state_quat
```

`_align_frame_at_cursor` and `extract_frames` both need their `prev_ee_state` local renamed
to `prev_ee_state_quat` and rewired to come from the second element of this new return
tuple, at both yield sites (the main streaming loop, extractor.py:809, and the
buffer-flush tail, extractor.py:850 — same "two yield paths, both need the update" trap the
architecture report already flagged once for this exact variable; worth double-checking
both are updated when this lands).

**New `encode_rotation` dispatcher, in `config/encodings.py` alongside the layout table:**

```python
def encode_rotation(quat_xyzw: np.ndarray, observation_encoding: str) -> np.ndarray:
    """Encode a single quaternion sample into the selected observation rotation encoding."""
    if observation_encoding == "quaternion":
        return np.asarray(quat_xyzw, dtype=np.float64)
    from anvil_shared.rotation import quat_to_matrix, matrix_to_rot6d, matrix_to_axis_angle
    R = quat_to_matrix(quat_xyzw)
    if observation_encoding == "rot6d":
        return matrix_to_rot6d(R)
    if observation_encoding == "axis_angle":
        return matrix_to_axis_angle(R)
    raise ValueError(f"unknown observation_encoding: {observation_encoding!r}")
```

**`schema.py`:** new field `observation_encoding: str = "quaternion"` (default preserves
exact current behavior for every existing shipped config, none of which set this field).

**`loader.py` / `validators.py`:** parse + validate against
`encodings.VALID_OBSERVATION_ENCODINGS`, same two-tier pattern as `action_encoding` (all
three values are actually implemented this pass, so there is no
"valid-but-not-implemented" tier here — unlike `action_encoding`'s `"relative"`).

**`writer.py:_define_features`:** state feature names/shape built from
`encodings.observation_state_names_per_arm`/`observation_state_dim_per_arm` instead of the
current hardcoded 8-name tuple. Action feature stays exactly as-is (always 10-dim rot6d,
`action_encoding` and `observation_encoding` are independent knobs).

### New primitives required in `anvil_shared/rotation.py`

No axis-angle conversion exists anywhere in the repo today (`rotation.py` currently exports
only quaternion ↔ matrix ↔ rot6d). Two new functions, matching the file's existing style
exactly (hand-rolled numpy, no new dependency, scalar + batch variants, explicit degenerate-
input handling rather than silent wrong output):

- `matrix_to_axis_angle(R) -> np.ndarray`: standard Rodrigues extraction, `angle =
  arccos(clip((trace(R)-1)/2, -1, 1))`, axis from the skew-symmetric part
  `(1/(2 sin(angle))) * [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]`, return `axis *
  angle`. **Two degenerate regimes require explicit branches, not just an epsilon guard**:
  - `angle ≈ 0` (identity): return the zero vector directly — the axis is undefined but the
    output is unambiguous, no division needed.
  - `angle ≈ π`: `sin(angle) ≈ 0`, so the skew-symmetric extraction above is a 0/0 division
    — this is exactly the numerical regime the diagnosis doc
    (`claude_docs/ee-space-libero-vs-production-diagnosis.md`, §1.3) flagged as a historical
    axis-angle suspect on the LIBERO side (there, the concern was later disproven and turned
    out to be about a different representation's discontinuity — but the underlying
    "θ≈π needs a different extraction formula" fact about axis-angle itself is real and
    must actually be handled here, not hand-waved, since this pass is genuinely introducing
    axis-angle as a first-class on-disk encoding for the first time). At `angle ≈ π`, extract
    the axis from the diagonal of `(R + I) / 2` (which equals `axis ⊗ axis` in this regime),
    taking `sqrt` of each diagonal entry and disambiguating signs from the off-diagonal
    terms — the standard textbook resolution, not a novel derivation.
- `axis_angle_to_matrix(v) -> np.ndarray`: `angle = norm(v)`; if `angle ≈ 0`, return
  identity directly (no division by zero risk); else `axis = v / angle`, apply Rodrigues'
  rotation formula `R = I + sin(angle) K + (1 - cos(angle)) K@K` where `K` is the
  skew-symmetric matrix of `axis`. This direction has no π-singularity — Rodrigues' formula
  is well-defined at every angle including π.
- Batch counterparts `matrices_to_axis_angles`/`axis_angles_to_matrices` (arbitrary leading
  dims), built the same way `matrices_to_quats` already handles its own multi-branch
  selection: per-sample branch selection via `np.where`, not a Python-level loop, matching
  the file's existing vectorization convention.

**Degenerate-input policy for the new functions:** the architecture report flagged an
existing inconsistency in this file — `rot6d_to_matrix` (scalar) raises on degenerate input,
while `rot6ds_to_matrices` (batch) silently clamps. The new axis-angle functions do not need
to resolve that pre-existing inconsistency (out of scope here), but should not introduce a
*third*, differently-inconsistent policy — recommend the new functions raise on degenerate
input in both scalar and batch form (there is no real "degenerate axis-angle vector" case
analogous to rot6d's near-parallel-columns problem; angle≈0 and angle≈π are both
well-defined, handled branches, not failure cases), so no clamp-vs-raise decision is even
needed here.

### Explicit limitation of this pass (state loudly, do not let this be discovered later)

**Only `mcap_converter` becomes rotation-encoding-aware in this pass.** Every downstream
consumer of `observation.state` — `anvil_trainer`'s `EEAbsTransform`/`EEDeltaTransform`
(`ee_obs_abs_forward`, which assumes 8-dim quaternion input), `anvil_eval`'s `evaluator.py`
and `gt_replay.py` (same assumption, via `anvil_shared.ee_transform.n_arms_from_dims`
hardcoding `EE_STATE_DIM_PER_ARM = 8`), and the ROS2 `inference_node.py`'s
`ee_abs_uses_rot6d_obs` heuristic (which already distinguishes two *policy-facing* obs
encodings but has no notion of a *dataset* stored in anything other than quaternion) —
still hard-assumes `observation.state` is quaternion on disk. **Confirmed by actually
running it, not just reasoned about:** converting a real dataset with
`observation_encoding: "rot6d"` through the real CLI succeeds cleanly (correct 10-dim
`observation.state`, confirmed via `meta/info.json`), but running `anvil-gt-replay
--encoding absolute` against it does **not** gracefully pass as this section originally
(inaccurately) claimed — it raises `ValueError: EE observation.state dim 10 is not a
positive multiple of 8` from `n_arms_from_dims`, a hard crash, not a silent wrong answer.
Correcting the claim rather than leaving it: the dataset is not yet trainable or usable
anywhere downstream of mcap_converter, and GT-replay specifically fails loudly rather than
"passing" — which, given this codebase's established "fail loud, not silent-wrong"
discipline, is the *acceptable* shape of this scope boundary, just not the shape originally
written here. This is a deliberate scope boundary for this pass (per the "mcap_converter
only" decision), not a gap discovered after the fact — the same discipline the architecture
report asked for when it criticized Item 4 of the original plan for being silently
unfinished. If/when downstream consumers are made encoding-aware, that is exactly the kind
of cross-package follow-up Part 2 already deferred.

---

## Sequencing

1. `config/versioning.py` (new module) — `CURRENT_SCHEMA_VERSION`, `FieldMigration`,
   `VersionMigration`, `MIGRATIONS` (one real entry: `ee_action_encoding`→
   `action_encoding`), `migrate_to_current`, `_apply_field_migrations`,
   `_detected_version`. No behavior change yet; nothing imports it.
2. `schema.py` — move `ConfigurationError` + `validate_joint_name_pattern`/
   `validate_feature_mapping` in from `validators.py` (resolves the circular-import
   direction ahead of everything else that depends on it); add `DataConfig.validate()`
   absorbing `validate_config()`'s logic; add `schema_version` field; rename
   `ee_action_encoding` → `action_encoding`; add `observation_encoding` field; add
   `is_action_delta`/`output_subdir` properties (replacing `is_ee_delta` and
   `convert.py`'s inline ternary); add `DataConfig.to_dict()`.
3. `validators.py` — shrinks to `validate_topics_exist()` only, importing `DataConfig`/
   `ConfigurationError` from `schema.py`.
4. `config/encodings.py` (new module) — constants + `encode_rotation` +
   `observation_state_*_per_arm` helpers. No behavior change yet; nothing imports it.
5. `anvil_shared/rotation.py` — add `matrix_to_axis_angle`/`axis_angle_to_matrix` +
   batch counterparts, with their own dedicated unit tests (round-trip exactness at
   generic angles, and explicit tests at `angle≈0` and `angle≈π` — the two branches called
   out above must each have a test that actually exercises that branch, not just generic
   random-angle coverage that might never land near either singularity).
6. `loader.py` — rewritten to hydration-only: `from_dict` calls `migrate_to_current` as its
   first line, unconditionally (both `strict` modes); drops its own inline
   `data_space`/`ee_action_encoding` value validation (now `DataConfig.validate()`'s job
   exclusively); adds the unrecognized-top-level-key rejection gated behind a new `strict:
   bool = True` parameter (explicit YAML-key allowlist, including aliases like
   `joint_names`; `strict=False` skips this check entirely — see "Strict vs. lenient
   hydration" in Part 0); parses the renamed `action_encoding` and new
   `observation_encoding` fields; new `ConfigLoader.to_yaml(config, path)` method.
7. `extractor.py` — `_align_ee_signals` signature change (returns `(frame, state_quat)`);
   `_align_frame_at_cursor`/`extract_frames` rewired for `prev_ee_state_quat` at both yield
   sites; `is_ee_delta` → `is_action_delta` rename at the one call site.
8. `writer.py` — `_define_features` reads layout from `encodings.py` instead of its
   hardcoded tuple.
9. `convert.py` — `space_suffix = config.output_subdir` (one-line simplification);
   `validate_config(config)` call site becomes `config.validate()`; conversion-config-save
   branch becomes `ConfigLoader.to_yaml(config, conversion_config_dest)`.
10. `anvil_eval/gt_replay.py` and `mcap_converter/utils/debug_plot.py` — both switch from
    ad hoc `yaml.safe_load` + `.get(...)` reads of `conversion_config.yaml` to
    `ConfigLoader.from_yaml(path, strict=False)` (which now also runs migration
    transparently), reading the needed field(s) off the resulting `DataConfig` — never
    calling `.validate()` on it. `anvil_eval_ros/cli.py`'s ad hoc reader is explicitly NOT
    touched — see the Status section decision and the recorded memory.
11. `cli/migrate_config.py` (new file) + `pyproject.toml` console-script entry
    (`dataset-config-migrate`) — the migration CLI described in Part 0b § 3: version check
    (true no-op if current), backup-collision check (`--force` to override), interactive
    yes/no confirmation (not skippable by `--force`), then rename-to-backup + write-upgraded.
12. Cosmetic: update the five docstring-only `ee_action_encoding` mentions in
    `anvil_trainer/` to the new name.
13. New/updated tests:
    - `test_ee_encoding.py` gains `observation_encoding` coverage (all three values,
      including a rot6d/axis-angle dataset's `observation.state` shape/values, and a test
      that `action_encoding="delta"` produces the identical baked `action` column
      regardless of `observation_encoding` — proving the quat/encoded decoupling works); a
      config-level test for the `"relative"` reserved-value error message; a test asserting
      an unrecognized top-level YAML key is rejected under `strict=True` (regression test
      for the `openarm_bimanual.yaml`-shaped failure mode) with a companion test proving the
      SAME legacy-shaped YAML loads without error under `strict=False`; existing
      `validate_config(...)`-calling tests updated to call `config.validate()` instead.
    - `test_rotation.py` — axis-angle primitives per step 5.
    - New `test_versioning.py`: `migrate_to_current` correctly upgrades a raw v1.0 dict
      (missing `schema_version`, `ee_action_encoding` present) to current with
      `action_encoding` set correctly (**this is the regression test for the original
      triggering bug**); a **required** chained-migration test injecting a synthetic
      `1.1→1.2` step via the `registry`/`current_version` kwargs and asserting both hops
      applied in order (Gap 5 — do not let coverage stop at the one real migration); a test
      for the "old key wins" conflict rule firing only when a migration step actually runs,
      and NOT firing when `schema_version` already equals current (proving the Gap 1
      boundary); a test for a pure-removal `FieldMigration` (`new_key=None`, synthetic —
      logs a warning and drops the key) proving Gap 6's second case works, not just rename.
    - New `test_migrate_config.py` (or CLI-level integration test): already-current config
      → zero file writes/renames, informational message only (Gap 4); existing
      version-tagged backup present + no `--force` → refuses, no file ops (Gap 3); `--force`
      overwrites the backup but the yes/no prompt still fires regardless (Gap 2); a full
      happy-path run (confirm "yes") produces the renamed backup + the upgraded file with
      correct content.
14. A new debug config exercising `observation_encoding: "rot6d"` (or `"axis_angle"`),
    parallel to `configs/mcap_converter/openarm_ee_delta_debug.yaml`, run through the real
    CLI against the existing `tests/smoke/fixtures/ee-session/` fixtures and confirmed with
    `anvil-gt-replay` — same "prove it through the real CLI, not just unit tests" discipline
    Part 1's own motivation (the architecture report's finding that `action_encoding="delta"`
    had never been exercised end-to-end) is meant to instill going forward.
