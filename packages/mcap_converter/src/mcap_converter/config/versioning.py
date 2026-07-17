"""Schema-version migration registry for :class:`DataConfig`/``conversion_config.yaml``.

**What v1.0 and v1.1 actually mean — verified against real files, not assumed:**

- **v1.0** = the schema as it exists on the ``main`` branch's
  ``configs/mcap_converter/*.yaml`` today — a pure joint-space, pre-unification format
  with NO `data_space`/`observation_topics`/EE support of any kind. Confirmed by reading
  every config file under ``main``'s ``configs/mcap_converter/``; three real sub-shapes
  exist there, all handled by :func:`_migrate_legacy_v1_0_shape` below:

  1. Leader-follower (``openarm_bimanual.yaml``): singular ``robot_state_topic``, action
     derived from ``leader_``-prefixed names in the same ``/joint_states`` topic via
     ``joint_names.source``, no ``action_topics`` key at all.
  2. Quest teleop with real command topics (``openarm_bimanual_quest*.yaml``,
     ``openarm_single_quest.yaml``): singular ``robot_state_topic``, action from
     separate ``Float64MultiArray`` command topics, ``action_topics`` **topic-keyed**
     (``{topic: {arm, joint_order}}`` — inverted from the current schema's
     ``{arm: {topic, joint_order}}``).
  3. Quest teleop with ``action_from_observation`` (``openarm_single_quest_afo.yaml``):
     same as #2, plus a boolean flag (optionally paired with
     ``action_from_observation_n``) meaning "if the command topic isn't present in this
     particular recording, fall back to using the observation as the action instead."

  v1.0 also covers a second, narrower case that never existed on `main` at all: this
  branch's OWN unified joint+EE schema (``data_space``, ``observation_topics`` already
  arm-keyed, EE support) as it stood immediately before this session — using
  ``ee_action_encoding``, with no ``schema_version`` key. Absence of ``schema_version`` is
  the v1.0 signal for BOTH cases; :func:`migrate_to_current` distinguishes them by which
  legacy fields are actually present (see :func:`_migrate_legacy_v1_0_shape`'s trigger
  condition), not by any separate flag.

- **v1.1** = the current ``implement-ee-space`` tip's schema: the unified joint+EE format
  plus ``action_encoding``/``observation_encoding`` (renamed from ``ee_action_encoding``,
  Part 1 of ``claude_docs/mcap-converter-encoding-refactor-plan.md``) plus this module's
  own ``schema_version`` field.

Earlier drafts of this module incorrectly described v1.0 as "the schema before
action_encoding/observation_encoding existed" without checking `main` — that description
matched only the second, branch-internal case above, not what's actually on `main`. Fixed
here after that was pointed out and verified directly against `main`'s real config files.

This module is deliberately generic, not a one-off patch for a single rename: adding a
future migration is a matter of appending one ``VersionMigration`` entry to ``MIGRATIONS``
(with a ``custom`` callable when the change isn't expressible as simple field rename/
removal — see the real v1.0→v1.1 step below for a worked example), never rewriting
:func:`migrate_to_current` itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .schema import CURRENT_SCHEMA_VERSION, ConfigurationError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldMigration:
    """One simple field-level change within a single version step: a straight rename or a
    pure removal — see :class:`VersionMigration`'s ``custom`` field for changes that need
    actual data restructuring (renamed dict shapes, values derived from other fields),
    which this declarative type cannot express.

    ``new_key`` set  -> rename: ``old_key``'s value moves to ``new_key``.
    ``new_key=None`` -> pure removal: ``old_key`` is dropped, no replacement (e.g. a future
                        feature deprecated entirely, not renamed to anything).
    """

    old_key: str
    new_key: Optional[str]
    reason: str  # human-readable; always logged when triggered


@dataclass(frozen=True)
class VersionMigration:
    """The full set of changes needed to go from one version to the next.

    ``field_migrations`` handles simple renames/removals declaratively. ``custom``, if
    set, is applied AFTER ``field_migrations`` and can perform arbitrary restructuring —
    needed for real migrations that aren't expressible as key renames (e.g. inverting a
    topic-keyed dict to arm-keyed, deriving a new field's value from two old ones).
    """

    from_version: str
    to_version: str
    field_migrations: Tuple[FieldMigration, ...] = ()
    custom: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def _migrate_legacy_v1_0_shape(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Restructures `main` branch's pre-unification legacy joint-only schema (singular
    ``robot_state_topic``, topic-keyed ``action_topics``, ``action_from_observation*``)
    into the current unified joint+EE schema's shape.

    A no-op for a v1.0 file that's already in the (branch-internal, never-on-main)
    unified-schema-pre-rename shape — those already have ``observation_topics``, so the
    trigger condition below doesn't fire; only the declarative ``ee_action_encoding``
    rename (registered separately in ``MIGRATIONS``) applies to that case.
    """
    if "robot_state_topic" not in raw or "observation_topics" in raw:
        return raw

    out = dict(raw)
    robot_state_topic = out.pop("robot_state_topic")

    arms = list(out.get("joint_names", {}).get("arms", {}).values())
    if not arms:
        log.warning(
            "[config-migrate] legacy robot_state_topic present but joint_names.arms is "
            "empty — cannot derive per-arm observation_topics; leaving it empty (the "
            "migrated config will fail DataConfig.validate())."
        )
    out["observation_topics"] = {arm: robot_state_topic for arm in arms}
    out.setdefault("data_space", "joint")

    # Topic-keyed action_topics (main's shape: {topic: {arm, joint_order}}) -> arm-keyed
    # (current schema's shape: {arm: {topic, joint_order}}). Leader-follower configs have
    # no action_topics key at all (action comes from joint_names.source instead) — nothing
    # to invert, correctly left absent/empty.
    legacy_action_topics = out.get("action_topics")
    if isinstance(legacy_action_topics, dict) and legacy_action_topics:
        looks_topic_keyed = all(
            isinstance(v, dict) and "arm" in v for v in legacy_action_topics.values()
        )
        if looks_topic_keyed:
            out["action_topics"] = {
                spec["arm"]: {"topic": topic, "joint_order": spec.get("joint_order", [])}
                for topic, spec in legacy_action_topics.items()
            }

    # action_from_observation[_n]: no field-level equivalent in the current schema (its
    # own loader docstring already calls these "no longer accepted") — pure removal, not a
    # rename. The closest current mechanism is empty action_topics (extractor.py already
    # treats that as act-from-obs). When action_topics was ALSO present, main's semantics
    # were a per-recording CONDITIONAL fallback ("use obs only if the command topic wasn't
    # actually recorded") — that can't be decided statically from the config alone, so this
    # is a genuine, real semantic narrowing, not a mechanical rename. Warn loudly rather
    # than silently pick a side.
    afo = out.pop("action_from_observation", None)
    out.pop("action_from_observation_n", None)
    if afo:
        if out.get("action_topics"):
            log.warning(
                "[config-migrate] legacy action_from_observation=true alongside real "
                "action_topics has no exact equivalent in the current schema (main's "
                "conditional per-recording fallback isn't expressible statically) — "
                "migrated config KEEPS action_topics as-is (strict, no automatic "
                "fallback). Pass --act-from-obs at convert time for the old fallback "
                "behavior, or manually set action_topics: {} to force it."
            )
        else:
            log.warning(
                "[config-migrate] legacy action_from_observation=true with no "
                "action_topics -> migrated to action_topics: {} (current schema's "
                "act-from-obs convention)."
            )
            out["action_topics"] = {}

    for removed_key in ("robot_state_topics", "motor_feature_mapping"):
        if removed_key in out:
            log.warning(
                "[config-migrate] dropping deprecated field %r (no replacement in "
                "current schema)", removed_key,
            )
            out.pop(removed_key)

    return out


MIGRATIONS: Tuple[VersionMigration, ...] = (
    VersionMigration(
        from_version="1.0",
        to_version="1.1",
        field_migrations=(
            FieldMigration(
                old_key="ee_action_encoding",
                new_key="action_encoding",
                reason=(
                    "renamed for generality (joint-space encoding support planned) — see "
                    "claude_docs/mcap-converter-encoding-refactor-plan.md"
                ),
            ),
        ),
        custom=_migrate_legacy_v1_0_shape,
    ),
    # Future: VersionMigration(from_version="1.1", to_version="1.2", field_migrations=(...))
)


def detected_version(raw: Dict[str, Any]) -> str:
    """Absence of ``schema_version`` means v1.0 — that absence IS the v1.0 signal."""
    return str(raw.get("schema_version", "1.0"))


def _apply_field_migrations(raw: Dict[str, Any], step: VersionMigration) -> Dict[str, Any]:
    out = dict(raw)
    for fm in step.field_migrations:
        if fm.old_key not in out:
            continue
        old_value = out.pop(fm.old_key)
        if fm.new_key is None:
            log.warning(
                "[config-migrate] %s->%s: dropping deprecated field %r (%s)",
                step.from_version, step.to_version, fm.old_key, fm.reason,
            )
            continue
        if fm.new_key in out:
            # Migration-only conflict rule — see Gap 1 in the design doc. This branch can
            # only ever run from inside this function, which only ever runs from
            # migrate_to_current's loop, which only ever runs when the file's OWN detected
            # version differs from current. A file that already claims to be current-version
            # never reaches this code at all, so this rule can never fire outside an actual
            # version upgrade.
            log.warning(
                "[config-migrate] %s: both legacy %r and current %r present — legacy value "
                "wins (migration-only conflict rule)",
                step.from_version, fm.old_key, fm.new_key,
            )
        else:
            log.warning(
                "[config-migrate] Upgrading from v%s: field %r is deprecated, use %r (%s)",
                step.from_version, fm.old_key, fm.new_key, fm.reason,
            )
        out[fm.new_key] = old_value
    return out


def _apply_version_migration(raw: Dict[str, Any], step: VersionMigration) -> Dict[str, Any]:
    out = _apply_field_migrations(raw, step)
    if step.custom is not None:
        out = step.custom(out)
    return out


def migrate_to_current(
    raw: Dict[str, Any],
    *,
    registry: Tuple[VersionMigration, ...] = MIGRATIONS,
    current_version: str = CURRENT_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Apply the registered migration chain from ``raw``'s detected version up to
    ``current_version``, applying as many intermediate steps as needed in sequence.

    Returns a NEW dict; ``raw`` is never mutated. A true no-op (only stamps
    ``schema_version``, no field changes) when already current.

    ``registry``/``current_version`` are overridable purely for testing chained N-step
    migration without adding synthetic entries to the real, production ``MIGRATIONS``.

    Raises
    ------
    ConfigurationError
        If the detected version isn't current and no registered step starts from it (a
        config too old — or too garbled — for any known migration path).
    """
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
        out = _apply_version_migration(out, step)
        version = step.to_version
    out["schema_version"] = current_version
    return out
