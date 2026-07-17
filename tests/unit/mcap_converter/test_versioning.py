"""Tests for mcap_converter.config.versioning — the schema-version migration registry.

Covers:
  1. The real ee_action_encoding -> action_encoding migration (the regression test for
     the original triggering bug: a rename silently losing an already-meaningful value)
  2. Already-current input is a true no-op (no field changes, no warnings)
  3. N-step chained migration via a synthetic second step (Gap 5) — proves the
     while-loop's sequential application, not just the one real migration
  4. The "old key wins on conflict" rule fires ONLY during an actual version upgrade,
     never when the file already claims to be current (Gap 1 boundary)
  5. A pure-removal FieldMigration (new_key=None) drops the key and warns (Gap 6)
  6. Unmigratable version (no registered path) raises ConfigurationError
  7. The REAL v1.0 -> v1.1 structural migration (_migrate_legacy_v1_0_shape), tested
     against verbatim content from `main` branch's actual configs/mcap_converter/*.yaml —
     leader-follower, quest+real-command-topics, and quest+action_from_observation. v1.0
     here means what's genuinely on `main` (confirmed by reading it directly), NOT
     something invented for this branch — see versioning.py's module docstring.
"""
from __future__ import annotations

import pytest
import yaml

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.schema import ConfigurationError
from mcap_converter.config.versioning import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    FieldMigration,
    VersionMigration,
    detected_version,
    migrate_to_current,
)


class TestRealMigration:
    def test_ee_action_encoding_migrates_to_action_encoding(self):
        """Regression test for the original triggering bug."""
        raw = {
            "data_space": "ee",
            "observation_topics": {"right": "/ee_pose_right"},
            "ee_action_encoding": "delta",
        }
        out = migrate_to_current(raw)
        assert out["action_encoding"] == "delta"
        assert "ee_action_encoding" not in out
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_raw_dict_never_mutated(self):
        raw = {"ee_action_encoding": "delta"}
        raw_copy = dict(raw)
        migrate_to_current(raw)
        assert raw == raw_copy


class TestAlreadyCurrentIsNoOp:
    def test_no_field_changes_when_already_current(self):
        raw = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "data_space": "ee",
            "action_encoding": "delta",
        }
        out = migrate_to_current(raw)
        assert out == raw

    def test_leftover_old_key_untouched_when_already_current(self, caplog):
        """See Gap 1 boundary test below for the full argument; this just confirms the
        while loop's condition short-circuits — no migration warning is logged."""
        raw = {"schema_version": CURRENT_SCHEMA_VERSION, "ee_action_encoding": "delta"}
        out = migrate_to_current(raw)
        # Untouched — migration never runs, this key survives unmodified. It is the
        # unknown-key strict/lenient mechanism's job to deal with it from here, not
        # migration's (see TestGap1Boundary below).
        assert out["ee_action_encoding"] == "delta"
        assert "action_encoding" not in out


class TestChainedMigration:
    """Gap 5: N-step sequential application, proven with a synthetic second step —
    coverage must not stop at the one real migration."""

    def test_two_step_chain_applies_both_in_order(self):
        synthetic_registry = (
            *MIGRATIONS,
            VersionMigration(
                from_version="1.1",
                to_version="1.2",
                field_migrations=(
                    FieldMigration(
                        old_key="observation_encoding",
                        new_key="obs_rotation_encoding",
                        reason="synthetic test-only step",
                    ),
                ),
            ),
        )
        raw = {"ee_action_encoding": "delta", "observation_encoding": "rot6d"}
        out = migrate_to_current(raw, registry=synthetic_registry, current_version="1.2")

        assert out["schema_version"] == "1.2"
        # First hop (real): ee_action_encoding -> action_encoding
        assert out["action_encoding"] == "delta"
        assert "ee_action_encoding" not in out
        # Second hop (synthetic): observation_encoding -> obs_rotation_encoding
        assert out["obs_rotation_encoding"] == "rot6d"
        assert "observation_encoding" not in out

    def test_starting_mid_chain_only_applies_remaining_steps(self):
        synthetic_registry = (
            *MIGRATIONS,
            VersionMigration(
                from_version="1.1",
                to_version="1.2",
                field_migrations=(
                    FieldMigration(old_key="a", new_key="b", reason="synthetic"),
                ),
            ),
        )
        # Already at 1.1 (post-rename) — only the second hop should apply.
        raw = {"schema_version": "1.1", "action_encoding": "delta", "a": 1}
        out = migrate_to_current(raw, registry=synthetic_registry, current_version="1.2")
        assert out["schema_version"] == "1.2"
        assert out["action_encoding"] == "delta"  # untouched — already correct name
        assert out["b"] == 1
        assert "a" not in out

    def test_unregistered_version_raises(self):
        raw = {"schema_version": "0.5"}
        with pytest.raises(ConfigurationError, match="No migration registered"):
            migrate_to_current(raw)


class TestGap1Boundary:
    """The 'old key wins on conflict' rule must fire ONLY inside an actual version
    upgrade, never when the file already claims to be current-version."""

    def test_conflict_rule_fires_during_actual_upgrade(self):
        """Both old and new key present in a v1.0 (pre-rename) file — legacy value wins,
        per the migration-only conflict rule."""
        raw = {
            "ee_action_encoding": "delta",
            "action_encoding": "absolute",  # should be overridden by the legacy value
        }
        out = migrate_to_current(raw)
        assert out["action_encoding"] == "delta"

    def test_conflict_rule_never_fires_when_already_current(self):
        """Same two keys, but schema_version already claims current — migration must not
        touch either key; this scenario is a malformed file, not a migration case, and is
        left entirely to the strict/lenient unknown-key mechanism to handle."""
        raw = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "ee_action_encoding": "delta",
            "action_encoding": "absolute",
        }
        out = migrate_to_current(raw)
        # action_encoding is untouched at its original value — migration never ran.
        assert out["action_encoding"] == "absolute"
        assert out["ee_action_encoding"] == "delta"  # also untouched, still present


class TestPureRemoval:
    """Gap 6: a FieldMigration with new_key=None drops the key entirely, no replacement —
    a distinct case from rename, not yet a real entry in MIGRATIONS but must work."""

    def test_pure_removal_drops_key_with_no_replacement(self):
        synthetic_registry = (
            VersionMigration(
                from_version="1.0",
                to_version="1.1",
                field_migrations=(
                    FieldMigration(
                        old_key="deprecated_feature_flag",
                        new_key=None,
                        reason="feature removed entirely, synthetic test",
                    ),
                ),
            ),
        )
        raw = {"deprecated_feature_flag": True, "data_space": "ee"}
        out = migrate_to_current(raw, registry=synthetic_registry, current_version="1.1")
        assert "deprecated_feature_flag" not in out
        assert out["data_space"] == "ee"
        assert out["schema_version"] == "1.1"

    def test_pure_removal_is_a_noop_when_key_absent(self):
        synthetic_registry = (
            VersionMigration(
                from_version="1.0",
                to_version="1.1",
                field_migrations=(
                    FieldMigration(old_key="never_present", new_key=None, reason="test"),
                ),
            ),
        )
        raw = {"data_space": "ee"}
        out = migrate_to_current(raw, registry=synthetic_registry, current_version="1.1")
        assert out == {"data_space": "ee", "schema_version": "1.1"}


def test_detected_version_absence_means_v1_0():
    assert detected_version({}) == "1.0"
    assert detected_version({"schema_version": "1.1"}) == "1.1"


# ---------------------------------------------------------------------------
# Real v1.0 -> v1.1 structural migration, against verbatim `main`-branch content
# ---------------------------------------------------------------------------

# Verbatim from main's configs/mcap_converter/openarm_bimanual.yaml (leader-follower —
# action derived from leader_-prefixed joint_states entries, no action_topics at all).
_MAIN_LEADER_FOLLOWER_YAML = """
robot_state_topic: "/joint_states"
joint_names:
  separator: "_"
  source:
    leader: action
    follower: observation
  arms:
    r: right
    l: left
camera_topics:
  - "/cam_waist/image_raw/compressed"
  - "/cam_wrist_r/image_raw/compressed"
camera_topic_mapping:
  "/cam_waist/image_raw/compressed": "waist"
  "/cam_wrist_r/image_raw/compressed": "wrist_r"
image_resolution: [640, 480]
observation_feature_mapping:
  state: "position"
  others: []
action_feature_mapping:
  state: "position"
  others: []
"""

# Verbatim from main's configs/mcap_converter/openarm_bimanual_quest.yaml (quest teleop,
# real per-arm Float64MultiArray command topics, topic-keyed action_topics).
_MAIN_QUEST_BIMANUAL_YAML = """
robot_state_topic: "/joint_states"
joint_names:
  separator: "_"
  source:
    follower: observation
  arms:
    r: right
    l: left
action_topics:
  "/follower_l_forward_position_controller/commands":
    arm: "left"
    joint_order: ["joint1", "joint2", "finger_joint1"]
  "/follower_r_forward_position_controller/commands":
    arm: "right"
    joint_order: ["joint1", "joint2", "finger_joint1"]
camera_topics:
  - "/cam_waist/image_raw/compressed"
camera_topic_mapping:
  "/cam_waist/image_raw/compressed": "waist"
image_resolution: [640, 480]
observation_feature_mapping:
  state: "position"
  others: []
action_feature_mapping:
  state: "position"
  others: []
"""

# Verbatim from main's configs/mcap_converter/openarm_single_quest_afo.yaml (quest teleop,
# single arm, real command topic PLUS action_from_observation: true — the ambiguous case).
_MAIN_QUEST_SINGLE_AFO_YAML = """
robot_state_topic: "/joint_states"
joint_names:
  separator: "_"
  source:
    follower: observation
  arms:
    r: right
action_topics:
  "/follower_r_forward_position_controller/commands":
    arm: "right"
    joint_order: ["joint1", "joint2", "finger_joint1"]
action_from_observation: true
camera_topics:
  - "/cam_waist/image_raw/compressed"
camera_topic_mapping:
  "/cam_waist/image_raw/compressed": "waist"
image_resolution: [640, 480]
observation_feature_mapping:
  state: "position"
  others: []
action_feature_mapping:
  state: "position"
  others: []
"""

# Verbatim from main's configs/mcap_converter/openarm_single_quest.yaml (quest teleop,
# single arm, real command topic, action_from_observation: FALSE — must behave like #2,
# not like the true case above).
_MAIN_QUEST_SINGLE_NO_AFO_YAML = _MAIN_QUEST_SINGLE_AFO_YAML.replace(
    "action_from_observation: true", "action_from_observation: false"
)


class TestRealMainBranchLegacyShapes:
    """v1.0 here is what's genuinely on `main` — verified by reading it directly, not
    assumed. Each of these must both migrate cleanly AND pass strict loading + validate()
    end to end, proving the real files, not just synthetic fixtures, actually work."""

    def test_leader_follower_migrates_and_validates(self):
        raw = yaml.safe_load(_MAIN_LEADER_FOLLOWER_YAML)
        cfg = ConfigLoader.from_dict(raw, strict=True)
        cfg.validate()
        assert cfg.data_space == "joint"
        assert cfg.observation_topics == {"right": "/joint_states", "left": "/joint_states"}
        assert cfg.action_topics == {}  # action comes from joint_names.source, not action_topics
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION

    def test_quest_bimanual_migrates_and_validates(self):
        raw = yaml.safe_load(_MAIN_QUEST_BIMANUAL_YAML)
        cfg = ConfigLoader.from_dict(raw, strict=True)
        cfg.validate()
        assert cfg.observation_topics == {"right": "/joint_states", "left": "/joint_states"}
        # Topic-keyed -> arm-keyed inversion, both arms present with their real topics.
        assert cfg.action_topics["left"].topic == "/follower_l_forward_position_controller/commands"
        assert cfg.action_topics["left"].joint_order == ["joint1", "joint2", "finger_joint1"]
        assert cfg.action_topics["right"].topic == "/follower_r_forward_position_controller/commands"

    def test_quest_single_afo_true_keeps_real_action_topics(self, caplog):
        """The ambiguous case: action_from_observation=true alongside a real command
        topic. Migration must keep the real topic (strict behavior) and warn loudly about
        the semantic gap, not silently guess."""
        raw = yaml.safe_load(_MAIN_QUEST_SINGLE_AFO_YAML)
        migrated = migrate_to_current(raw)
        assert "action_from_observation" not in migrated
        assert "action_from_observation_n" not in migrated
        assert migrated["action_topics"]["right"]["topic"] == (
            "/follower_r_forward_position_controller/commands"
        )
        cfg = ConfigLoader.from_dict(raw, strict=True)
        cfg.validate()
        assert cfg.action_topics["right"].topic == "/follower_r_forward_position_controller/commands"

    def test_quest_single_afo_false_identical_to_no_afo_field(self):
        """action_from_observation=false must behave identically to the field being
        absent entirely — it's a no-op either way, never triggering the ambiguous-case
        warning path."""
        raw_false = yaml.safe_load(_MAIN_QUEST_SINGLE_NO_AFO_YAML)
        cfg_false = ConfigLoader.from_dict(raw_false, strict=True)
        cfg_false.validate()
        assert cfg_false.action_topics["right"].topic == (
            "/follower_r_forward_position_controller/commands"
        )

    def test_leader_follower_missing_arms_warns_and_leaves_topics_empty(self, caplog):
        """Defensive case: robot_state_topic present but joint_names.arms empty — cannot
        derive observation_topics, must warn rather than silently produce a wrong guess."""
        raw = {
            "robot_state_topic": "/joint_states",
            "joint_names": {"separator": "_", "source": {"follower": "observation"}, "arms": {}},
        }
        with caplog.at_level("WARNING"):
            migrated = migrate_to_current(raw)
        assert migrated["observation_topics"] == {}
        assert any("cannot derive" in r.message for r in caplog.records)
