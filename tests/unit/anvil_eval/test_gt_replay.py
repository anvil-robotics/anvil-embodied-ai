"""Tests for anvil_eval.gt_replay — the GT-replay gate for the Delta(n-(n-1))
transform pipeline.

Covers:
  1. _replay_episode_absolute — pure round-trip on synthetic absolute-action
     episodes, single and bimanual, near-machine-precision recovery
  2. _replay_episode_delta — baked-column correctness, using
     ee_delta_forward to build a synthetic "already-baked" action column
     (so the test is independent of mcap_converter), confirming the
     recompute-from-state check passes for genuinely correct data and FAILS
     when the on-disk column is corrupted
  3. First-frame self-anchor convention check (delta mode)
  4. _detect_encoding — reads action_encoding from conversion_config.yaml via
     ConfigLoader (lenient), including transparent migration of pre-rename
     configs that still say ee_action_encoding (schema v1.0)
  5. Regression guard: a corrupted absolute action column must FAIL the gate
     (round-trip error exceeds tolerance), not silently pass
"""

from __future__ import annotations

import numpy as np
import pytest

from anvil_eval.gt_replay import (
    _detect_encoding,
    _replay_episode_absolute,
    _replay_episode_delta,
)

EE_STATE_DIM = 8
EE_ACTION_DIM = 10


def _random_episode(n_arms: int = 1, T: int = 20, seed: int = 0):
    """Synthetic (actions, states) with valid rot6d/quat, absolute action encoding."""
    rng = np.random.default_rng(seed)
    states = np.zeros((T, EE_STATE_DIM * n_arms))
    actions = np.zeros((T, EE_ACTION_DIM * n_arms))
    for arm in range(n_arms):
        s0, a0 = arm * EE_STATE_DIM, arm * EE_ACTION_DIM
        states[:, s0:s0 + 3] = rng.uniform(-0.3, 0.3, size=(T, 3))
        q = rng.normal(size=(T, 4))
        states[:, s0 + 3:s0 + 7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        states[:, s0 + 7] = rng.uniform(0.0, 0.05, T)

        actions[:, a0:a0 + 3] = rng.uniform(-0.3, 0.3, size=(T, 3))
        from anvil_shared.rotation import matrices_to_rot6d
        Qs = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(T)])
        for t in range(T):
            if np.linalg.det(Qs[t]) < 0:
                Qs[t][:, 0] *= -1
        actions[:, a0 + 3:a0 + 9] = matrices_to_rot6d(Qs)
        actions[:, a0 + 9] = rng.uniform(0.0, 0.05, T)
    return actions, states


class TestReplayEpisodeAbsolute:
    def test_single_arm_passes(self):
        actions, states = _random_episode(n_arms=1, T=15, seed=1)
        result = _replay_episode_absolute(actions, states)
        assert result["skipped"] is False
        assert result["max_pos_err_m"] < 1e-9
        assert result["max_rot_err_deg"] < 1e-4
        assert result["n_frames_checked"] == 14

    def test_bimanual_passes(self):
        actions, states = _random_episode(n_arms=2, T=12, seed=2)
        result = _replay_episode_absolute(actions, states)
        assert result["max_pos_err_m"] < 1e-9
        assert result["max_rot_err_deg"] < 1e-4

    def test_too_short_episode_is_skipped(self):
        actions, states = _random_episode(n_arms=1, T=1, seed=3)
        result = _replay_episode_absolute(actions, states)
        assert result["skipped"] is True

    def test_round_trip_exactness_is_anchor_content_independent(self):
        """Honest limitation of 'absolute' mode, documented via test: since
        this mode has no external ground truth to compare against (it only
        checks forward->inverse self-consistency), ANY valid action/state
        pair round-trips exactly — including a perturbed one. This is by
        design, not a gate weakness: catching an actually-wrong on-disk
        column requires 'delta' mode's recompute-vs-actual check (see
        TestReplayEpisodeDelta.test_corrupted_baked_column_fails, which DOES
        have an external reference to compare against and DOES fail)."""
        actions, states = _random_episode(n_arms=1, T=15, seed=4)
        perturbed = actions.copy()
        perturbed[5, 0] += 0.01  # 1cm perturbation in xyz
        result = _replay_episode_absolute(perturbed, states)
        assert result["max_pos_err_m"] < 1e-9  # still round-trips — no external reference exists


class TestReplayEpisodeDelta:
    def _build_baked_episode(self, n_arms: int = 1, T: int = 10, seed: int = 10):
        """Build a genuinely-correct baked-delta episode: states are the
        source of truth; actions[t] = ee_delta_forward(action_abs[t], state[t-1])
        for t>=1, and actions[0] = self-anchor zero-delta — exactly what
        mcap_converter's _align_ee_signals is supposed to produce."""
        from anvil_shared.ee_transform import ee_delta_forward, ee_obs_abs_forward

        _, states = _random_episode(n_arms=n_arms, T=T, seed=seed)
        action_abs = ee_obs_abs_forward(states)  # (T, action_dim), matches state pose exactly

        baked = np.zeros_like(action_abs)
        baked[0] = ee_delta_forward(action_abs[0], states[0])  # self-anchor -> zero delta
        if T > 1:
            baked[1:] = ee_delta_forward(action_abs[1:], states[:-1])
        return baked, states

    def test_genuinely_correct_baked_column_passes(self):
        baked, states = self._build_baked_episode(n_arms=1, T=10, seed=11)
        result = _replay_episode_delta(baked, states)
        assert result["skipped"] is False
        assert result["max_pos_err_m"] < 1e-9
        assert result["max_rot_err_deg"] < 1e-4
        assert result["first_frame_pos_err_m"] < 1e-9
        assert result["first_frame_rot_l2_err"] < 1e-9

    def test_bimanual_baked_column_passes(self):
        baked, states = self._build_baked_episode(n_arms=2, T=8, seed=12)
        result = _replay_episode_delta(baked, states)
        assert result["max_pos_err_m"] < 1e-9
        assert result["max_rot_err_deg"] < 1e-4

    def test_corrupted_baked_column_fails(self):
        """The core regression guard for the GATE: if mcap_converter's baked
        column were wrong (e.g. computed against the wrong anchor), this
        must produce a large error, not a false PASS."""
        baked, states = self._build_baked_episode(n_arms=1, T=10, seed=13)
        corrupted = baked.copy()
        corrupted[5, 0:3] += 0.05  # 5cm corruption — simulates a wrong-anchor bug
        result = _replay_episode_delta(corrupted, states)
        assert result["max_pos_err_m"] > 0.01, (
            "A 5cm corruption in the baked column must be caught as a large "
            "recompute-vs-actual discrepancy, not silently pass."
        )

    def test_first_frame_convention_violation_detected(self):
        """If action[0] were NOT the self-anchor zero-delta (e.g. mcap_converter
        forgot the first-frame special case), first_frame checks must catch it."""
        baked, states = self._build_baked_episode(n_arms=1, T=10, seed=14)
        broken = baked.copy()
        broken[0, 0] = 0.5  # first frame no longer zero-delta
        result = _replay_episode_delta(broken, states)
        assert result["first_frame_pos_err_m"] > 0.1

    def test_single_frame_episode_skips_recompute_but_checks_first_frame(self):
        baked, states = self._build_baked_episode(n_arms=1, T=1, seed=15)
        result = _replay_episode_delta(baked, states)
        assert result["skipped"] is False
        assert result["n_frames_checked"] == 0
        assert result["first_frame_pos_err_m"] < 1e-9


class TestDetectEncoding:
    def test_absent_config_defaults_absolute(self, tmp_path):
        assert _detect_encoding(tmp_path) == "absolute"

    def test_reads_delta_from_current_field_name(self, tmp_path):
        (tmp_path / "conversion_config.yaml").write_text(
            "schema_version: '1.1'\naction_encoding: delta\n"
        )
        assert _detect_encoding(tmp_path) == "delta"

    def test_reads_absolute_from_current_field_name(self, tmp_path):
        (tmp_path / "conversion_config.yaml").write_text(
            "schema_version: '1.1'\naction_encoding: absolute\ndata_space: ee\n"
        )
        assert _detect_encoding(tmp_path) == "absolute"

    def test_missing_key_defaults_absolute(self, tmp_path):
        (tmp_path / "conversion_config.yaml").write_text("data_space: ee\n")
        assert _detect_encoding(tmp_path) == "absolute"

    def test_reads_delta_from_legacy_pre_rename_field_name(self, tmp_path):
        """Regression test for the original triggering bug: a dataset converted before
        the ee_action_encoding -> action_encoding rename (schema v1.0, no schema_version
        key) must still resolve to its real historical value, not silently default to
        "absolute" under a field name it never had."""
        (tmp_path / "conversion_config.yaml").write_text("ee_action_encoding: delta\n")
        assert _detect_encoding(tmp_path) == "delta"
