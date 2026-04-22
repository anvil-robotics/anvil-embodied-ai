"""Tests for anvil_shared.splits."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil_shared.splits import (
    compute_split_episodes,
    load_split_info,
    save_split_info,
)


# =============================================================================
# compute_split_episodes
# =============================================================================


class TestComputeSplitEpisodes:
    def test_3way_split_disjoint_and_covers_all(self):
        splits = compute_split_episodes(100, [8, 1, 1], seed=0)
        all_eps = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert all_eps == set(range(100))
        # Disjoint
        assert len(set(splits["train"]) & set(splits["val"])) == 0
        assert len(set(splits["train"]) & set(splits["test"])) == 0
        assert len(set(splits["val"]) & set(splits["test"])) == 0

    def test_ratio_proportions(self):
        splits = compute_split_episodes(100, [8, 1, 1], seed=0)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_zero_val_test_gives_all_train(self):
        """Ratio [1, 0, 0] → 100% train, empty val/test."""
        splits = compute_split_episodes(50, [1, 0, 0], seed=0)
        assert len(splits["train"]) == 50
        assert splits["val"] == []
        assert splits["test"] == []

    def test_2element_ratio_treated_as_3way(self):
        """Ratio [8, 2] is [8, 2, 0] — no test set."""
        splits = compute_split_episodes(100, [8, 2], seed=0)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 20
        assert splits["test"] == []

    def test_deterministic_with_seed(self):
        """Same seed → identical splits."""
        a = compute_split_episodes(30, [7, 2, 1], seed=42)
        b = compute_split_episodes(30, [7, 2, 1], seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = compute_split_episodes(30, [7, 2, 1], seed=0)
        b = compute_split_episodes(30, [7, 2, 1], seed=1)
        # At least one split should differ under different seeds
        assert a != b

    def test_sorted_episode_lists(self):
        splits = compute_split_episodes(30, [8, 1, 1], seed=0)
        for key in ("train", "val", "test"):
            assert splits[key] == sorted(splits[key])

    def test_invalid_total_raises(self):
        with pytest.raises(ValueError, match="total_episodes"):
            compute_split_episodes(0, [8, 1, 1], seed=0)

    def test_all_zero_ratio_raises(self):
        with pytest.raises(ValueError, match="ratio must sum"):
            compute_split_episodes(10, [0, 0, 0], seed=0)

    def test_invalid_ratio_length_raises(self):
        with pytest.raises(ValueError, match="2 or 3 elements"):
            compute_split_episodes(10, [1, 2, 3, 4], seed=0)


# =============================================================================
# load_split_info / save_split_info
# =============================================================================


class TestSplitInfoRoundtrip:
    def test_save_then_load(self, tmp_path):
        info = {
            "split_ratio": [8, 1, 1],
            "total_episodes": 50,
            "train_episodes": [0, 1, 2],
            "val_episodes": [3],
            "test_episodes": [4],
        }
        path = tmp_path / "split_info.json"
        save_split_info(path, info)
        assert path.exists()
        loaded = load_split_info(path)
        assert loaded == info

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_split_info(tmp_path / "nonexistent.json") is None

    def test_load_malformed_returns_none_with_warning(self, tmp_path):
        path = tmp_path / "malformed.json"
        path.write_text("{not valid json")
        assert load_split_info(path) is None

    def test_save_creates_readable_json(self, tmp_path):
        info = {"a": 1, "b": [1, 2, 3]}
        path = tmp_path / "out.json"
        save_split_info(path, info)
        # Verify it's valid JSON that round-trips
        assert json.loads(path.read_text()) == info
