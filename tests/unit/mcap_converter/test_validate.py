"""Tests for mcap_converter.cli.validate — the dataset-valid CLI.

Covers the raw-frame-printing feature merged in from the former standalone
`dataset-inspect` command (`print_raw_frames`, gated behind --print-episodes), and the
CLI wiring (`--print-episodes 0` = old default behavior only, `> 0` also prints).
`test_dataset()` itself (the pre-existing LeRobotDataset load/read checks) is untouched
and not re-tested here — it requires a fully valid, video-backed LeRobotDataset, which is
exactly what `print_raw_frames` was designed to avoid needing.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mcap_converter.cli.validate import main, parse_args, print_raw_frames


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "ee-space-testing"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 4,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8], "names": [
                "right_x", "right_y", "right_z",
                "right_qx", "right_qy", "right_qz", "right_qw", "right_gripper",
            ]},
            "action": {"dtype": "float32", "shape": [10], "names": [
                "right_x", "right_y", "right_z",
                "right_r0", "right_r1", "right_r2", "right_r3", "right_r4", "right_r5",
                "right_gripper",
            ]},
            "observation.images.chest": {"dtype": "video", "shape": [3, 480, 640]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    rows = []
    for ep in range(2):
        for f in range(2):
            rows.append({
                "observation.state": np.arange(8, dtype=np.float32) + ep * 100 + f,
                "action": np.zeros(10, dtype=np.float32) if f == 0 else np.arange(10, dtype=np.float32),
                "timestamp": float(f) / 30.0,
                "frame_index": f,
                "episode_index": ep,
                "index": ep * 2 + f,
                "task_index": 0,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    return root


# ---------------------------------------------------------------------------
# print_raw_frames — direct tests (no LeRobotDataset needed)
# ---------------------------------------------------------------------------


def test_print_raw_frames_labels_by_feature_name(tiny_dataset, capsys):
    ok = print_raw_frames(str(tiny_dataset), n_episodes=2, n_frames=2)
    out = capsys.readouterr().out
    assert ok
    assert "=== Episode 0 " in out
    assert "=== Episode 1 " in out
    assert "right_qw" in out
    assert "right_gripper" in out


def test_print_raw_frames_no_labels_falls_back_to_indices(tiny_dataset, capsys):
    print_raw_frames(str(tiny_dataset), n_episodes=1, n_frames=1, no_labels=True)
    out = capsys.readouterr().out
    assert "right_qw" not in out
    assert "[ 0]" in out or "[0]" in out.replace(" ", "")


def test_print_raw_frames_caps_at_actual_episode_and_frame_counts(tiny_dataset, capsys):
    """Requesting more episodes/frames than exist must not crash — just show what's there."""
    ok = print_raw_frames(str(tiny_dataset), n_episodes=50, n_frames=50)
    out = capsys.readouterr().out
    assert ok
    assert "=== Episode 0 " in out
    assert "=== Episode 1 " in out
    assert "Episode 2" not in out  # only 2 episodes exist


def test_print_raw_frames_missing_dataset_fails_gracefully():
    assert print_raw_frames("/nonexistent/path/xyz", n_episodes=1, n_frames=1) is False


def test_print_raw_frames_excludes_video_features(tiny_dataset, capsys):
    """Image/video features are never printed — not parquet columns at all (stored as
    separate video files), and excluded from the summary listing too."""
    print_raw_frames(str(tiny_dataset), n_episodes=1, n_frames=1)
    out = capsys.readouterr().out
    assert "observation.images.chest" not in out
    assert "vector features found: ['observation.state', 'action']" in out


def test_print_raw_frames_respects_columns_filter(tiny_dataset, capsys):
    """--columns restricts which parquet columns are READ and printed per-frame — the
    summary header (from meta/info.json, independent of the filter) still lists every
    feature the dataset makes available, so check the per-frame BLOCK specifically, not
    the whole output."""
    ok = print_raw_frames(str(tiny_dataset), n_episodes=1, n_frames=1,
                           columns="action,frame_index,episode_index")
    out = capsys.readouterr().out
    assert ok
    assert "  observation.state:\n" not in out  # no per-frame block for the excluded column
    assert "right_r0" in out  # from action, which was requested


# ---------------------------------------------------------------------------
# CLI wiring — --print-episodes/--print-frames default to 3/5 (raw-frame printing is
# ON by default); passing --print-episodes 0 explicitly opts back out.
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_three_episodes_five_frames():
    args = parse_args(["--root", "/tmp/whatever"])
    assert args.print_episodes == 3
    assert args.print_frames == 5


def test_main_prints_raw_frames_by_default(tmp_path):
    """Default behavior (no --print-episodes passed) must call print_raw_frames with the
    3/5 defaults — raw-frame printing is on by default now, not opt-in."""
    with patch("mcap_converter.cli.validate.test_dataset", return_value=True), \
         patch("mcap_converter.cli.validate.print_raw_frames", return_value=True) as mock_print:
        main(["--root", str(tmp_path)])
    mock_print.assert_called_once_with(str(tmp_path), 3, 5, None, False)


def test_main_skips_raw_frames_when_print_episodes_explicitly_zero(tmp_path):
    """Explicit --print-episodes 0 must still fully opt out — the escape hatch preserved
    from before the default flipped."""
    with patch("mcap_converter.cli.validate.test_dataset", return_value=True) as mock_test, \
         patch("mcap_converter.cli.validate.print_raw_frames") as mock_print:
        main(["--root", str(tmp_path), "--print-episodes", "0"])
    mock_test.assert_called_once()
    mock_print.assert_not_called()


def test_main_calls_print_raw_frames_with_custom_counts(tmp_path):
    with patch("mcap_converter.cli.validate.test_dataset", return_value=True), \
         patch("mcap_converter.cli.validate.print_raw_frames", return_value=True) as mock_print:
        main(["--root", str(tmp_path), "--print-episodes", "10", "--print-frames", "7"])
    mock_print.assert_called_once_with(str(tmp_path), 10, 7, None, False)


def test_main_exits_nonzero_if_print_raw_frames_fails(tmp_path):
    with patch("mcap_converter.cli.validate.test_dataset", return_value=True), \
         patch("mcap_converter.cli.validate.print_raw_frames", return_value=False), \
         pytest.raises(SystemExit) as exc_info:
        main(["--root", str(tmp_path), "--print-episodes", "1"])
    assert exc_info.value.code == 1
