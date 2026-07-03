"""Tests for post-conversion output file permissions.

Reproduces a real bug: lerobot's video encoder (PyAV/libavformat, via
encode_video_frames() in lerobot/datasets/video_utils.py) writes video
files with 0600 permissions, bypassing the process umask. Every other
mcap-convert output (.parquet, .json, .png) gets the normal 0644/0664.

This breaks any tool that reads the dataset as a different UID than the
one that ran the conversion -- e.g. nginx in the dataset-viz feature,
which gets a 403 Forbidden serving video files without a fix-up.
"""

import stat
from pathlib import Path

from mcap_converter.cli.convert import convert_session
from mcap_converter.config.loader import ConfigLoader

FIXTURES = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
MCAP_ROOT = FIXTURES / "test-session"
CONVERT_CONFIG = FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml"

# Both group-read and other-read bits.
FILE_READ_BITS = stat.S_IRGRP | stat.S_IROTH
# Group/other read + execute, needed for a directory to be traversable.
DIR_READ_BITS = stat.S_IRGRP | stat.S_IROTH | stat.S_IXGRP | stat.S_IXOTH


def _convert(output_dir: Path):
    config = ConfigLoader.from_yaml(str(CONVERT_CONFIG))
    return convert_session(
        input_dir=str(MCAP_ROOT),
        output_dir=str(output_dir),
        repo_id="test/test-session",
        robot_type="anvil_openarm",
        fps=30,
        task="manipulation",
        config=config,
        config_path=str(CONVERT_CONFIG),
    )


class TestOutputPermissions:
    def test_every_output_file_and_dir_is_group_and_other_readable(self, tmp_path):
        output_dir = tmp_path / "dataset"
        _convert(output_dir)

        video_files = list(output_dir.rglob("*.mp4"))
        assert video_files, (
            "expected at least one .mp4 under output_dir/videos/ -- test is vacuous otherwise"
        )

        bad_dirs = []
        bad_files = []
        for path in output_dir.rglob("*"):
            mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                if mode & DIR_READ_BITS != DIR_READ_BITS:
                    bad_dirs.append((path, oct(mode)))
            else:
                if mode & FILE_READ_BITS != FILE_READ_BITS:
                    bad_files.append((path, oct(mode)))

        assert not bad_dirs, f"directories missing group/other read+execute: {bad_dirs}"
        assert not bad_files, f"files missing group/other read: {bad_files}"

        # Explicitly re-check the actual bug symptom: video files specifically.
        for video in video_files:
            assert video.stat().st_size > 0
            mode = stat.S_IMODE(video.stat().st_mode)
            assert mode & FILE_READ_BITS == FILE_READ_BITS, (
                f"{video} is not group/other readable (mode={oct(mode)})"
            )
