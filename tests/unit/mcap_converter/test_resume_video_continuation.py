"""Regression test for BUG-002: --resume creating an extra unnecessary video chunk file."""

from pathlib import Path

import pytest

from mcap_converter.cli.convert import convert_session
from mcap_converter.config.loader import ConfigLoader
from mcap_converter.core.writer import _patch_resume_video_continuation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "smoke" / "fixtures" / "test-session"
_FIXTURE_CONFIG = (
    _REPO_ROOT / "tests" / "smoke" / "fixtures" / "configs" / "mcap-converter-smoke-test-cmd.yaml"
)


def _count_video_files(dataset_dir: Path) -> dict:
    """Return {camera_key: file_count} for every camera's video directory."""
    videos_root = dataset_dir / "videos"
    counts = {}
    for camera_dir in sorted(videos_root.iterdir()):
        counts[camera_dir.name] = len(list(camera_dir.rglob("*.mp4")))
    return counts


class TestResumeDoesNotFragmentVideo:
    def test_resuming_mid_conversion_produces_same_video_file_count_as_single_pass(self, tmp_path):
        mcap_files = sorted(_FIXTURE_DIR.glob("*/*.mcap"))
        assert len(mcap_files) >= 4, (
            "fixture must have at least 4 episodes for this test to be meaningful"
        )

        config = ConfigLoader.from_yaml(str(_FIXTURE_CONFIG))

        # Reference: convert all episodes in a single pass.
        single_pass_dir = tmp_path / "single-pass"
        convert_session(
            input_dir=str(_FIXTURE_DIR),
            output_dir=str(single_pass_dir),
            repo_id="local/single-pass",
            robot_type="anvil_openarm",
            fps=30,
            tolerance_s=1e-4,
            task="test",
            config=config,
            mcap_files=mcap_files,
            debug_plot_episodes=0,
        )
        single_pass_counts = _count_video_files(single_pass_dir)

        # Resumed: convert the first half, then resume for the rest.
        resumed_dir = tmp_path / "resumed"
        half = len(mcap_files) // 2
        convert_session(
            input_dir=str(_FIXTURE_DIR),
            output_dir=str(resumed_dir),
            repo_id="local/resumed",
            robot_type="anvil_openarm",
            fps=30,
            tolerance_s=1e-4,
            task="test",
            config=config,
            mcap_files=mcap_files[:half],
            debug_plot_episodes=0,
        )
        convert_session(
            input_dir=str(_FIXTURE_DIR),
            output_dir=str(resumed_dir),
            repo_id="local/resumed",
            robot_type="anvil_openarm",
            fps=30,
            tolerance_s=1e-4,
            task="test",
            config=config,
            mcap_files=mcap_files,
            resume_from=half,
            debug_plot_episodes=0,
        )
        resumed_counts = _count_video_files(resumed_dir)

        assert resumed_counts == single_pass_counts, (
            f"resuming mid-conversion should produce the SAME number of video "
            f"files per camera as a single-pass conversion, but got "
            f"single-pass={single_pass_counts} vs resumed={resumed_counts}"
        )


class TestPatchIsNoOpForFreshDataset:
    """`_patch_resume_video_continuation` must early-exit for a fresh (non-resumed)
    dataset, without touching `dataset.writer._save_episode_video` or
    `dataset.meta.save_episode` at all. This directly targets the early-exit
    condition `meta.episodes is None or len(meta.episodes) == 0`.
    """

    @pytest.mark.parametrize("episodes", [None, []], ids=["episodes=None", "episodes=[]"])
    def test_does_not_patch_when_no_prior_episodes_exist(self, episodes):
        class FakeMeta:
            pass

        class FakeWriter:
            def _save_episode_video(self, *args, **kwargs):
                pass

        class FakeDataset:
            meta = FakeMeta()
            writer = FakeWriter()

        dataset = FakeDataset()
        dataset.meta.episodes = episodes
        # Bound methods compare equal (==) when they wrap the same underlying
        # function on the same instance, but a fresh bound-method object is
        # created on every attribute access, so `is` would spuriously fail
        # even when nothing was patched. `==` is the correct "unchanged" check.
        original_video_fn = dataset.writer._save_episode_video

        # FakeMeta intentionally defines no `save_episode` attribute: if the
        # early-exit did not return before reaching the meta-patching code,
        # accessing `meta.save_episode` would raise AttributeError, failing
        # this test for exactly the right reason.
        _patch_resume_video_continuation(dataset)

        assert dataset.writer._save_episode_video == original_video_fn
        assert not hasattr(dataset.meta, "save_episode")
