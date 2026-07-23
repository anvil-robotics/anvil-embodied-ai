"""LeRobot dataset writer"""

import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, List

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..config.schema import DEFAULT_DATA_CONFIG, DataConfig


def _patch_resume_video_continuation(dataset: LeRobotDataset) -> None:
    """
    Work around a chunk-continuation bug in lerobot 0.5.1: after
    LeRobotDataset.resume(), LeRobotDatasetMetadata.latest_episode is None
    (never restored from existing metadata). Two independent pieces of
    lerobot's own code branch on this same None sentinel, but need OPPOSITE
    behavior on resume:

    - dataset_metadata.py's _save_episode_metadata() correctly and
      intentionally treats `latest_episode is None` + existing episodes as
      "we are resuming — open a new parquet metadata file" (to avoid
      overwriting the existing one). This is correct and must not be touched.
    - dataset_writer.py's _save_episode_video() uses the SAME sentinel to
      decide whether to start a brand-new video chunk file. On resume this
      needlessly opens a new video file for the very first post-resume
      episode, even when the existing file is nowhere near the size cap —
      exactly the same continuation behavior that happens for every OTHER
      episode once latest_episode is populated for real.

    Fix: for the first save_episode() call after resume only, synthesize a
    `latest_episode`-shaped dict (sourced from `meta.episodes[-1]`, one
    entry per camera, added lazily as each camera's video is processed) so
    the video-continuation check succeeds and appends to the existing file.
    Immediately before the metadata write for that same episode runs, reset
    `latest_episode` back to None so the parquet-file-continuation logic is
    completely unaffected and still correctly opens a new metadata file.
    After that first episode, `latest_episode` is populated for real by
    lerobot's own code and both patches become permanent no-ops.

    No-ops entirely for a fresh (non-resumed) dataset, where `meta.episodes`
    is empty/None and this bug cannot occur.

    WARNING — depends on private lerobot internals, not a stable public API:
    This function monkeypatches two PRIVATE, underscore-prefixed methods —
    `dataset.writer._save_episode_video` and `dataset.meta.save_episode`'s
    internal branching on `meta.latest_episode` — that are internal
    implementation details of the third-party `lerobot` package, not a
    contract lerobot guarantees to keep stable across versions. This repo
    currently pins `lerobot~=0.5.0` (see
    packages/mcap_converter/pyproject.toml). If that pin is ever bumped
    (including a patch/minor bump within the `~=0.5.0` range, or a move to
    0.6.x+), this function MUST be re-verified against the new version's
    `dataset_writer.py` (`_save_episode_video`) and `dataset_metadata.py`
    (`save_episode` / `latest_episode` handling) source, since lerobot could
    silently rename or restructure these internals without any warning from
    a normal `uv sync` — the failure mode would only surface later, either
    as a silent behavioral regression (extra video chunk files reappear) or
    a `KeyError`/`AttributeError` at conversion time.

    `tests/unit/mcap_converter/test_resume_video_continuation.py` is the
    regression test that would catch a real behavioral break here, and it
    catches more than it might seem at first:
    - If lerobot renames or removes `_save_episode_video` entirely, the line
      `original_save_episode_video = dataset.writer._save_episode_video`
      above would raise `AttributeError` immediately when this function
      runs — a loud, immediate failure, not a silent one. The regression
      test would fail too (as would every `--resume` conversion), so this
      class of drift is well covered.
    - The narrower, genuinely NOT-fully-covered case is a subtler internal
      change: `_save_episode_video`/`save_episode` keep their names and
      signatures, but lerobot alters what `latest_episode` needs to contain
      or how it's consumed internally (e.g. adds/renames a required key
      beyond `chunk_index`/`file_index`/`to_timestamp`, or changes the
      None-sentinel semantics this patch relies on). That WOULD still be
      caught by this regression test (it asserts resumed vs. single-pass
      video file counts match, so wrong/missing keys or broken continuation
      logic would surface as a count mismatch or an exception) — but only
      when the test suite is actually run against the new lerobot version.
      Nothing in `uv sync` or normal CI dependency resolution would trigger
      that test run automatically on a version bump, so the real risk is
      process (bumping the pin without re-running/re-verifying this
      specific test against it), not a fundamental blind spot in the test
      itself.
    """
    meta = dataset.meta
    if meta.episodes is None or len(meta.episodes) == 0:
        return  # fresh dataset — nothing to patch

    state = {"needs_reset": False}

    original_save_episode_video = dataset.writer._save_episode_video

    def patched_save_episode_video(video_key, episode_index, temp_path=None):
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        # NOTE: to_timestamp is also required here — the continuation branch
        # of lerobot's _save_episode_video() reads
        # latest_episode[f"videos/{video_key}/to_timestamp"][0] to compute
        # the running duration offset for the next episode. This key was
        # missing from the originally specified fix and was discovered while
        # testing (KeyError without it); it belongs to the same
        # meta.episodes[-1] row as chunk_index/file_index, so it is added
        # here using the same lazy, per-camera, list-wrapped pattern.
        to_timestamp_key = f"videos/{video_key}/to_timestamp"
        if meta.latest_episode is None or chunk_key not in meta.latest_episode:
            last_episode = meta.episodes[-1]
            if meta.latest_episode is None:
                meta.latest_episode = {}
                state["needs_reset"] = True
            meta.latest_episode[chunk_key] = [last_episode[chunk_key]]
            meta.latest_episode[file_key] = [last_episode[file_key]]
            meta.latest_episode[to_timestamp_key] = [last_episode[to_timestamp_key]]
        return original_save_episode_video(video_key, episode_index, temp_path=temp_path)

    dataset.writer._save_episode_video = patched_save_episode_video

    original_meta_save_episode = meta.save_episode

    def patched_meta_save_episode(*args, **kwargs):
        if state["needs_reset"]:
            meta.latest_episode = None
            state["needs_reset"] = False
        return original_meta_save_episode(*args, **kwargs)

    meta.save_episode = patched_meta_save_episode


class LeRobotWriter:
    """
    Write data to LeRobot v3.0 format dataset

    Manages LeRobot dataset lifecycle:
    1. Initialize dataset with features
    2. Add episodes sequentially
    3. Finalize dataset (write metadata)

    Supports both single-robot and multi-robot configurations:
    - Single robot: observation.state, action
    - Multi-robot: right.observation.state, right.action, left.observation.state, etc.

    Example:
        writer = LeRobotWriter(
            output_dir="data/processed/my_dataset",
            repo_id="anvil_robot/my_dataset",
            robot_type="anvil_openarm",
            fps=30,
        )

        # Single robot
        dataset = writer.create_dataset(
            joint_names={"": ["joint1", "joint2"]},  # empty string for single robot
            camera_names=["head"]
        )

        # Multi-robot
        dataset = writer.create_dataset(
            joint_names={"right": ["joint1", "joint2"], "left": ["joint1", "joint2"]},
            camera_names=["head"]
        )

        for episode_frames in episodes:
            writer.add_episode(dataset, episode_frames)

        writer.finalize(dataset)
    """

    def __init__(
        self,
        output_dir: str,
        repo_id: str,
        robot_type: str = "anvil_openarm",
        fps: int = 30,
        config: DataConfig = DEFAULT_DATA_CONFIG,
        vcodec: str = "h264",
        quiet: bool = False,
    ):
        """
        Initialize LeRobot writer

        Args:
            output_dir: Output directory for dataset
            repo_id: HuggingFace repository ID (e.g., "anvil_robot/dataset_name")
            robot_type: Robot type identifier
            fps: Video frames per second
            config: Data configuration
            vcodec: Video codec for encoding ("h264", "hevc", or "libsvtav1")
            quiet: If True, suppress all print output (default: False)
        """
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.config = config
        self.vcodec = vcodec
        self.quiet = quiet

    def _video_kwargs(self, fn: Any) -> dict[str, Any]:
        """Return video-encoding kwargs supported by the installed LeRobot API."""
        params = inspect.signature(fn).parameters
        if "rgb_encoder" in params:
            from lerobot.configs.video import RGBEncoderConfig

            return {"rgb_encoder": RGBEncoderConfig(vcodec=self.vcodec)}
        if "vcodec" in params:
            return {"vcodec": self.vcodec}
        return {}

    def create_dataset(
        self,
        joint_names: Dict[str, List[str]],
        camera_names: List[str],
    ) -> LeRobotDataset:
        """
        Create new LeRobot dataset

        Args:
            joint_names: Dictionary mapping robot prefix to joint names
                         - Single robot: {"": ["joint1", "joint2"]}
                         - Multi-robot: {"right": ["joint1", "joint2"], "left": ["joint1"]}
            camera_names: List of camera names

        Returns:
            Initialized LeRobotDataset instance
        """
        # Define features
        features = self._define_features(joint_names, camera_names)

        if not self.quiet:
            print("\n=== Creating LeRobot dataset (latest format) ===")
            print(f"Output: {self.output_dir}")
            print(f"Repo ID: {self.repo_id}")
            print(f"Features: {list(features.keys())}")

        # Create dataset
        # Note: root is the output directory itself (not parent)
        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=str(self.output_dir),
            robot_type=self.robot_type,
            features=features,
            use_videos=True,
            **self._video_kwargs(LeRobotDataset.create),
        )

        return dataset

    def add_episode(
        self,
        dataset: LeRobotDataset,
        episode_frames: List[Dict[str, Any]],
        episode_index: int = None,
    ):
        """
        Add one episode to dataset

        Args:
            dataset: LeRobotDataset instance
            episode_frames: List of aligned frames from TimeAligner
            episode_index: Episode index (for progress reporting)
        """
        if episode_index is not None and not self.quiet:
            print(f"\nAdding episode {episode_index + 1}")
            print(f"  About to write {len(episode_frames)} frames")

        # Add frames
        for frame_idx, frame_data in enumerate(episode_frames):
            dataset.add_frame(frame_data)

            # Progress reporting
            if episode_index is not None and not self.quiet and (frame_idx + 1) % 100 == 0:
                progress = (frame_idx + 1) / len(episode_frames) * 100
                print(
                    f"    Processing progress: {frame_idx + 1}/{len(episode_frames)} ({progress:.1f}%)"
                )

        # Save episode
        if not self.quiet:
            print("  - Saving episode and encoding images...")
        dataset.save_episode()

    def finalize(self, dataset: LeRobotDataset):
        """
        Finalize dataset (write metadata and close files)

        Based on LeRobot v3.0 official recommendations, must call finalize()
        after all episodes to write parquet footer with episodes metadata,
        otherwise dataset will be missing meta/episodes/ and cause
        dataset[0] and similar operations to fail

        Args:
            dataset: LeRobotDataset instance
        """
        if not self.quiet:
            print("  - Finalizing dataset (writing metadata and closing parquet)...")
        dataset.finalize()

        # Clean up temporary images directory
        if not self.quiet:
            print("  - Cleaning up temporary images directory...")
        images_tmp_dir = self.output_dir / "images"
        if images_tmp_dir.exists():
            shutil.rmtree(images_tmp_dir)
            if not self.quiet:
                print("    [OK] Cleaned images/")

    def _define_features(
        self,
        joint_names: Dict[str, List[str]],
        camera_names: List[str],
    ) -> Dict[str, Any]:
        """
        Define dataset features based on data

        For multi-robot setups (left/right), creates combined features:
        - observation.state: [left_joints..., right_joints...] (concatenated)
        - action: [left_joints..., right_joints...] (concatenated)

        Args:
            joint_names: Dictionary mapping robot prefix to joint names
                         - Single robot: {"": ["joint1", "joint2"]}
                         - Multi-robot: {"right": ["joint1"], "left": ["joint1"]}
            camera_names: List of camera names

        Returns:
            Features dictionary for LeRobotDataset
        """
        features = {}

        # Image features (shared across all robots)
        # Get resolution from config: [width, height] -> (height, width) for shape
        img_width, img_height = self.config.image_resolution
        for cam_name in camera_names:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": (3, img_height, img_width),  # (channels, height, width)
                "names": ["channel", "height", "width"],
            }

        # Check if multi-robot (has named robots like 'left', 'right')
        robots = sorted([r for r in joint_names.keys() if r])

        if robots:
            # Multi-robot: create combined features
            # Concatenate joint names in sorted order (left, right)
            all_joint_names = []
            for robot in robots:
                # Prefix joint names with robot identifier
                robot_joints = [f"{robot}_{name}" for name in joint_names[robot]]
                all_joint_names.extend(robot_joints)

            num_joints = len(all_joint_names)

            # Combined observation state
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (num_joints,),
                "names": all_joint_names,
            }

            # Combined action
            features["action"] = {
                "dtype": "float32",
                "shape": (num_joints,),
                "names": all_joint_names,
            }

            # Additional observation features (from observation_feature_mapping.others)
            for ft_key in self.config.observation_feature_mapping.others:
                features[f"observation.{ft_key}"] = {
                    "dtype": "float32",
                    "shape": (num_joints,),
                    "names": all_joint_names,
                }

            # Additional action features (from action_feature_mapping.others)
            for ft_key in self.config.action_feature_mapping.others:
                features[f"action.{ft_key}"] = {
                    "dtype": "float32",
                    "shape": (num_joints,),
                    "names": all_joint_names,
                }
        else:
            # Single robot: use original naming (no prefix)
            names = joint_names.get("", [])
            num_joints = len(names)

            features["observation.state"] = {
                "dtype": "float32",
                "shape": (num_joints,),
                "names": names,
            }

            features["action"] = {
                "dtype": "float32",
                "shape": (num_joints,),
                "names": names,
            }

            # Additional observation features
            for ft_key in self.config.observation_feature_mapping.others:
                features[f"observation.{ft_key}"] = {
                    "dtype": "float32",
                    "shape": (num_joints,),
                    "names": names,
                }

            # Additional action features
            for ft_key in self.config.action_feature_mapping.others:
                features[f"action.{ft_key}"] = {
                    "dtype": "float32",
                    "shape": (num_joints,),
                    "names": names,
                }

        return features

    def load_dataset_for_writing(self) -> LeRobotDataset:
        """
        Load an existing dataset in write mode for appending new episodes.

        Uses LeRobotDataset.resume() to load existing metadata and prepare
        the dataset for writing. Use this when --resume is set and
        the output directory already contains a partial conversion.

        Returns:
            LeRobotDataset instance ready to accept add_frame / save_episode calls
        """
        dataset = LeRobotDataset.resume(
            repo_id=self.repo_id,
            root=str(self.output_dir),
            **self._video_kwargs(LeRobotDataset.resume),
        )
        _patch_resume_video_continuation(dataset)
        return dataset

    def __repr__(self) -> str:
        return f"LeRobotWriter(output_dir='{self.output_dir}', repo_id='{self.repo_id}')"
