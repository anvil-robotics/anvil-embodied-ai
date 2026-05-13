"""LeRobot dataset writer"""

import shutil
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..config import DataConfig
from .constants import OBSERVATION_EXTRAS


class LeRobotWriter:
    """
    Write data to LeRobot v3.0 format datasets.

    Per-arm joint features are concatenated in sorted arm order (left, right) into
    flat `observation.state` / `observation.velocity` / `observation.effort` /
    `action` arrays. Each entry's name is prefixed with the arm.
    """

    def __init__(
        self,
        output_dir: str,
        repo_id: str,
        robot_type: str = "anvil_openarm",
        frequency: int = 30,
        config: DataConfig | None = None,
        vcodec: str = "h264",
        quiet: bool = False,
    ):
        if config is None:
            raise ValueError("config is required")
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.frequency = frequency
        self.config = config
        self.vcodec = vcodec
        self.quiet = quiet

    def create_dataset(
        self,
        joint_names: dict[str, list[str]],
        camera_names: list[str],
    ) -> LeRobotDataset:
        """Create a new LeRobot dataset.

        Args:
            joint_names: {arm -> [joint_id, ...]} for each arm in `config.arms`.
            camera_names: dataset camera names (values of camera_topic_mapping).
        """
        features = self._define_features(joint_names, camera_names)

        if not self.quiet:
            print("\n=== Creating LeRobot dataset (latest format) ===")
            print(f"Output: {self.output_dir}")
            print(f"Repo ID: {self.repo_id}")
            print(f"Features: {list(features.keys())}")

        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.frequency,  # lerobot API uses `fps` for the dataset sample rate
            root=str(self.output_dir),
            robot_type=self.robot_type,
            features=features,
            use_videos=True,
            vcodec=self.vcodec,
        )

    def add_episode(
        self,
        dataset: LeRobotDataset,
        episode_frames: list[dict[str, Any]],
        episode_index: int | None = None,
    ) -> None:
        if episode_index is not None and not self.quiet:
            print(f"\nAdding episode {episode_index + 1}")
            print(f"  About to write {len(episode_frames)} frames")

        for frame_idx, frame_data in enumerate(episode_frames):
            dataset.add_frame(frame_data)
            if episode_index is not None and not self.quiet and (frame_idx + 1) % 100 == 0:
                progress = (frame_idx + 1) / len(episode_frames) * 100
                print(
                    f"    Processing progress: {frame_idx + 1}/{len(episode_frames)} "
                    f"({progress:.1f}%)"
                )

        if not self.quiet:
            print("  - Saving episode and encoding images...")
        dataset.save_episode()

    def finalize(self, dataset: LeRobotDataset) -> None:
        if not self.quiet:
            print("  - Finalizing dataset (writing metadata and closing parquet)...")
        dataset.finalize()

        if not self.quiet:
            print("  - Cleaning up temporary images directory...")
        images_tmp_dir = self.output_dir / "images"
        if images_tmp_dir.exists():
            shutil.rmtree(images_tmp_dir)
            if not self.quiet:
                print("    [OK] Cleaned images/")

    def _define_features(
        self,
        joint_names: dict[str, list[str]],
        camera_names: list[str],
    ) -> dict[str, Any]:
        features: dict[str, Any] = {}

        img_width, img_height = self.config.image_resolution
        for cam_name in camera_names:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": (3, img_height, img_width),
                "names": ["channel", "height", "width"],
            }

        # Concatenate joint names in sorted-arm order, prefixed with arm.
        arms = sorted(joint_names.keys())
        all_joint_names: list[str] = []
        for arm in arms:
            all_joint_names.extend(f"{arm}_{j}" for j in joint_names[arm])
        num_joints = len(all_joint_names)

        for key in ("observation.state", "action", *(f"observation.{x}" for x in OBSERVATION_EXTRAS)):
            features[key] = {
                "dtype": "float32",
                "shape": (num_joints,),
                "names": all_joint_names,
            }

        return features

    def load_dataset_for_writing(self) -> LeRobotDataset:
        return LeRobotDataset.resume(
            repo_id=self.repo_id,
            root=str(self.output_dir),
            vcodec=self.vcodec,
        )

    def __repr__(self) -> str:
        return f"LeRobotWriter(output_dir='{self.output_dir}', repo_id='{self.repo_id}')"
