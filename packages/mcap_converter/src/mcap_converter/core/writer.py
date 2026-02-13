"""LeRobot dataset writer"""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..config.schema import DEFAULT_DATA_CONFIG, DataConfig


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
        """
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.config = config
        self.vcodec = vcodec

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
            vcodec=self.vcodec,
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
        if episode_index is not None:
            print(f"\nAdding episode {episode_index + 1}")
            print(f"  About to write {len(episode_frames)} frames")

        # Add frames
        for frame_idx, frame_data in enumerate(episode_frames):
            dataset.add_frame(frame_data)

            # Progress reporting
            if episode_index is not None and (frame_idx + 1) % 100 == 0:
                progress = (frame_idx + 1) / len(episode_frames) * 100
                print(
                    f"    Processing progress: {frame_idx + 1}/{len(episode_frames)} ({progress:.1f}%)"
                )

        # Save episode
        print("  - Saving episode and encoding images...")
        dataset.save_episode()

        # Stop image writer
        print("  - Stopping image writer...")
        dataset.stop_image_writer()

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
        print("  - Finalizing dataset (writing metadata and closing parquet)...")
        dataset.finalize()

        # Clean up temporary images directory
        print("  - Cleaning up temporary images directory...")
        images_tmp_dir = self.output_dir / "images"
        if images_tmp_dir.exists():
            shutil.rmtree(images_tmp_dir)
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

    def __repr__(self) -> str:
        return f"LeRobotWriter(output_dir='{self.output_dir}', repo_id='{self.repo_id}')"
