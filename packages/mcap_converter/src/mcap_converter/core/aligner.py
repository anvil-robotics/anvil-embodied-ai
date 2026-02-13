"""Time synchronization for multi-sensor data"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.schema import DEFAULT_DATA_CONFIG, DataConfig
from ..utils.image_utils import resize_image


class TimeAligner:
    """
    Align multi-sensor data to unified time axis

    Strategy:
    - Use first available camera as main time axis
    - Find nearest neighbors in time for other sensors
    - Create aligned frames with all sensor data

    Supports both single-robot and multi-robot configurations:
    - Single robot: joint_states_observation, joint_states_action
    - Multi-robot: joint_states_right_observation, joint_states_left_action, etc.

    Example:
        aligner = TimeAligner(config)
        frames = aligner.align_sensors(
            extracted_data,
            camera_names=['head', 'wrist_left']
        )

        for frame in frames:
            print(frame['timestamp'])
            print(frame['observation.state'])  # or 'right.observation.state'
            print(frame['action'])             # or 'right.action'
    """

    def __init__(self, config: DataConfig = DEFAULT_DATA_CONFIG, tolerance_s: float = 1e-3):
        """
        Initialize time aligner

        Args:
            config: Data configuration
            tolerance_s: Time tolerance for synchronization (not currently used)
        """
        self.config = config
        self.tolerance_s = tolerance_s

    def align_sensors(
        self,
        extracted_data: Dict,
        camera_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Align all sensor data to unified time axis

        Args:
            extracted_data: Extracted data from DataExtractor
            camera_names: List of camera names to align

        Returns:
            List of aligned frames, each containing:
            Single robot:
            {
                'timestamp': float,
                'frame_index': int,
                'observation.images.{camera}': np.ndarray,
                'observation.state': np.ndarray,
                'observation.velocity': np.ndarray,  # if in observation_feature_mapping.others
                'observation.effort': np.ndarray,    # if in observation_feature_mapping.others
                'action': np.ndarray,
                'action.velocity': np.ndarray,       # if in action_feature_mapping.others
            }

            Multi-robot:
            {
                'timestamp': float,
                'frame_index': int,
                'observation.images.{camera}': np.ndarray,
                'right.observation.state': np.ndarray,
                'right.observation.velocity': np.ndarray,
                'left.observation.state': np.ndarray,
                'right.action': np.ndarray,
                'left.action': np.ndarray,
            }

        Raises:
            ValueError: If no camera data or joint states available
        """
        # Find first camera with data (main time axis)
        main_cam_name = self._find_main_camera(extracted_data, camera_names)

        if main_cam_name is None:
            raise ValueError("No available camera data found")

        main_cam_info = extracted_data[main_cam_name]
        print(
            f"Using {main_cam_name} as main time axis, total {len(main_cam_info['timestamp'])} frames"
        )

        # Discover robot prefixes and prepare joint data
        robot_prefixes = self._get_robot_prefixes(extracted_data)
        joint_data = self._prepare_all_joint_data(extracted_data, robot_prefixes)

        # Create aligned frames
        synced_frames = []

        for frame_idx, time_ns in enumerate(main_cam_info["timestamp"]):
            frame = self._create_aligned_frame(
                time_ns,
                frame_idx,
                main_cam_name,
                extracted_data,
                camera_names,
                joint_data,
                robot_prefixes,
            )
            synced_frames.append(frame)

        print(f"[OK] Successfully synchronized {len(synced_frames)} frames of data")
        return synced_frames

    def _find_main_camera(self, extracted_data: Dict, camera_names: List[str]) -> Optional[str]:
        """Find first camera with data to use as main time axis"""
        for cam_name in camera_names:
            if cam_name in extracted_data and len(extracted_data[cam_name]["image_data"]) > 0:
                return cam_name
        return None

    def _get_robot_prefixes(self, extracted_data: Dict) -> List[str]:
        """
        Get list of robot prefixes from extracted data keys.

        Returns:
            List of robot prefixes (e.g., ['right', 'left']) or [''] for single robot
        """
        prefixes = set()

        for key in extracted_data.keys():
            if key.startswith("joint_states_"):
                # Parse key: joint_states_{robot}_{role} or joint_states_{role}
                parts = key.replace("joint_states_", "").split("_")
                if len(parts) == 2:
                    # Multi-robot: joint_states_right_observation
                    prefixes.add(parts[0])
                elif len(parts) == 1:
                    # Single robot: joint_states_observation
                    prefixes.add("")

        return sorted(prefixes) if prefixes else [""]

    def _prepare_all_joint_data(
        self, extracted_data: Dict, robot_prefixes: List[str]
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Prepare joint state data for all robots and roles.

        Args:
            extracted_data: Extracted data from DataExtractor
            robot_prefixes: List of robot prefixes

        Returns:
            Dictionary keyed by (robot, role) with prepared joint data
        """
        joint_data = {}

        for robot in robot_prefixes:
            for role in ["observation", "action"]:
                # Build key: joint_states_{robot}_{role} or joint_states_{role}
                if robot:
                    key = f"joint_states_{robot}_{role}"
                else:
                    key = f"joint_states_{role}"

                if key in extracted_data:
                    joint_data[(robot, role)] = self._prepare_joint_data(extracted_data[key], role)

        return joint_data

    def _prepare_joint_data(self, joint_states: Dict, role: str) -> Dict:
        """
        Prepare joint state data for alignment.

        Args:
            joint_states: Joint state data from DataExtractor
            role: 'observation' or 'action'

        Returns:
            Dictionary with timestamp, state, and features
        """
        # Get feature mapping based on role
        if role == "observation":
            feature_mapping = self.config.observation_feature_mapping
        else:
            feature_mapping = self.config.action_feature_mapping

        state_key = feature_mapping.state

        data = {
            "timestamp": joint_states["timestamp"],
            "state": joint_states.get(state_key, np.array([])),
            "features": {},
        }

        # Extract additional features based on role's feature mapping
        for ft_key in feature_mapping.others:
            if ft_key in joint_states and joint_states[ft_key].size > 0:
                data["features"][ft_key] = joint_states[ft_key]

        return data

    def _create_aligned_frame(
        self,
        timestamp: float,
        frame_idx: int,
        main_cam_name: str,
        extracted_data: Dict,
        camera_names: List[str],
        joint_data: Dict[Tuple[str, str], Dict],
        robot_prefixes: List[str],
    ) -> Dict[str, Any]:
        """
        Create one aligned frame at given timestamp.

        Args:
            timestamp: Target timestamp for alignment
            frame_idx: Frame index in main camera
            main_cam_name: Name of main camera
            extracted_data: Full extracted data
            camera_names: List of camera names
            joint_data: Prepared joint data keyed by (robot, role)
            robot_prefixes: List of robot prefixes

        Returns:
            Dictionary with aligned frame data
        """
        frame_data = {
            "timestamp": float(timestamp),
            "frame_index": frame_idx,
        }

        # Resize target size from config (width, height)
        target_size = tuple(self.config.image_resolution)  # (width, height)

        # Add main camera image (resize to target resolution)
        main_img = extracted_data[main_cam_name]["image_data"][frame_idx]
        resized_main_img = resize_image(main_img, target_size)
        frame_data[f"observation.images.{main_cam_name}"] = resized_main_img

        # Add other camera images (find closest in time and resize)
        for cam_name in camera_names:
            if cam_name == main_cam_name:
                continue

            if cam_name not in extracted_data:
                continue

            cam_info = extracted_data[cam_name]
            if cam_info["timestamp"].size == 0:
                continue

            closest_idx = np.argmin(np.abs(cam_info["timestamp"] - timestamp))
            cam_img = cam_info["image_data"][closest_idx]
            resized_cam_img = resize_image(cam_img, target_size)
            frame_data[f"observation.images.{cam_name}"] = resized_cam_img

        # Add joint states for each robot
        for robot in robot_prefixes:
            self._add_observation_data(frame_data, joint_data, robot, timestamp)
            self._add_action_data(frame_data, joint_data, robot, timestamp)

        return frame_data

    def _add_observation_data(
        self,
        frame_data: Dict,
        joint_data: Dict[Tuple[str, str], Dict],
        robot: str,
        timestamp: float,
    ):
        """
        Add observation data to frame.

        Args:
            frame_data: Frame data dictionary to update
            joint_data: Prepared joint data
            robot: Robot prefix ('' for single robot)
            timestamp: Target timestamp
        """
        key = (robot, "observation")
        if key not in joint_data:
            return

        obs_data = joint_data[key]
        if obs_data["timestamp"].size == 0:
            raise ValueError(f"No observation data for robot '{robot or 'default'}'")

        obs_idx = int(np.argmin(np.abs(obs_data["timestamp"] - timestamp)))

        # Build key prefix: 'right.observation' or 'observation'
        if robot:
            prefix = f"{robot}.observation"
        else:
            prefix = "observation"

        # Add state
        frame_data[f"{prefix}.state"] = obs_data["state"][obs_idx].astype(np.float32).copy()

        # Add optional features
        for ft_key, ft_values in obs_data["features"].items():
            frame_data[f"{prefix}.{ft_key}"] = ft_values[obs_idx].astype(np.float32).copy()

    def _add_action_data(
        self,
        frame_data: Dict,
        joint_data: Dict[Tuple[str, str], Dict],
        robot: str,
        timestamp: float,
    ):
        """
        Add action data to frame.

        Args:
            frame_data: Frame data dictionary to update
            joint_data: Prepared joint data
            robot: Robot prefix ('' for single robot)
            timestamp: Target timestamp
        """
        key = (robot, "action")
        if key not in joint_data:
            return

        action_data = joint_data[key]
        if action_data["timestamp"].size == 0:
            raise ValueError(f"No action data for robot '{robot or 'default'}'")

        action_idx = int(np.argmin(np.abs(action_data["timestamp"] - timestamp)))

        # Build key: 'right.action' or 'action'
        if robot:
            action_key = f"{robot}.action"
        else:
            action_key = "action"

        # Add state as action
        frame_data[action_key] = action_data["state"][action_idx].astype(np.float32).copy()

        # Add optional action features
        for ft_key, ft_values in action_data["features"].items():
            frame_data[f"{action_key}.{ft_key}"] = ft_values[action_idx].astype(np.float32).copy()

    def interpolate_missing(self, data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Interpolate missing data points (future feature)

        Args:
            data: Data array with possible missing values
            timestamps: Timestamps for data points

        Returns:
            Interpolated data array
        """
        # TODO: Implement interpolation for missing data
        # For now, just use nearest neighbor (already done in _create_aligned_frame)
        return data
