"""Observation management for LeRobot inference.

Manages and aggregates observations from multiple sensors (cameras, joints)
and prepares them in the format expected by LeRobot models.
"""

import time

import numpy as np
import torch
from sensor_msgs.msg import JointState


class ObservationManager:
    """
    Manage and aggregate observations from multiple sensors.

    Collects:
    - Joint states (position, velocity, effort)
    - Camera images

    Prepares observations in format expected by LeRobot models.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize observation manager.

        Args:
            device: Device for tensors ("cuda" or "cpu")
        """
        self.device = device
        self.latest_joint_state: JointState | None = None
        self.latest_images: dict[str, np.ndarray] = {}

        # Timestamps for time sync validation
        self._joint_timestamp: float | None = None
        self._image_timestamps: dict[str, float] = {}

    def update_joint_state(self, msg: JointState):
        """
        Update latest joint state.

        Args:
            msg: JointState message from ROS2
        """
        self.latest_joint_state = msg
        self._joint_timestamp = time.time()

    def update_image(self, camera_name: str, image: np.ndarray):
        """
        Update latest camera image.

        Args:
            camera_name: Name of the camera
            image: Image as numpy array (H, W, C)
        """
        self.latest_images[camera_name] = image
        self._image_timestamps[camera_name] = time.time()

    def has_complete_observation(self, required_cameras: list[str]) -> bool:
        """
        Check if we have all required observations.

        Args:
            required_cameras: List of required camera names

        Returns:
            True if all observations available
        """
        if self.latest_joint_state is None:
            return False

        for cam in required_cameras:
            if cam not in self.latest_images:
                return False

        return True

    def get_observation(
        self, camera_names: list[str], include_velocity: bool = True, include_effort: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Get observation dictionary for model inference.

        Images are returned normalized to [0, 1] range.

        Args:
            camera_names: List of camera names to include
            include_velocity: Whether to include velocity
            include_effort: Whether to include effort

        Returns:
            Dictionary with format:
            {
                'observation.state': torch.Tensor [1, num_joints],
                'observation.images.{camera}': torch.Tensor [1, 3, H, W],
                'observation.velocity': torch.Tensor [1, num_joints],
                'observation.effort': torch.Tensor [1, num_joints],
            }
        """
        if not self.has_complete_observation(camera_names):
            raise ValueError("Incomplete observation")

        observation = {}

        # Joint positions
        positions = self._get_joint_values("position")
        observation["observation.state"] = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Velocities
        if include_velocity:
            velocities = self._get_joint_values("velocity")
            if velocities:
                observation["observation.velocity"] = torch.tensor(
                    velocities, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

        # Efforts
        if include_effort:
            efforts = self._get_joint_values("effort")
            if efforts:
                observation["observation.effort"] = torch.tensor(
                    efforts, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

        # Camera images
        for cam_name in camera_names:
            if cam_name in self.latest_images:
                img_tensor = self._image_to_tensor(self.latest_images[cam_name])
                observation[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(
                    self.device
                )

        return observation

    def _get_joint_values(self, field: str) -> list[float]:
        """
        Get joint values for specified field.

        Args:
            field: Field name ('position', 'velocity', or 'effort')

        Returns:
            List of values, empty if field not available
        """
        if self.latest_joint_state is None:
            return []

        data = getattr(self.latest_joint_state, field, None)
        return list(data) if data else []

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor.

        Args:
            image: Numpy array (H, W, C) in range [0, 255]

        Returns:
            Tensor (C, H, W) in range [0, 1]
        """
        # Normalize to [0, 1] and convert to CHW
        img_float = image.astype(np.float32) / 255.0
        return torch.from_numpy(img_float).permute(2, 0, 1)

    def get_time_drift(self) -> float | None:
        """
        Get maximum time drift between sensor observations.

        Returns:
            Maximum time difference in seconds between any two sensors,
            or None if not enough data
        """
        timestamps = []

        if self._joint_timestamp is not None:
            timestamps.append(self._joint_timestamp)

        timestamps.extend(self._image_timestamps.values())

        if len(timestamps) < 2:
            return None

        return max(timestamps) - min(timestamps)
