"""
Base Protocol for Inference Strategies

Defines the interface that all observation acquisition strategies must implement.
"""

from typing import Protocol, Dict, List, Optional, Any
import torch


class InferenceStrategy(Protocol):
    """
    Protocol for observation acquisition strategies.

    Implementations handle the details of:
    - Setting up ROS2 subscriptions (camera images, joint states)
    - Managing observation data (storage, synchronization)
    - Providing observations for model inference

    Two implementations:
    - MultiProcessStrategy (mp): Shared memory + worker processes, better isolation
    - SingleProcessStrategy (single): Callbacks + threading, simpler for debugging
    """

    def setup(
        self,
        node: Any,
        config: dict,
        camera_mapping: Dict[str, str],
        joint_names_config: dict,
        joint_state_topic: str,
        image_shape: tuple,
        metrics: Any = None,
    ) -> None:
        """
        Initialize the strategy with ROS2 node and configuration.

        Args:
            node: ROS2 node for creating subscriptions/publishers
            config: Full configuration dictionary
            camera_mapping: Dict mapping ROS topic -> ML camera name
            joint_names_config: Joint naming configuration
            joint_state_topic: Topic for joint state messages
            image_shape: Expected image shape (H, W, C)
            metrics: Optional MetricsTracker for recording stats
        """
        ...

    def get_observation(
        self,
        camera_names: List[str],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get observation if complete, else None.

        Returns a dictionary with keys like:
        - 'observation.state': Joint positions tensor
        - 'observation.images.{camera_name}': Image tensor for each camera

        Args:
            camera_names: List of camera names required for complete observation

        Returns:
            Observation dict if all sensors have data, None otherwise
        """
        ...

    def get_current_joint_positions(self) -> Dict[str, float]:
        """
        Get current joint positions for delta action limiting.

        Returns:
            Dict mapping joint name -> position value
        """
        ...

    def get_incomplete_reason(self) -> str:
        """
        Get human-readable reason why observation is incomplete.

        Useful for debugging when inference is skipped.

        Returns:
            Description of what's missing (e.g., "waiting for cameras: ['waist']")
        """
        ...

    def record_metrics(self, metrics_tracker: Any) -> None:
        """
        Record any strategy-specific metrics.

        Called periodically to update metrics for stats logging.

        Args:
            metrics_tracker: MetricsTracker instance
        """
        ...

    def cleanup(self) -> None:
        """
        Release resources (workers, shared memory, etc).

        Called when the node is being destroyed.
        """
        ...
