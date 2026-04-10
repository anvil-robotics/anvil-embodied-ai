"""Metrics tracking for LeRobot inference.

Tracks input reception rates, inference statistics, and timing information
for monitoring and validation.
"""

import time


class MetricsTracker:
    """
    Tracks input reception and inference statistics.

    Monitors:
    - Image reception rates per camera
    - Joint state reception rate
    - Control loop execution rate
    - Inference count and timing
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self._start_time: float | None = None
        self._image_counts: dict[str, int] = {}
        self._joint_count: int = 0
        self._control_loop_count: int = 0
        self._inference_count: int = 0
        self._action_output_count: int = 0

    def reset(self):
        """Reset all metrics."""
        self._start_time = None
        self._image_counts.clear()
        self._joint_count = 0
        self._control_loop_count = 0
        self._inference_count = 0
        self._action_output_count = 0

    def _ensure_started(self):
        """Start timing if not already started."""
        if self._start_time is None:
            self._start_time = time.time()

    def record_image(self, camera_name: str):
        """
        Record an image reception event.

        Args:
            camera_name: Name of the camera that received the image
        """
        self._ensure_started()
        self._image_counts[camera_name] = self._image_counts.get(camera_name, 0) + 1

    def record_joint_state(self):
        """Record a joint state reception event."""
        self._ensure_started()
        self._joint_count += 1

    def record_control_loop(self):
        """Record a control loop execution."""
        self._ensure_started()
        self._control_loop_count += 1

    def record_inference(self):
        """Record an inference execution."""
        self._ensure_started()
        self._inference_count += 1

    def record_action_output(self):
        """Record an action successfully published to the robot."""
        self._ensure_started()
        self._action_output_count += 1

    def get_elapsed_time(self) -> float:
        """Get elapsed time since tracking started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary containing:
            - elapsed_sec: Time since tracking started
            - image_counts: Raw image counts per camera
            - image_fps: Image reception FPS per camera
            - joint_count: Raw joint state count
            - joint_fps: Joint state reception FPS
            - control_loop_count: Raw control loop count
            - control_loop_fps: Control loop execution FPS
            - inference_count: Raw inference count
            - inference_fps: Inference execution FPS
        """
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            elapsed = 0.001  # Avoid division by zero

        image_fps = {camera: count / elapsed for camera, count in self._image_counts.items()}

        return {
            "elapsed_sec": elapsed,
            "image_counts": dict(self._image_counts),
            "image_fps": image_fps,
            "total_image_fps": sum(image_fps.values()),
            "joint_count": self._joint_count,
            "joint_fps": self._joint_count / elapsed,
            "control_loop_count": self._control_loop_count,
            "control_loop_fps": self._control_loop_count / elapsed,
            "inference_count": self._inference_count,
            "inference_fps": self._inference_count / elapsed,
            "action_output_count": self._action_output_count,
            "action_output_fps": self._action_output_count / elapsed,
        }

    def get_summary(self) -> str:
        """
        Get a human-readable summary of metrics.

        Returns:
            Formatted string with key metrics
        """
        stats = self.get_stats()
        lines = [
            f"Elapsed: {stats['elapsed_sec']:.1f}s",
            f"Control loop: {stats['control_loop_fps']:.1f} Hz",
            f"Joint state: {stats['joint_fps']:.1f} Hz",
            f"Inference: {stats['inference_fps']:.2f} Hz ({stats['inference_count']} calls)",
        ]

        for camera, fps in stats["image_fps"].items():
            lines.append(f"Camera {camera}: {fps:.1f} Hz")

        return "\n".join(lines)
