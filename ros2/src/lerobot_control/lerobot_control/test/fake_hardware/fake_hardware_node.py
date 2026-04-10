#!/usr/bin/env python3
"""Mock controller node for integration testing of lerobot_control.

This node simulates a robot controller by:
- Publishing dummy CompressedImage (configurable resolution and FPS)
- Publishing dummy joint states at 500Hz (matches real robot)
- Subscribing to action commands and validating them

ROS2 parameters:
    timeout (float): Seconds before exit with failure (default 30.0)
    required_actions (int): Valid actions needed before exit success (default 10)
    camera_resolution (str): "480p", "720p", or "1080p" (default "480p")
    camera_fps (int): Camera publish rate in Hz (default 30)

The node exits with code 0 after receiving the required number of valid actions,
or exits with code 1 on timeout or invalid data.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray


_RESOLUTION_MAP = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


class MockControllerNode(Node):
    """ROS2 node that simulates a robot controller for testing."""

    def __init__(self):
        super().__init__("mock_controller")

        # Declare parameters
        self.declare_parameter("timeout", 30.0)
        self.declare_parameter("required_actions", 10)
        self.declare_parameter("camera_resolution", "480p")
        self.declare_parameter("camera_fps", 30)

        # Get parameter values
        self._timeout = self.get_parameter("timeout").value
        self._required_actions = self.get_parameter("required_actions").value
        self._camera_res_label = self.get_parameter("camera_resolution").value
        self._camera_fps = self.get_parameter("camera_fps").value

        # Resolve resolution
        h, w = _RESOLUTION_MAP.get(self._camera_res_label, (480, 640))

        self.get_logger().info(
            f"MockControllerNode initialized: timeout={self._timeout}s, "
            f"required_actions={self._required_actions}, "
            f"resolution={w}x{h} ({self._camera_res_label}), camera_fps={self._camera_fps}"
        )

        # Publishers — 4 CompressedImage cameras matching production topics
        self._camera_topics = [
            "/cam_waist/image_raw/compressed",
            "/cam_wrist_r/image_raw/compressed",
            "/cam_chest/image_raw/compressed",
            "/cam_wrist_l/image_raw/compressed",
        ]
        self.image_pubs = [
            self.create_publisher(CompressedImage, topic, 10)
            for topic in self._camera_topics
        ]
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        # Subscribers — one per arm, matching inference_node publish topics
        self._action_subs = [
            self.create_subscription(Float64MultiArray, topic, self.action_callback, 10)
            for topic in [
                "/follower_l_forward_position_controller/commands",
                "/follower_r_forward_position_controller/commands",
            ]
        ]

        # Separate timers: 500Hz joint states, configurable camera FPS
        self.joint_timer = self.create_timer(1.0 / 500.0, self.publish_joint_state)
        self.image_timer = self.create_timer(1.0 / self._camera_fps, self.publish_image)

        # Timeout check timer (1Hz)
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)

        # State
        self.valid_actions_received = 0
        self.start_time = self.get_clock().now()

        # Joint names for 16-DOF robot (8 joints per arm: finger + 7 joints)
        # Naming matches production: follower_{l,r}_{joint_id}
        self.joint_names = [
            "follower_l_finger_joint1",
            "follower_l_joint1",
            "follower_l_joint2",
            "follower_l_joint3",
            "follower_l_joint4",
            "follower_l_joint5",
            "follower_l_joint6",
            "follower_l_joint7",
            "follower_r_finger_joint1",
            "follower_r_joint1",
            "follower_r_joint2",
            "follower_r_joint3",
            "follower_r_joint4",
            "follower_r_joint5",
            "follower_r_joint6",
            "follower_r_joint7",
        ]

        # Random number generator
        self._rng = np.random.default_rng()

        # Pre-generate a dummy image and JPEG-encode once (reuse across frames)
        dummy_rgb = self._rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        _, self._jpeg_data = cv2.imencode(".jpg", dummy_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])

    def publish_image(self):
        """Publish dummy CompressedImage on all 4 cameras at 30Hz."""
        stamp = self.get_clock().now().to_msg()
        data = self._jpeg_data.tobytes()
        for pub in self.image_pubs:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera_link"
            msg.format = "jpeg"
            msg.data = data
            pub.publish(msg)

    def publish_joint_state(self):
        """Publish dummy joint states at 500Hz."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = "base_link"
        joint_msg.name = self.joint_names
        joint_msg.position = (self._rng.random(16) * 2 * np.pi - np.pi).tolist()
        joint_msg.velocity = [0.0] * 16
        joint_msg.effort = [0.0] * 16
        self.joint_pub.publish(joint_msg)

    def check_timeout(self):
        """Check if timeout has been exceeded."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed > self._timeout:
            self.get_logger().error(
                f"Timeout after {elapsed:.1f}s. "
                f"Received {self.valid_actions_received}/{self._required_actions} valid actions."
            )
            raise SystemExit(1)

    def action_callback(self, msg: Float64MultiArray):
        """Handle incoming action commands from per-arm controller topics."""
        # Validate action values are finite
        for i, val in enumerate(msg.data):
            if not np.isfinite(val):
                self.get_logger().error(
                    f"Invalid action value at index {i}: {val} (must be finite)"
                )
                raise SystemExit(1)

        self.valid_actions_received += 1
        self.get_logger().info(
            f"Valid action received [{self.valid_actions_received}/{self._required_actions}]"
        )

        if self.valid_actions_received >= self._required_actions:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.get_logger().info(
                f"Test PASSED! Received {self._required_actions} valid actions in {elapsed:.1f}s"
            )
            raise SystemExit(0)


def main(args=None):
    """Entry point for the mock controller node."""
    rclpy.init(args=args)
    node = MockControllerNode()

    exit_code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0
        node.get_logger().info(f"Node shutting down with exit code {exit_code}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
