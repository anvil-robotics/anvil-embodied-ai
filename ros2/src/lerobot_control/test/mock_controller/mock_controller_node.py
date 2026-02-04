#!/usr/bin/env python3
"""Mock controller node for integration testing of lerobot_control.

This node simulates a robot controller by:
- Publishing dummy camera images at 30Hz
- Publishing dummy joint states at 30Hz
- Subscribing to action commands and validating them

The node exits with code 0 after receiving the required number of valid actions,
or exits with code 1 on timeout or invalid data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


class MockControllerNode(Node):
    """ROS2 node that simulates a robot controller for testing."""

    def __init__(self):
        super().__init__('mock_controller')

        # Declare parameters
        self.declare_parameter('timeout', 30.0)
        self.declare_parameter('required_actions', 10)

        # Get parameter values
        self._timeout = self.get_parameter('timeout').value
        self._required_actions = self.get_parameter('required_actions').value

        self.get_logger().info(
            f'MockControllerNode initialized with timeout={self._timeout}s, '
            f'required_actions={self._required_actions}'
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Subscriber
        self.action_sub = self.create_subscription(
            Float64MultiArray, '/actions', self.action_callback, 10
        )

        # Timer for publishing at 30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.publish_data)

        # Timeout check timer (1Hz)
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)

        # State
        self.valid_actions_received = 0
        self.start_time = self.get_clock().now()

        # Joint names for 14-DOF robot (7 joints per arm for dual-arm setup)
        self.joint_names = [
            'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4',
            'left_joint_5', 'left_joint_6', 'left_joint_7',
            'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4',
            'right_joint_5', 'right_joint_6', 'right_joint_7'
        ]

        # Random number generator for joint positions
        self._rng = np.random.default_rng()

    def publish_data(self):
        """Publish dummy image and joint state data at 30Hz."""
        current_time = self.get_clock().now().to_msg()

        # Publish dummy image (640x480 RGB)
        img_msg = Image()
        img_msg.header.stamp = current_time
        img_msg.header.frame_id = 'camera_link'
        img_msg.height = 480
        img_msg.width = 640
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = 640 * 3  # row length in bytes
        # Create dummy RGB data (random noise for visual variety)
        img_data = self._rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        img_msg.data = img_data.tobytes()
        self.image_pub.publish(img_msg)

        # Publish dummy joint states with random positions
        joint_msg = JointState()
        joint_msg.header.stamp = current_time
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = self.joint_names
        # Generate random joint positions in range [-pi, pi]
        joint_msg.position = (self._rng.random(14) * 2 * np.pi - np.pi).tolist()
        # Add zero velocities and efforts
        joint_msg.velocity = [0.0] * 14
        joint_msg.effort = [0.0] * 14
        self.joint_pub.publish(joint_msg)

    def check_timeout(self):
        """Check if timeout has been exceeded."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed > self._timeout:
            self.get_logger().error(
                f'Timeout after {elapsed:.1f}s. '
                f'Received {self.valid_actions_received}/{self._required_actions} valid actions.'
            )
            raise SystemExit(1)

    def action_callback(self, msg: Float64MultiArray):
        """Handle incoming action commands.

        Args:
            msg: Float64MultiArray containing action values (expected 14 values)
        """
        # Validate action size
        if len(msg.data) != 14:
            self.get_logger().error(
                f'Invalid action size: {len(msg.data)}, expected 14'
            )
            raise SystemExit(1)

        # Validate action values are finite
        for i, val in enumerate(msg.data):
            if not np.isfinite(val):
                self.get_logger().error(
                    f'Invalid action value at index {i}: {val} (must be finite)'
                )
                raise SystemExit(1)

        self.valid_actions_received += 1
        self.get_logger().info(
            f'Valid action received [{self.valid_actions_received}/{self._required_actions}]'
        )

        if self.valid_actions_received >= self._required_actions:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.get_logger().info(
                f'Test PASSED! Received {self._required_actions} valid actions in {elapsed:.1f}s'
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
        node.get_logger().info(f'Node shutting down with exit code {exit_code}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
