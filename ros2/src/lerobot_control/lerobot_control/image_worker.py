"""
Image Worker Process for Multi-Process Inference

Each image worker runs in a separate process, subscribing to a single camera topic,
decompressing JPEG images, and writing to shared memory. This eliminates GIL
contention with the main inference process.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import time
from typing import Tuple

from .shared_image_buffer import SharedImageBuffer


class ImageWorkerNode(Node):
    """
    ROS2 node that subscribes to a single camera and writes to shared memory.

    Runs in its own process for true parallelism (no GIL).
    """

    def __init__(
        self,
        camera_topic: str,
        camera_name: str,
        image_shape: Tuple[int, int, int],
        buffer_name_prefix: str = "lerobot_img_"
    ):
        super().__init__(f'image_worker_{camera_name}')

        self.camera_name = camera_name
        self.camera_topic = camera_topic
        self.image_shape = image_shape

        # Connect to shared memory (created by main process)
        self.shared_buffer = SharedImageBuffer(
            camera_names=[camera_name],
            image_shape=image_shape,
            create=False,
            buffer_name_prefix=buffer_name_prefix
        )

        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.last_log_time = 0
        self.log_interval = 5.0  # Log every 5 seconds

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            CompressedImage,
            camera_topic,
            self._image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Image worker started: {camera_topic} -> {camera_name}"
        )

    def _image_callback(self, msg: CompressedImage):
        """Process incoming compressed image."""
        if self.start_time is None:
            self.start_time = time.time()

        try:
            # Decompress JPEG (CPU-intensive, but no GIL contention in separate process)
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                self.get_logger().warn(f"Failed to decode image from {self.camera_name}")
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if image.shape[:2] != self.image_shape[:2]:
                image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

            # Get timestamp from message
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # Write to shared memory
            self.shared_buffer.write(self.camera_name, image, timestamp)

            self.frame_count += 1

            # Periodic logging
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                elapsed = current_time - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.get_logger().info(
                    f"[{self.camera_name}] Processed {self.frame_count} frames, "
                    f"FPS: {fps:.1f}"
                )
                self.last_log_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def destroy_node(self):
        """Cleanup."""
        self.shared_buffer.close()
        super().destroy_node()


def run_image_worker(
    camera_topic: str,
    camera_name: str,
    image_shape: Tuple[int, int, int],
    buffer_name_prefix: str = "lerobot_img_",
    stop_event=None
):
    """
    Entry point for running image worker in a separate process.

    Args:
        camera_topic: ROS2 topic to subscribe to
        camera_name: Name of the camera (e.g., 'waist')
        image_shape: Shape of images (H, W, C)
        buffer_name_prefix: Prefix for shared memory names
        stop_event: Optional multiprocessing.Event to signal shutdown
    """
    rclpy.init()

    node = ImageWorkerNode(
        camera_topic=camera_topic,
        camera_name=camera_name,
        image_shape=image_shape,
        buffer_name_prefix=buffer_name_prefix
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            # Spin with stop check
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            # Spin forever
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


class JointStateWorkerNode(Node):
    """
    ROS2 node that subscribes to joint states and writes to shared memory.

    Joint state processing is lightweight, but we run it in a worker for consistency.
    """

    def __init__(
        self,
        joint_topic: str,
        joint_names: list,
        buffer_name: str = "lerobot_joint_state"
    ):
        super().__init__('joint_state_worker')

        from sensor_msgs.msg import JointState
        from .shared_image_buffer import SharedJointStateBuffer

        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # Connect to shared memory
        self.shared_buffer = SharedJointStateBuffer(
            num_joints=self.num_joints,
            create=False,
            buffer_name=buffer_name
        )

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            joint_topic,
            self._joint_callback,
            10
        )

        self.frame_count = 0
        self.start_time = None

        self.get_logger().info(f"Joint state worker started: {joint_topic}")

    def _joint_callback(self, msg):
        """Process incoming joint state."""
        if self.start_time is None:
            self.start_time = time.time()

        try:
            # Extract positions in order
            positions = np.zeros(self.num_joints, dtype=np.float64)
            msg_names = list(msg.name)
            msg_positions = list(msg.position)

            for i, name in enumerate(self.joint_names):
                if name in msg_names:
                    idx = msg_names.index(name)
                    positions[i] = msg_positions[idx]

            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.shared_buffer.write(positions, timestamp)
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")

    def destroy_node(self):
        self.shared_buffer.close()
        super().destroy_node()


def run_joint_state_worker(
    joint_topic: str,
    joint_names: list,
    buffer_name: str = "lerobot_joint_state",
    stop_event=None
):
    """Entry point for joint state worker process."""
    rclpy.init()

    node = JointStateWorkerNode(
        joint_topic=joint_topic,
        joint_names=joint_names,
        buffer_name=buffer_name
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
