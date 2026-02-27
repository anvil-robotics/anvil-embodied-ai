"""
Custom ROS2 image message converter / 自定义 ROS2 图像消息转换器
Alternative to cv_bridge, avoids NumPy version conflicts / cv_bridge 的替代方案，避免 NumPy 版本冲突
"""

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


class ImageConverter:
    """Convert ROS2 sensor_msgs/Image to NumPy array / 将 ROS2 sensor_msgs/Image 转换为 NumPy 数组"""

    def __init__(self):
        self.encoding_to_dtype = {
            "mono8": np.uint8,
            "mono16": np.uint16,
            "bgr8": np.uint8,
            "rgb8": np.uint8,
            "bgra8": np.uint8,
            "rgba8": np.uint8,
            "8UC1": np.uint8,
            "8UC3": np.uint8,
            "16UC1": np.uint16,
            "32FC1": np.float32,
        }

        self.encoding_to_channels = {
            "mono8": 1,
            "mono16": 1,
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "8UC1": 1,
            "8UC3": 3,
            "16UC1": 1,
            "32FC1": 1,
        }

    def imgmsg_to_numpy(self, img_msg: Image, desired_encoding: str = "rgb8") -> np.ndarray:
        """
        Convert ROS2 Image message to NumPy array / 将 ROS2 图像消息转换为 NumPy 数组

        Args:
            img_msg: ROS2 sensor_msgs/Image message / ROS2 sensor_msgs/Image 消息
            desired_encoding: Target encoding format (default 'rgb8') / 目标编码格式（默认 'rgb8'）

        Returns:
            NumPy array with shape (height, width, channels) or (height, width) /
            形状为 (height, width, channels) 或 (height, width) 的 NumPy 数组
        """
        dtype = self.encoding_to_dtype.get(img_msg.encoding, np.uint8)
        channels = self.encoding_to_channels.get(img_msg.encoding, 3)

        # Convert byte data to NumPy array / 将字节数据转换为 NumPy 数组
        img_array = np.frombuffer(img_msg.data, dtype=dtype)

        # Reshape to image shape / 重新调整为图像形状
        if channels == 1:
            img_array = img_array.reshape((img_msg.height, img_msg.width))
        else:
            img_array = img_array.reshape((img_msg.height, img_msg.width, channels))

        # Handle color space conversion / 处理颜色空间转换
        if img_msg.encoding == "bgr8" and desired_encoding == "rgb8":
            # BGR -> RGB / BGR 转 RGB
            img_array = img_array[..., ::-1]
        elif img_msg.encoding == "bgra8" and desired_encoding == "rgb8":
            # BGRA -> RGB / BGRA 转 RGB
            img_array = img_array[..., [2, 1, 0]]
        elif img_msg.encoding == "rgba8" and desired_encoding == "rgb8":
            # RGBA -> RGB / RGBA 转 RGB
            img_array = img_array[..., :3]

        return img_array

    def numpy_to_imgmsg(self, img_array: np.ndarray, encoding: str = "rgb8") -> Image:
        """
        Convert NumPy array to ROS2 Image message / 将 NumPy 数组转换为 ROS2 图像消息

        Args:
            img_array: NumPy array / NumPy 数组
            encoding: Encoding format (default 'rgb8') / 编码格式（默认 'rgb8'）

        Returns:
            ROS2 sensor_msgs/Image message / ROS2 sensor_msgs/Image 消息
        """
        img_msg = Image()
        img_msg.encoding = encoding

        if len(img_array.shape) == 2:
            # Grayscale image / 灰度图像
            img_msg.height, img_msg.width = img_array.shape
            img_msg.step = img_msg.width
        else:
            # Color image / 彩色图像
            img_msg.height, img_msg.width, channels = img_array.shape
            img_msg.step = img_msg.width * channels

        img_msg.is_bigendian = 0
        img_msg.data = img_array.tobytes()

        return img_msg

    def compressed_imgmsg_to_numpy(
        self, img_msg: CompressedImage, desired_encoding: str = "rgb8"
    ) -> np.ndarray:
        """
        Convert ROS2 CompressedImage message to NumPy array.

        Args:
            img_msg: ROS2 sensor_msgs/CompressedImage message
            desired_encoding: Target encoding format (default 'rgb8')

        Returns:
            NumPy array with shape (height, width, channels)
        """
        # Decompress image data
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_array is None:
            raise ValueError(f"Failed to decode compressed image: {img_msg.format}")

        # cv2.imdecode returns BGR, convert to RGB if needed
        if desired_encoding == "rgb8":
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        return img_array
