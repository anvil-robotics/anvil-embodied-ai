"""MCAP file reader for ROS2 messages"""

from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages


class McapReader:
    """
    Read and parse MCAP files containing ROS2 messages

    Example:
        reader = McapReader("recording.mcap")
        topics = reader.list_topics()

        for message in reader.read_messages(topics=["/camera/image"]):
            process(message)
    """

    def __init__(self, mcap_path: str):
        """
        Initialize MCAP reader

        Args:
            mcap_path: Path to MCAP file

        Raises:
            FileNotFoundError: If MCAP file doesn't exist
        """
        self.mcap_path = Path(mcap_path)

        if not self.mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    def read_messages(self, topics: Optional[List[str]] = None) -> Iterator:
        """
        Read messages from MCAP file

        Args:
            topics: List of topics to read. If None, read all topics.

        Yields:
            Messages from MCAP file with structure:
                - message.channel.topic: Topic name
                - message.ros_msg: ROS message object
                - message.log_time: Message timestamp

        Example:
            for msg in reader.read_messages(topics=["/camera/image"]):
                print(f"Topic: {msg.channel.topic}")
                print(f"Time: {msg.log_time}")
                print(f"Data: {msg.ros_msg}")
        """
        return read_ros2_messages(str(self.mcap_path), topics=topics)

    def list_topics(self) -> Dict[str, Any]:
        """
        List all available topics in MCAP file

        Returns:
            Dictionary mapping topic names to topic info:
            {
                "/camera/image": {
                    "type": "sensor_msgs/Image",
                    "count": 1234,
                },
                ...
            }

        Note:
            This reads through entire file to count messages.
            For large files, this may take time.
        """
        topics_info = {}

        for msg in read_ros2_messages(str(self.mcap_path)):
            topic_name = msg.channel.topic

            if topic_name not in topics_info:
                topics_info[topic_name] = {
                    "type": msg.channel.schema.name,
                    "count": 0,
                }

            topics_info[topic_name]["count"] += 1

        return topics_info

    def get_duration(self) -> float:
        """
        Get recording duration in seconds

        Returns:
            Duration in seconds, or 0.0 if no messages
        """
        timestamps = []

        for msg in read_ros2_messages(str(self.mcap_path)):
            # Extract timestamp from message
            if hasattr(msg.ros_msg, 'header'):
                time_ns = (
                    msg.ros_msg.header.stamp.sec * 1e9 +
                    msg.ros_msg.header.stamp.nanosec
                ) / 1e9
                timestamps.append(time_ns)

        if len(timestamps) < 2:
            return 0.0

        return max(timestamps) - min(timestamps)

    def __repr__(self) -> str:
        return f"McapReader('{self.mcap_path}')"
