#!/usr/bin/env python3
"""
MCAP to MP4 Direct Converter

Streams video frames directly from MCAP to MP4 without intermediate disk writes.
Auto-detects all image topics and creates one MP4 file per camera.

Memory-efficient: processes one frame at a time, never accumulates in RAM.

Requirements:
    pip install mcap mcap-ros2-support numpy opencv-python tqdm

Usage:
    # Auto-detect and convert all image topics
    mcap2mp4 -i recording.mcap -o ./videos

    # Scan topics only (no conversion)
    mcap2mp4 -i recording.mcap --scan-only

    # Convert specific topics
    mcap2mp4 -i recording.mcap -o ./videos --topics /cam_waist/image_raw/compressed

    # With options
    mcap2mp4 -i recording.mcap -o ./videos --fps 30 --codec libx264 --crf 20 --resize 640x480
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass


def scan_image_topics(mcap_path: str) -> dict[str, dict]:
    """
    Scan MCAP file to find all image topics.

    Args:
        mcap_path: Path to MCAP file

    Returns:
        Dict mapping topic name to info:
        {
            "/camera/image_raw": {"type": "sensor_msgs/msg/Image", "count": 1000},
            "/camera/image_raw/compressed": {"type": "sensor_msgs/msg/CompressedImage", "count": 1000},
        }
    """
    from mcap.reader import make_reader

    image_topics: dict[str, dict] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        # Get channel info
        for _schema_id, _schema in reader.get_summary().schemas.items():
            pass  # Just need to iterate

        for channel_id, channel in reader.get_summary().channels.items():
            schema = reader.get_summary().schemas.get(channel.schema_id)
            if schema:
                msg_type = schema.name
                # Check if it's an image type
                if msg_type in ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"]:
                    image_topics[channel.topic] = {
                        "type": msg_type,
                        "schema_id": channel.schema_id,
                        "channel_id": channel_id,
                    }

        # Count messages per topic
        for topic in image_topics:
            image_topics[topic]["count"] = 0

        for _schema, channel, _message in reader.iter_messages(topics=list(image_topics.keys())):
            image_topics[channel.topic]["count"] += 1

    return image_topics


def topic_to_camera_name(topic: str) -> str:
    """
    Convert topic name to a clean camera name for the output file.

    Examples:
        "/usb_cam_waist/image_raw/compressed" -> "usb_cam_waist"
        "/camera/image_raw" -> "camera"
        "/cam_waist/image_raw/compressed" -> "waist"
        "/wrist_r/image" -> "wrist_r"
    """
    # Remove leading slash
    name = topic.lstrip("/")

    # Remove common suffixes
    suffixes_to_remove = [
        "/image_raw/compressed",
        "/image/compressed",
        "/compressed",
        "/image_raw",
        "/image",
    ]

    for suffix in suffixes_to_remove:
        suffix_clean = suffix.lstrip("/")
        if name.endswith(suffix_clean):
            name = name[: -len(suffix_clean)]
            break

    # Replace remaining slashes with underscores
    name = name.replace("/", "_")

    # Remove trailing underscores
    name = name.rstrip("_")

    # Handle empty name
    if not name:
        name = "camera"

    return name


class FFmpegWriter:
    """Write video frames directly to MP4 using ffmpeg subprocess."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: int = 30,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
    ):
        """
        Initialize FFmpeg writer.

        Args:
            output_path: Output MP4 file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (libx264, libx265, libaom-av1, etc.)
            crf: Constant rate factor (lower = better quality, 0-51 for x264)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        # FFmpeg command for piping raw frames
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",  # Read from stdin
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",  # Compatibility
            output_path,
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: numpy array of shape (H, W, 3) in RGB format
        """
        if self.process.stdin:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1

    def close(self) -> int:
        """Close the writer and finalize the video."""
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()

        if self.process.returncode != 0:
            stderr = self.process.stderr.read().decode() if self.process.stderr else ""
            if stderr:
                print(f"FFmpeg warning/error: {stderr[:500]}")

        return self.frame_count


def decode_ros_image(data: bytes, encoding: str, height: int, width: int) -> np.ndarray:
    """Decode ROS Image message data to numpy array (RGB format)."""
    if encoding in ["rgb8", "RGB8"]:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding in ["bgr8", "BGR8"]:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img = img[:, :, ::-1].copy()  # BGR -> RGB
    elif encoding in ["mono8", "MONO8"]:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        img = np.stack([img, img, img], axis=-1)  # Grayscale -> RGB
    elif encoding in ["rgba8", "RGBA8"]:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        img = img[:, :, :3].copy()  # Drop alpha
    elif encoding in ["bgra8", "BGRA8"]:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        img = img[:, :, 2::-1].copy()  # BGRA -> RGB
    else:
        # Fallback: assume RGB
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

    return img


def decode_compressed_ros_image(data: bytes, format_str: str) -> np.ndarray:
    """Decode ROS CompressedImage message data to numpy array (RGB format)."""
    # Decode compressed image (JPEG, PNG, etc.)
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode compressed image with format: {format_str}")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resize_frame(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image to target size (width, height)."""
    target_w, target_h = target_size
    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return img


def convert_mcap_to_mp4(
    mcap_path: str,
    output_dir: str,
    fps: int = 30,
    codec: str = "libx264",
    crf: int = 23,
    resize: tuple[int, int] | None = None,
    topics: list[str] | None = None,
) -> dict[str, int]:
    """
    Convert MCAP file to MP4 videos (one per camera).

    Memory-efficient: streams one frame at a time.

    Args:
        mcap_path: Path to MCAP file
        output_dir: Output directory for MP4 files
        fps: Output video FPS
        codec: Video codec
        crf: Constant rate factor
        resize: Optional (width, height) to resize frames
        topics: Optional list of topics to convert (auto-detect if None)

    Returns:
        Dict mapping camera name to frame count
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mcap_name = Path(mcap_path).stem

    # Scan for image topics if not specified
    print(f"\nScanning MCAP: {mcap_path}")
    image_topics = scan_image_topics(mcap_path)

    if not image_topics:
        print("  No image topics found!")
        return {}

    print(f"  Found {len(image_topics)} image topic(s):")
    for topic, info in image_topics.items():
        print(f"    - {topic}: {info['type']} ({info['count']} frames)")

    # Filter topics if specified
    if topics:
        image_topics = {t: v for t, v in image_topics.items() if t in topics}
        if not image_topics:
            print("  None of the specified topics found!")
            return {}

    # Video writers (created on first frame to get dimensions)
    writers: dict[str, FFmpegWriter] = {}
    frame_counts: dict[str, int] = {}
    topic_to_cam: dict[str, str] = {}

    # Map topics to camera names
    for topic in image_topics:
        cam_name = topic_to_camera_name(topic)
        topic_to_cam[topic] = cam_name
        frame_counts[cam_name] = 0

    # Calculate total frames for progress bar
    total_expected = sum(info["count"] for info in image_topics.values())

    print("\nConverting to MP4...")
    print(f"  Output: {output_dir}")
    print(f"  Total frames: {total_expected}")

    # Stream through MCAP with progress bar
    decoder = DecoderFactory()

    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_expected, desc="Converting", unit="frames")
    except ImportError:
        pbar = None
        print("  (Install tqdm for progress bar: pip install tqdm)")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[decoder])

        for _schema, channel, _message, ros_msg in reader.iter_decoded_messages(
            topics=list(image_topics.keys())
        ):
            topic = channel.topic
            cam_name = topic_to_cam[topic]
            msg_type = image_topics[topic]["type"]

            # Decode image based on type
            if msg_type == "sensor_msgs/msg/Image":
                img = decode_ros_image(
                    bytes(ros_msg.data),
                    ros_msg.encoding,
                    ros_msg.height,
                    ros_msg.width,
                )
            else:  # CompressedImage
                img = decode_compressed_ros_image(
                    bytes(ros_msg.data),
                    ros_msg.format,
                )

            # Resize if requested
            if resize:
                img = resize_frame(img, resize)

            height, width = img.shape[:2]

            # Create writer on first frame
            if cam_name not in writers:
                output_file = str(output_path / f"{mcap_name}_{cam_name}.mp4")
                writers[cam_name] = FFmpegWriter(
                    output_path=output_file,
                    width=width,
                    height=height,
                    fps=fps,
                    codec=codec,
                    crf=crf,
                )
                if pbar:
                    from tqdm import tqdm as tqdm_module

                    tqdm_module.write(f"  Creating: {mcap_name}_{cam_name}.mp4 ({width}x{height})")
                else:
                    print(f"  Creating: {mcap_name}_{cam_name}.mp4 ({width}x{height})")

            # Write frame
            writers[cam_name].write_frame(img)
            frame_counts[cam_name] += 1

            # Update progress
            if pbar:
                pbar.update(1)
            else:
                total_frames = sum(frame_counts.values())
                if total_frames % 1000 == 0:
                    print(f"  Processed {total_frames}/{total_expected} frames...")

    if pbar:
        pbar.close()

    # Close all writers
    print("\nFinalizing videos...")
    for cam_name, writer in writers.items():
        count = writer.close()
        duration = count / fps
        print(f"  {cam_name}: {count} frames ({duration:.1f}s)")

    return frame_counts


def main(args: list[str] | None = None) -> None:
    """Main entry point for mcap2mp4 CLI."""
    parser = argparse.ArgumentParser(
        description="Extract MCAP image topics directly to MP4 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-to-video -i recording.mcap -o ./videos
  mcap-to-video -i recording.mcap --scan-only
  mcap-to-video -i recording.mcap -o ./videos --topics /cam_waist/image_raw/compressed
  mcap-to-video -i recording.mcap -o ./videos --fps 30 --codec libx264 --resize 640x480
""",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="input MCAP file or directory containing MCAP files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./videos",
        help="output directory for MP4 files (default: ./videos)",
    )
    parser.add_argument(
        "--topics", type=str, nargs="+",
        help="specific topics to convert (default: auto-detect all image topics)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="output video FPS (default: 30)",
    )
    parser.add_argument(
        "--codec", type=str, default="libx264",
        choices=["libx264", "libx265", "libaom-av1"],
        help="video codec (default: libx264)",
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="constant rate factor, lower is better quality (default: 23)",
    )
    parser.add_argument(
        "--resize", type=str,
        help="resize frames to WxH, e.g. 640x480",
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="only scan and list image topics, don't convert",
    )

    parsed_args = parser.parse_args(args)

    # Parse resize
    resize = None
    if parsed_args.resize:
        w, h = parsed_args.resize.lower().split("x")
        resize = (int(w), int(h))

    # Find MCAP files
    input_path = Path(parsed_args.input)
    if input_path.is_file():
        mcap_files = [input_path]
    else:
        mcap_files = sorted(input_path.glob("**/*.mcap"))

    if not mcap_files:
        print(f"No MCAP files found in {parsed_args.input}")
        sys.exit(1)

    print("=" * 60)
    print("MCAP to MP4 Direct Converter")
    print("=" * 60)
    print(f"Input: {parsed_args.input}")
    print(f"Files: {len(mcap_files)} MCAP file(s)")
    if not parsed_args.scan_only:
        print(f"Output: {parsed_args.output_dir}")
        print(f"FPS: {parsed_args.fps}")
        print(f"Codec: {parsed_args.codec}")
        print(f"CRF: {parsed_args.crf}")
        if resize:
            print(f"Resize: {resize[0]}x{resize[1]}")
        if parsed_args.topics:
            print(f"Topics: {parsed_args.topics}")
        else:
            print("Topics: auto-detect")
    print("=" * 60)

    # Process each MCAP file
    for mcap_path in mcap_files:
        if parsed_args.scan_only:
            # Just scan and print topics
            print(f"\nScanning: {mcap_path}")
            image_topics = scan_image_topics(str(mcap_path))
            if image_topics:
                print(f"  Found {len(image_topics)} image topic(s):")
                for topic, info in image_topics.items():
                    cam_name = topic_to_camera_name(topic)
                    print(f"    {topic}")
                    print(f"      Type: {info['type']}")
                    print(f"      Frames: {info['count']}")
                    print(f"      Output name: {cam_name}")
            else:
                print("  No image topics found")
        else:
            # Convert
            print(f"\n{'=' * 60}")
            print(f"Processing: {mcap_path}")
            print("=" * 60)

            try:
                convert_mcap_to_mp4(
                    mcap_path=str(mcap_path),
                    output_dir=parsed_args.output_dir,
                    fps=parsed_args.fps,
                    codec=parsed_args.codec,
                    crf=parsed_args.crf,
                    resize=resize,
                    topics=parsed_args.topics,
                )
            except Exception as e:
                print(f"Error processing {mcap_path}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
