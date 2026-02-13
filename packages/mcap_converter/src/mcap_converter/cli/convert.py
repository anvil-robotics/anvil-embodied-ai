"""
MCAP to LeRobot Dataset Converter (Modular Version)

Uses extracted core modules for cleaner, testable code.
"""

import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List

import huggingface_hub

from mcap_converter import (
    ConfigLoader,
    DataConfig,
    LeRobotWriter,
    McapReader,
)
from mcap_converter.core.extractor import BufferedStreamExtractor


def get_timestamp() -> str:
    """Get current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def collect_mcap_files(input_dir: str) -> List[Path]:
    """Recursively collect all MCAP files under input directory"""
    mcap_paths = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mcap"):
                mcap_paths.append(Path(root) / file)
    return sorted(mcap_paths)


def quick_scan_joint_names(mcap_path: str, config: DataConfig) -> dict:
    """
    Quick scan to extract joint names from first JointState message.

    Only reads the first message, so memory-efficient for large files.

    Returns:
        Dictionary mapping robot prefix to joint names:
        - {"right": ["joint1", ...], "left": [...]} for multi-robot
        - {"": ["joint1", ...]} for single robot
        Joint names are extracted from the observation role.
    """
    reader = McapReader(mcap_path)
    joint_pattern = config.joint_name_pattern
    sep = joint_pattern.separator

    for message in reader.read_messages(topics=[config.robot_state_topic]):
        ros_msg = message.ros_msg

        # Group joint names by robot prefix
        robot_joints: dict = {}  # {robot_prefix: [joint_ids]}

        for joint_name in ros_msg.name:
            # Find role prefix
            role = None
            robot = ""
            remaining = ""

            for prefix, role_name in joint_pattern.role_prefix.items():
                if joint_name.startswith(prefix + sep):
                    role = role_name
                    remaining = joint_name[len(prefix) + len(sep) :]
                    break

            if role != "observation":
                continue

            # Extract robot prefix and joint_id
            parts = remaining.split(sep, 1)
            if parts and parts[0] in joint_pattern.robot_prefix:
                robot = joint_pattern.robot_prefix[parts[0]]
                joint_id = parts[1] if len(parts) > 1 else parts[0]
            else:
                robot = ""
                joint_id = remaining

            if robot not in robot_joints:
                robot_joints[robot] = []
            robot_joints[robot].append(joint_id)

        if robot_joints:
            return robot_joints

    return {}


def convert_session(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    robot_type: str = "anvil_openarm",
    fps: int = 30,
    tolerance_s: float = 1e-3,
    task: str = "manipulation",
    config: DataConfig = None,
    buffer_seconds: float = 5.0,
    config_path: str = None,
    vcodec: str = "h264",
):
    """
    Convert MCAP session to LeRobot dataset

    Args:
        input_dir: Directory containing MCAP files
        output_dir: Output directory for dataset
        repo_id: HuggingFace repository ID
        robot_type: Robot type identifier
        fps: Video frames per second
        tolerance_s: Time synchronization tolerance
        task: Task name for the dataset
        config: Data configuration
        buffer_seconds: Buffer window for time alignment in seconds (default: 5.0)
        config_path: Path to the conversion config YAML file (for copying to output)
        vcodec: Video codec for encoding ("h264", "hevc", or "libsvtav1")
    """
    session_start_time = time.time()

    if config is None:
        config = ConfigLoader.get_default()

    # Find all MCAP files
    mcap_files = collect_mcap_files(input_dir)
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {input_dir}")

    print(f"[{get_timestamp()}] Found {len(mcap_files)} MCAP files")
    print(f"[{get_timestamp()}] Buffered streaming (buffer={buffer_seconds}s)")

    # Initialize writer
    writer = LeRobotWriter(
        output_dir=output_dir,
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        config=config,
        vcodec=vcodec,
    )

    # Get joint names
    print(f"[{get_timestamp()}] Quick scan for joint names: {mcap_files[0]}")
    joint_names = quick_scan_joint_names(str(mcap_files[0]), config)
    if not joint_names:
        raise ValueError("Cannot get joint names from reference MCAP (no observation joints found)")

    # Log detected robot mode
    robots = [r for r in joint_names.keys() if r]
    total_joints = sum(len(v) for v in joint_names.values())
    if robots:
        print(f"[{get_timestamp()}] Detected bimanual robot: {robots}")
        for robot in sorted(robots):
            print(f"  {robot}: {joint_names[robot]}")
    else:
        print(f"[{get_timestamp()}] Detected single-arm robot")
        print(f"  joints: {joint_names.get('', [])}")
    print(f"[{get_timestamp()}] Total joints: {total_joints} (observation + action)")

    # Get camera names
    camera_names = list(config.camera_topic_mapping.values())
    if not camera_names:
        raise ValueError("No camera images available, cannot create dataset image features")
    print(f"[{get_timestamp()}] Cameras: {camera_names}")

    # Create dataset
    dataset = writer.create_dataset(
        joint_names=joint_names,
        camera_names=camera_names,
    )

    # Copy conversion config for inference generation during training
    conversion_config_dest = os.path.join(output_dir, "conversion_config.yaml")
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, conversion_config_dest)
        print(f"[{get_timestamp()}] Copied conversion config: {conversion_config_dest}")
    else:
        # Save config from DataConfig object
        import yaml

        with open(conversion_config_dest, "w") as f:
            yaml.dump(
                {
                    "robot_state_topic": config.robot_state_topic,
                    "joint_names": {
                        "separator": config.joint_name_pattern.separator,
                        "source": config.joint_name_pattern.source,
                        "arms": config.joint_name_pattern.arms,
                    },
                    "camera_topic_mapping": config.camera_topic_mapping,
                },
                f,
                default_flow_style=False,
            )
        print(f"[{get_timestamp()}] Saved conversion config: {conversion_config_dest}")

    # Process each MCAP file as one episode
    total_frames = 0
    episode_times = []

    for episode_idx, mcap_path in enumerate(mcap_files):
        episode_start_time = time.time()
        print(f"\n{'=' * 70}")
        print(f"[{get_timestamp()}] Processing episode {episode_idx + 1}/{len(mcap_files)}")
        print(f"  MCAP: {mcap_path}")
        print(f"{'=' * 70}")

        # Use buffered streaming for memory-efficient extraction
        stream_extractor = BufferedStreamExtractor(
            config=config,
            buffer_seconds=buffer_seconds,
            fps=fps,
        )

        frame_count = 0
        for frame in stream_extractor.extract_frames(str(mcap_path), task=task):
            dataset.add_frame(frame)
            frame_count += 1

        print(f"[{get_timestamp()}] Saving episode ({frame_count} frames)...")
        dataset.save_episode()
        dataset.stop_image_writer()

        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        total_frames += frame_count
        print(f"[{get_timestamp()}] Episode completed in {format_duration(episode_time)}")

    # Finalize dataset
    writer.finalize(dataset)

    # Calculate timing statistics
    total_time = time.time() - session_start_time
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0
    fps_actual = total_frames / total_time if total_time > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"[{get_timestamp()}] Successfully created LeRobot dataset!")
    print(f"{'=' * 70}")
    print(f"  Episodes:        {dataset.meta.total_episodes}")
    print(f"  Total frames:    {total_frames}")
    print(f"  Location:        {output_dir}")
    print(f"  Conversion config: {conversion_config_dest}")
    print(f"{'=' * 70}")
    print(f"  Total time:      {format_duration(total_time)}")
    print(f"  Avg per episode: {format_duration(avg_episode_time)}")
    print(f"  Processing rate: {fps_actual:.1f} frames/sec")
    print(f"{'=' * 70}")

    return dataset


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert MCAP recordings to LeRobot v3.0 dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-convert -i data/raw/session -o /tmp/dataset --config configs/mcap_converter/openarm_bimanual.yaml
  mcap-convert -i data/raw/session -o /tmp/dataset --vcodec libsvtav1
  mcap-convert -i data/raw/session -o /tmp/dataset --fps 15 --push-to-hub
""",
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="input directory containing MCAP files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="data/processed/dataset",
        help="output directory (default: data/processed/dataset)",
    )
    parser.add_argument(
        "--config", type=str,
        help="path to YAML config file",
    )
    parser.add_argument(
        "--hf-user", type=str,
        help="Hugging Face username (default: auto-detect)",
    )
    parser.add_argument(
        "--hf-repo", type=str,
        help="dataset repository name (default: output dir name)",
    )
    parser.add_argument(
        "--robot-type", type=str, default="anvil_openarm",
        choices=["anvil_openarm", "anvil_yam"],
        help="robot type (default: anvil_openarm)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="video framerate (default: 30)",
    )
    parser.add_argument(
        "--tolerance-s", type=float, default=1e-3,
        help="timestamp sync tolerance in seconds (default: 0.001)",
    )
    parser.add_argument(
        "--task", type=str, default="manipulation",
        help="task name for the dataset (default: manipulation)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="upload to Hugging Face Hub after conversion",
    )
    parser.add_argument(
        "--buffer-seconds", type=float, default=5.0,
        help="buffer window for time alignment in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--vcodec", type=str, default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="video codec (default: h264). h264 is widely viewable; libsvtav1 gives best compression",
    )

    args = parser.parse_args(args)

    # Normalize paths
    args.output_dir = args.output_dir.rstrip("/")

    # Handle HuggingFace username
    if args.hf_user:
        hf_username = args.hf_user
    else:
        try:
            user_info = huggingface_hub.whoami()
            hf_username = user_info["name"]
        except Exception as e:
            print(f"[WARNING] Cannot get Hugging Face user information: {e}")
            hf_username = "anvil_robot"

    # Construct repo_id
    dataset_name = args.hf_repo if args.hf_repo else Path(args.output_dir).name
    repo_id = f"{hf_username}/{dataset_name}"

    # Load configuration
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
        print(f"[{get_timestamp()}] Loaded config from: {args.config}")
    else:
        config = ConfigLoader.get_default()
        print(f"[{get_timestamp()}] Using default configuration")

    print("=" * 70)
    print(f"[{get_timestamp()}] MCAP to LeRobot Dataset Converter")
    print("=" * 70)
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  HuggingFace Repo: {repo_id}")
    print(f"  Robot Type:       {args.robot_type}")
    print(f"  FPS:              {args.fps}")
    print(f"  Buffer seconds:   {args.buffer_seconds}")
    print(f"  Video codec:      {args.vcodec}")
    print("=" * 70)

    try:
        # Remove output directory if exists
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            print(f"[{get_timestamp()}] Removed existing output directory")

        # Convert session
        print(f"\n[{get_timestamp()}] Starting conversion...")
        dataset = convert_session(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            repo_id=repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            tolerance_s=args.tolerance_s,
            task=args.task,
            config=config,
            buffer_seconds=args.buffer_seconds,
            config_path=args.config,
            vcodec=args.vcodec,
        )

        print(f"\n[{get_timestamp()}] Dataset format: LeRobot v3.0")

        # Upload to Hub if requested
        if args.push_to_hub:
            print(f"\n[{get_timestamp()}] Uploading dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            print(f"[{get_timestamp()}] Dataset uploaded successfully!")

    except Exception as e:
        print(f"\n[{get_timestamp()}] Error occurred during conversion: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
