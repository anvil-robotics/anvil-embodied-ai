#!/usr/bin/env python3
"""
Video Preview Converter

Converts AV1-encoded videos from LeRobot datasets to H.264 for easy viewing.
LeRobot uses AV1 (best compression for ML training) but some players can't decode it.
This script creates H.264 copies for preview without modifying the original dataset.

Requirements:
    ffmpeg (system package)

Usage:
    # Convert all videos in a dataset to H.264
    preview-videos -i ../dataset/processed/my_dataset

    # Specify output directory
    preview-videos -i ../dataset/processed/my_dataset -o ../dataset/preview

    # Convert single video file
    preview-videos -i video.mp4 -o preview_video.mp4

    # Use MJPEG for maximum compatibility (larger files)
    preview-videos -i ../dataset/processed/my_dataset --codec mjpeg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_video_info(video_path: Path) -> dict:
    """Get video info using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,duration,nb_frames",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(",")
        if len(parts) >= 4:
            return {
                "codec": parts[0],
                "width": int(parts[1]) if parts[1] else 0,
                "height": int(parts[2]) if parts[2] else 0,
                "duration": float(parts[3]) if parts[3] and parts[3] != "N/A" else 0,
                "frames": int(parts[4]) if len(parts) > 4 and parts[4] else 0,
            }
    except (subprocess.CalledProcessError, ValueError):
        pass
    return {}


def convert_video(
    input_path: Path,
    output_path: Path,
    codec: str = "h264",
    crf: int = 23,
    quiet: bool = False,
) -> bool:
    """
    Convert a video file to the specified codec.

    Args:
        input_path: Source video file
        output_path: Destination video file
        codec: Target codec (h264, mjpeg)
        crf: Quality (lower = better, 0-51 for h264)
        quiet: Suppress ffmpeg output

    Returns:
        True if conversion succeeded
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if codec == "h264":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    elif codec == "mjpeg":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "mjpeg",
            "-q:v",
            "3",  # Quality 2-31, lower is better
            str(output_path),
        ]
    else:
        print(f"  Error: Unknown codec '{codec}'")
        return False

    try:
        if quiet:
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error converting {input_path.name}: {e}")
        if e.stderr:
            print(f"  {e.stderr.decode()[:200]}")
        return False


def find_videos(input_path: Path) -> list[Path]:
    """Find all MP4 video files in a directory."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".mp4" else []
    return sorted(input_path.glob("**/*.mp4"))


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024
    return f"{size_float:.1f} TB"


def main(args: list[str] | None = None) -> None:
    """Main entry point for preview-videos CLI."""
    parser = argparse.ArgumentParser(
        description="Convert AV1 videos to H.264/MJPEG for easy viewing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert dataset videos
  preview-videos -i ../dataset/processed/my_dataset

  # Convert to specific output directory
  preview-videos -i ../dataset/processed/my_dataset -o ./preview

  # Single file conversion
  preview-videos -i video.mp4 -o preview.mp4

  # MJPEG for maximum compatibility
  preview-videos -i ../dataset/processed/my_dataset --codec mjpeg
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input video file or dataset directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file or directory (default: {input}_preview)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="h264",
        choices=["h264", "mjpeg"],
        help="Output codec: h264 (universal) or mjpeg (maximum compatibility)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality for H.264 (0-51, lower=better, default: 23)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parsed_args = parser.parse_args(args)

    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg not found. Please install ffmpeg.")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)

    input_path = Path(parsed_args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if parsed_args.output:
        output_path = Path(parsed_args.output)
    else:
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_preview.mp4"
        else:
            output_path = input_path.parent / f"{input_path.name}_preview"

    # Find videos
    videos = find_videos(input_path)
    if not videos:
        print(f"No MP4 files found in {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("Video Preview Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Codec:  {parsed_args.codec}")
    print(f"Videos: {len(videos)} file(s)")
    print("=" * 60)

    # Convert videos
    success_count = 0
    total_input_size = 0
    total_output_size = 0

    for i, video in enumerate(videos, 1):
        # Determine output file path
        if input_path.is_file():
            out_file = output_path
        else:
            rel_path = video.relative_to(input_path)
            out_file = output_path / rel_path

        # Get input info
        info = get_video_info(video)
        codec_str = info.get("codec", "unknown")
        input_size = video.stat().st_size
        total_input_size += input_size

        print(f"\n[{i}/{len(videos)}] {video.name}")
        print(f"  Source: {codec_str}, {format_size(input_size)}")

        # Convert
        if convert_video(video, out_file, parsed_args.codec, parsed_args.crf, parsed_args.quiet):
            output_size = out_file.stat().st_size
            total_output_size += output_size
            ratio = output_size / input_size if input_size > 0 else 0
            print(f"  Output: {parsed_args.codec}, {format_size(output_size)} ({ratio:.1f}x)")
            success_count += 1
        else:
            print("  FAILED")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Converted: {success_count}/{len(videos)} videos")
    print(f"Input size:  {format_size(total_input_size)}")
    print(f"Output size: {format_size(total_output_size)}")
    if total_input_size > 0:
        print(f"Size ratio:  {total_output_size / total_input_size:.2f}x")
    print(f"\nPreview videos saved to: {output_path}")


if __name__ == "__main__":
    main()
