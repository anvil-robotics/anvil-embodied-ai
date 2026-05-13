"""
MCAP to LeRobot Dataset Converter (Modular Version)

Uses extracted core modules for cleaner, testable code.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import huggingface_hub
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mcap_converter import (
    ActionSource,
    DataConfig,
    LeRobotWriter,
    McapReader,
    load_config,
)
from mcap_converter.core.extractor import BufferedStreamExtractor, parse_joint_name
from mcap_converter.core.reader import snap_fps

console = Console()


def log(message: str) -> None:
    """Print a timestamped log message, left-aligned."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim][{ts}][/dim] {message}")


@contextlib.contextmanager
def suppress_fd_output():
    """Suppress stdout/stderr at the file descriptor level (catches C/ffmpeg output)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


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


def quick_scan_joint_names(mcap_path: Path, config: DataConfig) -> dict[str, list[str]]:
    """Scan the first JointState message; return {arm: [joint_ids]} per configured arm."""
    reader = McapReader(str(mcap_path))
    arms_wanted = set(config.arms)
    for message in reader.read_messages(topics=[config.robot_state_topic]):
        per_arm: dict[str, list[str]] = {arm: [] for arm in arms_wanted}
        for joint_name in message.ros_msg.name:
            parsed = parse_joint_name(joint_name)
            if parsed is None:
                continue
            role, arm, joint_id = parsed
            if role == "observation" and arm in per_arm:
                per_arm[arm].append(joint_id)
        if any(per_arm.values()):
            return {arm: sorted(joints) for arm, joints in per_arm.items() if joints}
    return {}


def convert_session(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    robot_type: str = "anvil_openarm",
    frequency: int = 30,
    tolerance_s: float = 1e-3,
    task: str = "manipulation",
    config: DataConfig = None,
    buffer_seconds: float = 5.0,
    config_path: str = None,
    vcodec: str = "h264",
    resume_from: int = 0,
    max_episodes: int = None,
    mcap_files: List[Path] = None,
    debug_plot_episodes: int = 5,
):
    """
    Convert MCAP session to LeRobot dataset

    Args:
        input_dir: Directory containing MCAP files
        output_dir: Output directory for dataset
        repo_id: HuggingFace repository ID
        robot_type: Robot type identifier
        frequency: Output dataset sample rate in Hz
        tolerance_s: Time synchronization tolerance
        task: Task name for the dataset
        config: Data configuration
        buffer_seconds: Buffer window for time alignment in seconds (default: 5.0)
        config_path: Path to the conversion config YAML file (for copying to output)
        vcodec: Video codec for encoding ("h264", "hevc", or "libsvtav1")
    """
    session_start_time = time.time()

    if config is None:
        raise ValueError("config is required")

    # Stamp the resolved frequency onto config so the extractor reads the same value.
    config.frequency = frequency

    # Find all MCAP files (use pre-collected list if provided)
    if mcap_files is None:
        mcap_files = collect_mcap_files(input_dir)
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {input_dir}")

    if max_episodes is not None:
        mcap_files = mcap_files[:max_episodes]
        log(f"Found [bold]{len(mcap_files)}[/bold] MCAP files (limited to first {max_episodes})")
    else:
        log(f"Found [bold]{len(mcap_files)}[/bold] MCAP files")
    log(f"Buffered streaming (buffer={buffer_seconds}s)")

    # Initialize writer (quiet — Rich handles output)
    writer = LeRobotWriter(
        output_dir=output_dir,
        repo_id=repo_id,
        robot_type=robot_type,
        frequency=frequency,
        config=config,
        vcodec=vcodec,
        quiet=True,
    )

    # Get joint names per arm from the first JointState message.
    log(f"Quick scan for joint names: [dim]{mcap_files[0]}[/dim]")
    joint_names = quick_scan_joint_names(mcap_files[0], config)
    if not joint_names:
        raise ValueError(
            "Cannot get joint names from reference MCAP (no follower_* joints matching "
            f"configured arms={config.arms} found in {config.robot_state_topic})"
        )

    total_joints = sum(len(v) for v in joint_names.values())
    layout = "bimanual" if len(joint_names) > 1 else "single-arm"
    log(
        f"Detected [bold cyan]{layout}[/bold cyan] robot "
        f"([bold magenta]{config.action_source.value}[/bold magenta]): "
        f"{sorted(joint_names.keys())}"
    )
    for arm in sorted(joint_names.keys()):
        log(f"  {arm}: {joint_names[arm]}")
    log(f"Total joints: [bold]{total_joints}[/bold] (observation + action)")

    # Get camera names
    camera_names = list(config.camera_topic_mapping.values())
    if not camera_names:
        raise ValueError("No camera images available, cannot create dataset image features")
    log(f"Cameras: {camera_names}")

    # Create or load dataset
    if resume_from > 0:
        dataset = writer.load_dataset_for_writing()
        log(f"Loaded existing dataset ({resume_from} episodes already converted)")
    else:
        dataset = writer.create_dataset(
            joint_names=joint_names,
            camera_names=camera_names,
        )

    # Snapshot the conversion config alongside the dataset (used by downstream tooling).
    conversion_config_dest = os.path.join(output_dir, "conversion_config.yaml")
    if resume_from > 0:
        log(f"Skipping config copy — using existing [dim]{conversion_config_dest}[/dim]")
    elif config_path and os.path.exists(config_path):
        shutil.copy(config_path, conversion_config_dest)
        log(f"Copied conversion config: [dim]{conversion_config_dest}[/dim]")
    else:
        import yaml
        with open(conversion_config_dest, "w") as f:
            yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
        log(f"Saved conversion config: [dim]{conversion_config_dest}[/dim]")

    # Process each MCAP file as one episode
    total_frames = 0
    episode_times = []
    episode_frame_counts = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.fields[status]}"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[bold blue]Converting episodes",
            total=len(mcap_files),
            status=f"{resume_from}/{len(mcap_files)} episodes",
        )

        for episode_idx, mcap_path in enumerate(mcap_files):
            if episode_idx < resume_from:
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                continue

            episode_start_time = time.time()

            episode_task = progress.add_task(
                f"  [dim]{mcap_path.name}[/dim]",
                total=None,
                status="starting...",
            )

            # Use buffered streaming for memory-efficient extraction (quiet — Rich handles output)
            frame_count = 0

            def on_frame_progress(count, _task=episode_task):
                nonlocal frame_count
                frame_count = count
                elapsed = time.time() - episode_start_time
                speed = count / elapsed if elapsed > 0 else 0
                progress.update(
                    _task,
                    completed=count,
                    status=f"[green]{count}[/green] frames [dim]({speed:.0f} f/s)[/dim]",
                )

            stream_extractor = BufferedStreamExtractor(
                config=config,
                buffer_seconds=buffer_seconds,
                quiet=True,
                progress_callback=on_frame_progress,
            )

            for frame in stream_extractor.extract_frames(mcap_path, task=task):
                dataset.add_frame(frame)

            if frame_count == 0:
                # Skip empty episodes — don't call save_episode on an empty buffer
                progress.update(
                    episode_task,
                    total=1,
                    completed=1,
                    status="[yellow]skipped (0 frames)[/yellow]",
                )
                progress.advance(overall_task)
                progress.update(
                    overall_task,
                    status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
                )
                episode_frame_counts.append(0)
                episode_times.append(time.time() - episode_start_time)
                continue

            # Save episode — suppress ffmpeg/libx264 noise
            progress.update(
                episode_task,
                status=f"[yellow]saving {frame_count} frames...[/yellow]",
            )
            with suppress_fd_output():
                dataset.save_episode()

            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            episode_frame_counts.append(frame_count)
            total_frames += frame_count

            # Mark episode done with green bar
            progress.update(
                episode_task,
                total=frame_count,
                completed=frame_count,
                status=f"[green]{frame_count} frames[/green] in {format_duration(episode_time)}",
            )
            progress.advance(overall_task)
            progress.update(
                overall_task,
                status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
            )

    # Check for all-empty conversion
    if total_frames == 0:
        console.print(
            "\n[bold red]ERROR: All episodes produced 0 frames.[/bold red]\n"
            "The extractor printed diagnostics above (scroll up).\n"
            "Common causes:\n"
            "  1. Camera topics in config don't match MCAP topics\n"
            "  2. Action topics don't exist in MCAP (quest mode)\n"
            "  3. Joint name prefixes don't match config source mapping\n"
            "  Run [bold]mcap-inspect[/bold] on your MCAP to see available topics.\n"
        )
        return dataset

    # Finalize dataset
    with console.status("[bold]Finalizing dataset (metadata & cleanup)..."):
        with suppress_fd_output():
            writer.finalize(dataset)

    # Debug plots: always generated after a successful conversion
    if total_frames > 0:
        from mcap_converter.utils.debug_plot import plot_conversion_debug
        with console.status("[bold]Generating debug plots..."):
            plot_conversion_debug(
                output_dir,
                n_episodes=debug_plot_episodes,
                action_n_step=config.action_n_step,
            )
        log(f"Debug plots saved to [dim]{output_dir}/debug_plots/[/dim]")

    # Calculate timing statistics
    total_time = time.time() - session_start_time
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0
    fps_actual = total_frames / total_time if total_time > 0 else 0

    # Build final report
    # Summary table
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Episodes", str(dataset.meta.total_episodes))
    summary.add_row("Total frames", str(total_frames))
    summary.add_row("Location", output_dir)
    summary.add_row("Conversion config", conversion_config_dest)

    # Per-episode table
    ep_table = Table(title="Per-Episode Breakdown", title_style="bold", title_justify="left", padding=(0, 1))
    ep_table.add_column("#", justify="right", style="dim")
    ep_table.add_column("MCAP File")
    ep_table.add_column("Frames", justify="right")
    ep_table.add_column("Duration", justify="right")
    ep_table.add_column("Speed", justify="right")
    for i, mcap_path in enumerate(mcap_files[resume_from:], start=resume_from):
        j = i - resume_from
        ep_fps = episode_frame_counts[j] / episode_times[j] if episode_times[j] > 0 else 0
        ep_table.add_row(
            str(i + 1),
            mcap_path.name,
            str(episode_frame_counts[j]),
            format_duration(episode_times[j]),
            f"{ep_fps:.1f} f/s",
        )

    # Timing table
    timing = Table(show_header=False, box=None, padding=(0, 2))
    timing.add_column(style="bold")
    timing.add_column()
    timing.add_row("Total time", format_duration(total_time))
    timing.add_row("Avg per episode", format_duration(avg_episode_time))
    timing.add_row("Processing rate", f"{fps_actual:.1f} frames/sec")

    report = Panel(
        Group(summary, "", Padding(ep_table, (0, 0, 0, 2)), "", timing),
        title="[bold green]LeRobot Dataset Created Successfully",
        border_style="green",
        padding=(1, 2),
    )
    console.print(report)

    return dataset


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert MCAP recordings to LeRobot v3.0 dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/openarm_bimanual.yaml
  # output goes to data/datasets/my-session/

  mcap-convert -i data/raw/my-session -o data/datasets --vcodec libsvtav1
  mcap-convert -i data/raw/my-session -o data/datasets --frequency 15 --push-to-hub
  mcap-convert -i data/raw/my-session -o data/datasets --max-episodes 5
  mcap-convert -i data/raw/my-session -o data/datasets --resume
""",
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="input directory containing MCAP files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="data/datasets",
        help="output base directory — dataset is saved to <output-dir>/<input-dir-name>/ (default: data/datasets)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
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
        "--frequency", type=int, default=None,
        help=(
            "output dataset frequency in Hz — overrides `frequency` from the YAML config. "
            "If the source MCAP rate is lower than this target, the converter clamps to "
            "the source rate (no upsampling)."
        ),
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
    parser.add_argument(
        "--resume", action="store_true",
        help="resume conversion — skip already-converted episodes and append new ones",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        metavar="N",
        help="only convert the first N episodes (default: convert all)",
    )
    parser.add_argument(
        "--action-n-step", type=int, default=None,
        metavar="N",
        help="override action_n_step in config (action[t] = observation[t+N]); only valid when "
             "action_source == future_observations",
    )
    parser.add_argument(
        "--debug-plot-episodes", type=int, default=5,
        metavar="N",
        help="number of episodes to include in debug plots (default: 5)",
    )
    args = parser.parse_args(args)

    # Resolve output path: <output-dir>/<input-dir-name>/
    input_name = Path(args.input_dir.rstrip("/")).name
    args.output_dir = str(Path(args.output_dir.rstrip("/")) / input_name)

    # Handle HuggingFace username
    if args.hf_user:
        hf_username = args.hf_user
    else:
        try:
            user_info = huggingface_hub.whoami()
            hf_username = user_info["name"]
        except Exception as e:
            log(f"[yellow]Cannot get Hugging Face user info: {e}[/yellow]")
            hf_username = "anvil_robot"

    # Construct repo_id
    dataset_name = args.hf_repo if args.hf_repo else Path(args.output_dir).name
    repo_id = f"{hf_username}/{dataset_name}"

    config = load_config(args.config)
    log(f"Loaded config from: [dim]{args.config}[/dim]")

    if args.action_n_step is not None:
        if config.action_source is not ActionSource.future_observations:
            console.print(
                "[bold red]ERROR: --action-n-step is only valid when "
                "action_source == future_observations[/bold red]"
            )
            exit(1)
        config.action_n_step = args.action_n_step
        log(f"action_n_step overridden to [bold]{args.action_n_step}[/bold] via --action-n-step")

    # Collect MCAP files once (reused for source rate detection and conversion)
    all_mcap_files = collect_mcap_files(args.input_dir)

    # Auto-detect source rate from MCAP (fast — reads summary only).
    ref_topic = list(config.camera_topic_mapping.keys())[0] if config.camera_topic_mapping else None
    ep_rates_raw = []
    if ref_topic:
        for f in all_mcap_files:
            v = McapReader(str(f)).estimate_fps(ref_topic)
            if v:
                ep_rates_raw.append(v)

    if ep_rates_raw:
        snapped = [snap_fps(v) for v in ep_rates_raw]
        source_frequency = snap_fps(min(ep_rates_raw))
        source_label = str(source_frequency)
        if len(set(snapped)) > 1:
            source_label = f"{source_frequency} [yellow](mixed: {snapped})[/yellow]"
    else:
        source_frequency = None
        source_label = "unknown"

    # Resolve output frequency. Priority: CLI --frequency > config.frequency.
    # Clamp at source rate (can't upsample).
    if args.frequency is not None:
        target_frequency = args.frequency
        target_origin = "CLI override"
    else:
        target_frequency = config.frequency
        target_origin = "from config"

    if source_frequency is None:
        log("[yellow]Cannot detect source rate — using target frequency as-is.[/yellow]")
        frequency = target_frequency
        output_frequency_label = f"{frequency} ({target_origin}; source unknown)"
    elif target_frequency > source_frequency:
        log(
            f"[yellow]Target frequency ({target_frequency} Hz, {target_origin}) exceeds "
            f"source rate ({source_frequency} Hz); clamping to source (no upsampling).[/yellow]"
        )
        frequency = source_frequency
        output_frequency_label = f"{frequency} (clamped to source; {target_origin} was {target_frequency})"
    else:
        frequency = target_frequency
        output_frequency_label = f"{frequency} ({target_origin})"

    # Startup banner
    banner = Table(show_header=False, box=None, padding=(0, 2))
    banner.add_column(style="bold")
    banner.add_column()
    banner.add_row("Input directory", args.input_dir)
    banner.add_row("Output directory", args.output_dir)
    banner.add_row("HuggingFace Repo", repo_id)
    banner.add_row("Robot Type", args.robot_type)
    banner.add_row("Source Session Rate", f"{source_label} Hz")
    banner.add_row("Output Frequency", f"{output_frequency_label} Hz")
    banner.add_row("Buffer", f"{args.buffer_seconds}s")
    banner.add_row("Video codec", args.vcodec)
    banner.add_row("Resume", "yes" if args.resume else "no")
    banner.add_row("Max episodes", str(args.max_episodes) if args.max_episodes else "all")
    banner.add_row("Action source", config.action_source.value)
    banner.add_row("Arms", ", ".join(config.arms))
    if config.action_source is ActionSource.future_observations:
        n_label = str(config.action_n_step)
        if args.action_n_step is not None:
            n_label += " [yellow](CLI override)[/yellow]"
        banner.add_row("action n-step", n_label)
    banner.add_row("Debug plots", f"first {args.debug_plot_episodes} episodes")

    console.print(Panel(
        banner,
        title="[bold]MCAP to LeRobot Dataset Converter",
        border_style="blue",
        padding=(1, 2),
    ))

    try:
        # Determine resume_from: number of already-converted episodes to skip
        resume_from = 0
        if args.resume and os.path.exists(args.output_dir):
            info_path = os.path.join(args.output_dir, "meta", "info.json")
            try:
                with open(info_path) as f:
                    resume_from = json.load(f).get("total_episodes", 0)
                log(f"Resuming from episode [bold]{resume_from}[/bold] — skipping already-converted episodes")
            except Exception as e:
                log(f"[yellow]Cannot read existing metadata ({e}) — starting fresh[/yellow]")
                shutil.rmtree(args.output_dir)
        elif os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            log("Removed existing output directory")

        # Convert session
        log("[bold]Starting conversion...[/bold]")
        dataset = convert_session(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            repo_id=repo_id,
            robot_type=args.robot_type,
            frequency=frequency,
            tolerance_s=args.tolerance_s,
            task=args.task,
            config=config,
            buffer_seconds=args.buffer_seconds,
            config_path=args.config,
            vcodec=args.vcodec,
            resume_from=resume_from,
            max_episodes=args.max_episodes,
            mcap_files=all_mcap_files,
            debug_plot_episodes=args.debug_plot_episodes,
        )

        # Upload to Hub if requested
        if args.push_to_hub:
            with console.status("[bold]Uploading dataset to Hugging Face Hub..."):
                dataset.push_to_hub()
            log("[green]Dataset uploaded successfully![/green]")

    except Exception:
        console.print_exception()
        exit(1)


if __name__ == "__main__":
    main()
