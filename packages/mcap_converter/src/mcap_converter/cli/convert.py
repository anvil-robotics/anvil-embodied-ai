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

from anvil_shared.provenance import git_provenance
from mcap_converter import (
    ConfigLoader,
    DataConfig,
    LeRobotWriter,
    McapReader,
)
from mcap_converter.config.schema import ConfigurationError
from mcap_converter.core.extractor import BufferedStreamExtractor
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


def quick_scan_joint_names(mcap_path: str, config: DataConfig) -> dict:
    """
    Quick scan to extract joint names from first JointState message (joint mode only).

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
    # "Quest mode" historically meant: actions come from separate command topics
    # rather than `leader_*` joints in /joint_states. Under the unified config
    # that's true whenever action_command_topics is non-empty.
    quest_mode = bool(config.action_command_topics)

    for message in reader.read_messages(topics=[config.robot_state_topic]):
        ros_msg = message.ros_msg

        # Group joint names by robot prefix
        robot_joints: dict = {}  # {robot_prefix: [joint_ids]}

        for joint_name in ros_msg.name:
            if quest_mode:
                # Quest teleop mode: all joints are observations (no leader prefix).
                # Parse arm identifier and joint_id directly.
                # Joint names are like "follower_l_joint1" — still use source
                # prefix to strip it, then extract arm and joint_id.
                remaining = joint_name
                robot = ""

                # Try to strip known source prefixes
                for prefix in joint_pattern.role_prefix.keys():
                    if joint_name.startswith(prefix + sep):
                        remaining = joint_name[len(prefix) + len(sep) :]
                        break

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
            else:
                # Leader-follower mode: only extract observation (follower) joints
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
            # Sort each arm's joint list for canonical ordering
            for robot in robot_joints:
                robot_joints[robot] = sorted(robot_joints[robot])
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
    resume_from: int = 0,
    max_episodes: int = None,
    mcap_files: List[Path] = None,
    debug_plot_episodes: int = 5,
    cli_act_from_obs: bool = False,
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
        config_path: Path to the conversion config YAML file that produced ``config``
            (kept for logging/reference; conversion_config.yaml is always (re-)serialized
            from ``config`` itself, not copied from this path — see the save-config note
            below)
        vcodec: Video codec for encoding ("h264", "hevc", or "libsvtav1")
        cli_act_from_obs: Force ``action[t] = observation.state[t]`` even when
            ``action_topics`` are configured (joint mode only; EE mode is
            always act-from-obs).
    """
    session_start_time = time.time()

    if config is None:
        config = ConfigLoader.get_default()

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
        fps=fps,
        config=config,
        vcodec=vcodec,
        quiet=True,
    )

    # Get joint names (joint mode only; EE mode has fixed per-arm dims)
    if config.is_ee:
        joint_names: dict = {}
        log(
            f"Detected [bold magenta]EE Cartesian[/bold magenta] mode "
            f"({len(config.arms)} arm(s)): {config.arms}"
        )
        for arm_id, topic in config.observation_topics.items():
            log(f"  {arm_id}: [dim]{topic}[/dim]")
    else:
        log(f"Quick scan for joint names: [dim]{mcap_files[0]}[/dim]")
        joint_names = quick_scan_joint_names(str(mcap_files[0]), config)
        if not joint_names:
            raise ValueError(
                "Cannot get joint names from reference MCAP (no observation joints found)"
            )

        robots = [r for r in joint_names.keys() if r]
        total_joints = sum(len(v) for v in joint_names.values())
        quest_mode = bool(config.action_command_topics)
        teleop_label = (
            "[bold magenta]quest teleop[/bold magenta]" if quest_mode
            else "[bold cyan]act-from-obs / leader-follower[/bold cyan]"
        )
        if robots:
            log(f"Detected [bold cyan]bimanual[/bold cyan] robot ({teleop_label}): {robots}")
            for robot in sorted(robots):
                log(f"  {robot}: {joint_names[robot]}")
        else:
            log(f"Detected [bold cyan]single-arm[/bold cyan] robot ({teleop_label})")
            log(f"  joints: {joint_names.get('', [])}")
        log(f"Total joints: [bold]{total_joints}[/bold] (observation + action)")
        if quest_mode:
            for topic, topic_cfg in config.action_command_topics.items():
                log(f"  Action topic ({topic_cfg.arm}): [dim]{topic}[/dim]")

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

    # Save conversion config for inference generation during training (skip if resuming).
    # Always (re-)serialized from the DataConfig object at the current schema version —
    # not copied verbatim from --config, even when one was given — so conversion_config.yaml
    # always records exactly what was actually used, in canonical current-schema form
    # (a raw copy of a legacy-shaped input config would immediately need its own
    # migration-on-read, which is avoidable for anything converted from this point on).
    conversion_config_dest = os.path.join(output_dir, "conversion_config.yaml")
    if resume_from > 0:
        log(f"Skipping config copy — using existing [dim]{conversion_config_dest}[/dim]")
    else:
        ConfigLoader.to_yaml(config, conversion_config_dest)
        log(f"Saved conversion config: [dim]{conversion_config_dest}[/dim]")

    # Append git provenance to conversion_config.yaml (skip when resuming — already present)
    if resume_from == 0:
        provenance = git_provenance()
        if provenance:
            import yaml
            with open(conversion_config_dest, "a") as _f:
                _f.write("\n# --- provenance ---\n")
                yaml.dump(provenance, _f, default_flow_style=False)

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
                console.print(f"  [dim]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}  skipped (already converted)[/dim]")
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
                fps=fps,
                quiet=True,
                progress_callback=on_frame_progress,
                cli_act_from_obs=cli_act_from_obs,
            )

            corrupt_frame_error: Exception | None = None
            try:
                for frame in stream_extractor.extract_frames(str(mcap_path), task=task):
                    dataset.add_frame(frame)
            except ValueError as exc:
                corrupt_frame_error = exc

            if corrupt_frame_error is not None:
                # Discard any partially-buffered frames for this episode
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                progress.update(
                    episode_task,
                    total=1,
                    completed=1,
                    status=f"[red]skipped (corrupt frame: {corrupt_frame_error})[/red]",
                )
                progress.advance(overall_task)
                progress.update(
                    overall_task,
                    status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
                )
                progress.remove_task(episode_task)
                console.print(
                    f"  [yellow]⚠[/yellow] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  [yellow]skipped (corrupt frame)[/yellow]"
                )
                episode_frame_counts.append(0)
                episode_times.append(time.time() - episode_start_time)
                log(
                    f"[yellow]⚠ Skipped episode {mcap_path.name} — corrupt frame: "
                    f"{corrupt_frame_error}[/yellow]"
                )
                continue

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
                progress.remove_task(episode_task)
                console.print(
                    f"  [yellow]⚠[/yellow] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  [yellow]skipped (0 frames)[/yellow]"
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
            progress.remove_task(episode_task)
            ep_fps = frame_count / episode_time if episode_time > 0 else 0
            console.print(
                f"  [green]✓[/green] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                f"  [green]{frame_count} frames[/green]"
                f"  {format_duration(episode_time)}"
                f"  {ep_fps:.0f} f/s"
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

    # Debug plots: joint mode only. The current plot reads observation.state /
    # action assuming joint positions; EE plots are out of scope for Task 1.
    if total_frames > 0 and not config.is_ee:
        from mcap_converter.utils.debug_plot import plot_conversion_debug
        with console.status("[bold]Generating debug plots..."):
            # action_from_observation_n is read from conversion_config.yaml if
            # present; new configs no longer set it (act-from-obs implies n=0).
            plot_conversion_debug(
                output_dir,
                n_episodes=debug_plot_episodes,
                action_from_observation_n=0 if cli_act_from_obs else None,
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
  # EE Cartesian bimanual (recommended for EE-space diffusion policy)
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/v1.1/openarm_ee_bimanual.yaml
  # output goes to data/datasets/my-session/

  # EE Cartesian left arm only
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/v1.1/openarm_ee_left.yaml

  # Joint space bimanual (Quest teleop, action from command topics)
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/v1.1/openarm_joint_bimanual.yaml

  # Joint space with action[t] = obs[t] (future window via delta_timestamps at train time)
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/v1.1/openarm_joint_bimanual.yaml --act-from-obs

  # Common options
  mcap-convert -i data/raw/my-session -o data/datasets --config ... --vcodec libsvtav1
  mcap-convert -i data/raw/my-session -o data/datasets --config ... --fps 15 --push-to-hub
  mcap-convert -i data/raw/my-session -o data/datasets --config ... --max-episodes 5
  mcap-convert -i data/raw/my-session -o data/datasets --config ... --resume
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
        "--output-path", type=str, default=None,
        help="full output path override — use this exact directory instead of <output-dir>/<input-dir-name>/",
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
        "--fps", type=int, default=None,
        help="output fps — overrides auto-detected source fps; must not exceed source fps",
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
        "--act-from-obs", action="store_true",
        help=(
            "force action[t] = observation.state[t] even when action_topics are "
            "configured (joint mode only; EE mode is always act-from-obs). The "
            "future window is applied by LeRobot delta_timestamps at train time."
        ),
    )
    parser.add_argument(
        "--debug-plot-episodes", type=int, default=5,
        metavar="N",
        help="number of episodes to include in debug plots (default: 5)",
    )
    args = parser.parse_args(args)

    # Load configuration early — needed to determine data_space suffix for output dir
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
        log(f"Loaded config from: [dim]{args.config}[/dim]")
    else:
        config = ConfigLoader.get_default()
        log("Using default configuration")

    # Validate config eagerly — better error messages than cryptic extraction failures
    try:
        config.validate()
    except ConfigurationError as exc:
        console.print(f"\n[bold red]Configuration error:[/bold red] {exc}\n")
        exit(1)

    # Resolve output path: --output-path wins; otherwise
    # <output-dir>/<data_space>-space/<input-dir-name>/
    # The <data_space>-space/ subdir (ee-space/ or joint-space/) splits EE and
    # joint datasets cleanly when both are converted from the same raw sessions.
    # action_encoding="delta" gets its own "ee-delta-space/" subdir so baked-delta
    # datasets never land next to absolute-action ee-space/ datasets on disk. See
    # DataConfig.output_subdir (schema.py) — the single source of truth for this decision.
    space_suffix = config.output_subdir
    input_name = Path(args.input_dir.rstrip("/")).name
    if args.output_path:
        args.output_dir = args.output_path.rstrip("/")
    else:
        args.output_dir = str(
            Path(args.output_dir.rstrip("/")) / space_suffix / input_name
        )

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

    # Collect MCAP files once (reused for fps detection and conversion)
    all_mcap_files = collect_mcap_files(args.input_dir)

    # Always auto-detect input fps from all episodes (fast — reads MCAP summary only)
    ref_topic = list(config.camera_topic_mapping.keys())[0] if config.camera_topic_mapping else None
    ep_fps_raw = []
    if ref_topic:
        for f in all_mcap_files:
            v = McapReader(str(f)).estimate_fps(ref_topic)
            if v:
                ep_fps_raw.append(v)

    if ep_fps_raw:
        snapped = [snap_fps(v) for v in ep_fps_raw]
        input_fps = snap_fps(min(ep_fps_raw))
        input_fps_label = str(input_fps)
        if len(set(snapped)) > 1:
            input_fps_label = f"{input_fps} [yellow](mixed: {snapped})[/yellow]"
    else:
        input_fps = None
        input_fps_label = "unknown"

    # Resolve output fps: CLI --fps > auto-detect min > 30
    if args.fps is not None:
        fps = args.fps
        output_fps_label = f"{fps} (manual override)"
        if input_fps is not None and fps > input_fps:
            console.print(
                f"\n[bold red]ERROR: Output fps ({fps}) is higher than source session fps ({input_fps}).[/bold red]\n"
                "Upsampling is not supported — it creates duplicate frames and degrades dataset quality.\n"
                f"Use [bold]--fps {input_fps}[/bold] or lower, or omit --fps to use the source fps automatically.\n"
            )
            exit(1)
    elif input_fps is not None:
        fps = input_fps
        output_fps_label = f"{fps} (default as source)"
    else:
        fps = 30
        output_fps_label = "30 (default)"
        log("[yellow]Cannot detect fps — defaulting to 30[/yellow]")

    # Startup banner
    banner = Table(show_header=False, box=None, padding=(0, 2))
    banner.add_column(style="bold")
    banner.add_column()
    banner.add_row("Input directory", args.input_dir)
    banner.add_row("Output directory", args.output_dir)
    banner.add_row("HuggingFace Repo", repo_id)
    banner.add_row("Robot Type", args.robot_type)
    banner.add_row("Source Session FPS", input_fps_label)
    banner.add_row("Output FPS", output_fps_label)
    banner.add_row("Buffer", f"{args.buffer_seconds}s")
    banner.add_row("Video codec", args.vcodec)
    banner.add_row("Resume", "yes" if args.resume else "no")
    banner.add_row("Max episodes", str(args.max_episodes) if args.max_episodes else "all")
    banner.add_row("Data space", config.data_space)
    if args.act_from_obs and config.action_command_topics:
        banner.add_row("Action source", "act-from-obs (CLI override)")
    elif config.is_ee or not config.action_command_topics:
        banner.add_row("Action source", "act-from-obs (empty action_topics)")
    else:
        banner.add_row("Action source", "command topics")
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
            fps=fps,
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
            cli_act_from_obs=args.act_from_obs,
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
