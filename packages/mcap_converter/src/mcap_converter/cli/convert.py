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
from rich.markup import escape
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
from mcap_converter.cli.mcap_valid import default_report_paths
from mcap_converter.core.extractor import BufferedStreamExtractor
from mcap_converter.core.quality import SEVERITY_CRITICAL, SEVERITY_PASS, SEVERITY_WARNING
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


# Single source of truth for severity ordering, shared with core/quality.py's
# SEVERITY_PASS/WARNING/CRITICAL constants (rather than re-hardcoding the same
# three strings here) so a future rename can't drift between the two files.
_SEVERITY_ORDER = [SEVERITY_PASS, SEVERITY_WARNING, SEVERITY_CRITICAL]


def resolve_quality_skip_paths(quality_report_path: str | None, include_flagged: str) -> dict:
    """
    Read a mcap-valid JSON report and return {resolved_path: severity} for the
    episodes that fall ABOVE the --include-flagged threshold and should be
    skipped during conversion.

    include_flagged is an inclusive threshold, not an exclusion list: "pass"
    converts only pass-severity episodes (skips warning+critical); "warning"
    (the CLI default) also converts warning episodes, skipping only critical;
    "critical" converts everything, skipping nothing.
    """
    if quality_report_path is None:
        return {}

    with open(quality_report_path) as f:
        payload = json.load(f)

    threshold_idx = _SEVERITY_ORDER.index(include_flagged)
    skip_severities = set(_SEVERITY_ORDER[threshold_idx + 1 :])
    return {
        ep["path"]: ep["severity"]
        for ep in payload.get("episodes", [])
        if ep["severity"] in skip_severities
    }


def parse_episode_index_spec(spec: str, total_episodes: int) -> set:
    """
    Parse a 1-based episode index spec into a concrete set of indices.

    Colon ranges follow Python slice convention: the end is EXCLUSIVE, e.g.
    "1:4" selects episodes 1, 2, 3 (not 4) — same as Python's range(1, 4).
    An omitted start defaults to 1; an omitted end reaches the actual last
    episode inclusively (there's nothing to exclude when no end is given).
    """
    result: set = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_str, end_str = token.split(":", 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            try:
                start = int(start_str) if start_str else 1
                end = int(end_str) if end_str else total_episodes + 1
            except ValueError:
                raise ValueError(f"invalid episode range token: '{token}'")
            if start >= end:
                raise ValueError(f"invalid range '{token}': start must be less than end (end is exclusive)")
            if start < 1 or end > total_episodes + 1:
                raise ValueError(f"range '{token}' out of bounds — episodes are numbered 1 to {total_episodes}")
            result.update(range(start, end))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"invalid episode index token: '{token}'")
            if not (1 <= idx <= total_episodes):
                raise ValueError(f"episode index {idx} out of range (1-{total_episodes})")
            result.add(idx)
    return result


def quick_scan_joint_names(mcap_path: str, config: DataConfig) -> dict:
    """
    Quick scan to extract joint names from first JointState message.

    Only reads the first message, so memory-efficient for large files.

    In leader-follower mode: parses joint names to find observation (follower) joints.
    In quest teleop mode: all joints in the JointState topic are observations,
    so we group by arm without filtering by source/role prefix.

    Returns:
        Dictionary mapping robot prefix to joint names:
        - {"right": ["joint1", ...], "left": [...]} for multi-robot
        - {"": ["joint1", ...]} for single robot
        Joint names are extracted from the observation role.
    """
    reader = McapReader(mcap_path)
    joint_pattern = config.joint_name_pattern
    sep = joint_pattern.separator
    quest_mode = bool(config.action_topics)

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
    quality_skip_paths: dict | None = None,
    skip_episode_indices: set | None = None,
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

    # Get joint names
    log(f"Quick scan for joint names: [dim]{mcap_files[0]}[/dim]")
    joint_names = quick_scan_joint_names(str(mcap_files[0]), config)
    if not joint_names:
        raise ValueError("Cannot get joint names from reference MCAP (no observation joints found)")

    # Log detected robot mode
    robots = [r for r in joint_names.keys() if r]
    total_joints = sum(len(v) for v in joint_names.values())
    quest_mode = bool(config.action_topics)
    teleop_label = "[bold magenta]quest teleop[/bold magenta]" if quest_mode else "[bold cyan]leader-follower[/bold cyan]"
    if robots:
        log(f"Detected [bold cyan]bimanual[/bold cyan] robot ({teleop_label}): {robots}")
        for robot in sorted(robots):
            log(f"  {robot}: {joint_names[robot]}")
    else:
        log(f"Detected [bold cyan]single-arm[/bold cyan] robot ({teleop_label})")
        log(f"  joints: {joint_names.get('', [])}")
    log(f"Total joints: [bold]{total_joints}[/bold] (observation + action)")
    if quest_mode:
        for topic, topic_cfg in config.action_topics.items():
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

    # Copy conversion config for inference generation during training (skip if resuming)
    conversion_config_dest = os.path.join(output_dir, "conversion_config.yaml")
    if resume_from > 0:
        log(f"Skipping config copy — using existing [dim]{conversion_config_dest}[/dim]")
    elif config_path and os.path.exists(config_path):
        shutil.copy(config_path, conversion_config_dest)
        log(f"Copied conversion config: [dim]{conversion_config_dest}[/dim]")
    else:
        # Save config from DataConfig object
        import yaml

        config_to_save = {
            "robot_state_topic": config.robot_state_topic,
            "joint_names": {
                "separator": config.joint_name_pattern.separator,
                "source": config.joint_name_pattern.source,
                "arms": config.joint_name_pattern.arms,
            },
            "camera_topic_mapping": config.camera_topic_mapping,
        }
        if config.action_topics:
            config_to_save["action_topics"] = config.action_topics

        with open(conversion_config_dest, "w") as f:
            yaml.dump(
                config_to_save,
                f,
                default_flow_style=False,
            )
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
    episode_original_indices = []

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

        skip_paths = quality_skip_paths or {}
        for episode_idx, mcap_path in enumerate(mcap_files):
            if episode_idx < resume_from:
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(f"  [dim]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}  skipped (already converted)[/dim]")
                continue

            quality_severity = skip_paths.get(str(mcap_path.resolve()))
            if quality_severity is not None:
                color = "red" if quality_severity == "critical" else "yellow"
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(
                    f"  [{color}]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (quality: {quality_severity})[/{color}]"
                )
                continue

            if skip_episode_indices and (episode_idx + 1) in skip_episode_indices:
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(
                    f"  [cyan]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (manual index)[/cyan]"
                )
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
                episode_original_indices.append(episode_idx)
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
                episode_original_indices.append(episode_idx)
                continue

            for robot, counts in stream_extractor.get_action_fill_stats().items():
                filled = counts["hold_last"] + counts["fallback_to_observation"]
                if filled == 0 and counts["dropped"] == 0:
                    continue
                robot_label = robot or "action"
                dropped_suffix = (
                    f", [red]{counts['dropped']} dropped[/red]" if counts["dropped"] else ""
                )
                console.print(
                    f"    [yellow]↺[/yellow] {robot_label}: {counts['exact']} exact, "
                    f"{counts['hold_last']} hold-last, "
                    f"{counts['fallback_to_observation']} fallback-to-obs{dropped_suffix}"
                )

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
            episode_original_indices.append(episode_idx)
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
            "  Run [bold]mcap-valid[/bold] on your MCAP to see all recorded topics and message types.\n"
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
                action_from_observation_n=config.action_from_observation_n,
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
    for j, i in enumerate(episode_original_indices):
        mcap_path = mcap_files[i]
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
  mcap-convert -i data/raw/my-session -o data/datasets --fps 15 --push-to-hub
  mcap-convert -i data/raw/my-session -o data/datasets --max-episodes 5
  mcap-convert -i data/raw/my-session -o data/datasets --resume
  mcap-convert -i data/raw/my-session -o data/datasets  # default: critical episodes skipped automatically
  mcap-convert -i data/raw/my-session -o data/datasets --include-flagged critical  # convert everything, even critical episodes
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
        "--act-from-obs-n-step", type=int, default=None,
        metavar="N",
        help="override action_from_observation_n in config: action[t] = observation[t+N] (default: use config value, factory default 10)",
    )
    parser.add_argument(
        "--debug-plot-episodes", type=int, default=5,
        metavar="N",
        help="number of episodes to include in debug plots (default: 5)",
    )
    parser.add_argument(
        "--quality-report", type=str, default=None,
        help=(
            "path to a mcap-valid JSON report. A report is REQUIRED to run mcap-convert — "
            "if omitted, it is auto-discovered at <input-dir>/mcap_valid_reports/report.json "
            "(run `mcap-valid -i INPUT_DIR` first to generate it); if neither is found, "
            "mcap-convert exits with an error before touching the output directory"
        ),
    )
    parser.add_argument(
        "--include-flagged",
        choices=_SEVERITY_ORDER,
        default=SEVERITY_WARNING,
        help=(
            "highest severity tier to include when converting, per the quality "
            "report. Inclusive threshold: 'pass' converts only clean episodes "
            "(skips warning AND critical); 'warning' (default) also converts "
            "warning-level episodes, skipping only critical ones automatically; "
            "'critical' converts every episode regardless of severity, skipping "
            "nothing."
        ),
    )
    parser.add_argument(
        "--skip-episode-idx", type=str, default=None,
        help=(
            "manually skip specific episodes by 1-based index, independent of "
            "--quality-report. Accepts a comma-separated list (1,2,5,6), a "
            "colon range with an EXCLUSIVE end matching Python slice convention "
            "(1:4 selects episodes 1,2,3 — NOT 4), an open-ended range (2: or :4), "
            "or a mix (1,3:5,8). Whitespace is tolerated."
        ),
    )
    args = parser.parse_args(args)

    # Resolve output path: --output-path wins; otherwise <output-dir>/<input-dir-name>/
    input_name = Path(args.input_dir.rstrip("/")).name
    if args.output_path:
        args.output_dir = args.output_path.rstrip("/")
    else:
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

    # Load configuration
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
        log(f"Loaded config from: [dim]{args.config}[/dim]")
    else:
        config = ConfigLoader.get_default()
        log("Using default configuration")

    # ── Mandatory quality-report gate ──────────────────────────────────
    # mcap-convert refuses to run without a mcap-valid quality report (explicit
    # --quality-report, or auto-discovered at the default path) so bad
    # recordings are caught before they enter a dataset. This only checks that
    # a report FILE exists — --include-flagged (below) is a separate mechanism
    # that reads the report's *contents* and defaults to "warning", so only
    # critical episodes are skipped automatically; pass --include-flagged
    # critical to opt out entirely and convert everything.
    report_path = args.quality_report
    default_json, _ = default_report_paths(Path(args.input_dir))
    if report_path is None:
        if default_json.is_file():
            report_path = str(default_json)
    if report_path is None or not Path(report_path).is_file():
        # escape(): input-dir/report paths are user/data-controlled and could
        # otherwise be parsed as Rich markup (e.g. a path containing "[red]").
        console.print(
            "\n[bold red]ERROR: No mcap-valid quality report found for this input.[/bold red]\n"
            "mcap-convert requires a quality report to exist before conversion, so bad\n"
            "recordings are caught before they enter a dataset.\n"
            f"Run mcap-valid first:\n"
            f"  [bold]uv run mcap-valid -i {escape(args.input_dir)}[/bold]\n"
            f"then re-run this command — the report is auto-discovered at\n"
            f"  {escape(str(default_json))}\n"
            "or pass --quality-report PATH to point at a report elsewhere.\n"
        )
        exit(1)

    quality_skip_paths = resolve_quality_skip_paths(report_path, args.include_flagged)

    if args.act_from_obs_n_step is not None:
        config.action_from_observation_n = args.act_from_obs_n_step
        log(f"action_from_observation_n overridden to [bold]{args.act_from_obs_n_step}[/bold] via --act-from-obs-n-step")

    # Collect MCAP files once (reused for fps detection and conversion)
    all_mcap_files = collect_mcap_files(args.input_dir)

    # Validate --skip-episode-idx early (before any output-dir mutation below)
    skip_episode_indices = None
    if args.skip_episode_idx:
        try:
            skip_episode_indices = parse_episode_index_spec(args.skip_episode_idx, len(all_mcap_files))
        except ValueError as exc:
            console.print(f"[red]✗ --skip-episode-idx error: {exc}[/red]")
            exit(1)
        log(f"Manually skipping {len(skip_episode_indices)} episode(s) by index: {sorted(skip_episode_indices)}")

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
    if config.action_from_observation:
        n_label = str(config.action_from_observation_n)
        if args.act_from_obs_n_step is not None:
            n_label += " [yellow](CLI override)[/yellow]"
        banner.add_row("act-from-obs n", n_label)
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
            quality_skip_paths=quality_skip_paths,
            skip_episode_indices=skip_episode_indices,
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
