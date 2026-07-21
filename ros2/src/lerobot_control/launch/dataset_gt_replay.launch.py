"""Launch file for the dataset GT-replayer node.

Replays a converted dataset's recorded ground-truth ``action`` rows through the
real inference pipeline (see ``dataset_gt_replayer_node.py`` module docstring
and claude_docs/dataset-gt-replayer-plan.md for the design). Mirrors
``inference.launch.py`` — same ``config_file``/topics/publishers — with
``model_path``/``deterministic``/``echo_topic_only`` replaced by the replay
params ``dataset``/``episode``/``loop``/``hold_last``/``dry_run``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description"""

    dataset_arg = DeclareLaunchArgument(
        "dataset", default_value="", description="Path to the converted LeRobot dataset (REQUIRED)"
    )

    episode_arg = DeclareLaunchArgument(
        "episode", default_value="0", description="Episode index to replay"
    )

    loop_arg = DeclareLaunchArgument(
        "loop", default_value="false", description="Restart from episode start when replay completes"
    )

    hold_last_arg = DeclareLaunchArgument(
        "hold_last",
        default_value="true",
        description="On replay completion: hold the last published command (true) or "
        "shut the node down (false). Ignored when loop:=true.",
    )

    dry_run_arg = DeclareLaunchArgument(
        "dry_run",
        default_value="false",
        description="Log each recorded action row without publishing it",
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file", default_value="", description="Path to inference config YAML file"
    )

    device_arg = DeclareLaunchArgument(
        "device", default_value="cpu", description="Device passed to the observation strategy (no model runs)"
    )

    control_freq_arg = DeclareLaunchArgument(
        "control_frequency", default_value="30.0", description="Control loop frequency (Hz)"
    )

    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug logging"
    )

    debug_image_dir_arg = DeclareLaunchArgument(
        "debug_image_dir",
        default_value="",
        description="Save pre-publish input frames to this directory (one sub-dir per camera). Empty = disabled.",
    )

    monitor_enable_arg = DeclareLaunchArgument(
        "monitor_enable",
        default_value="false",
        description="Publish /monitor/obs_state, /monitor/raw_output, /monitor/control_cmd for inference_monitor_node",
    )

    monitor_video_dir_arg = DeclareLaunchArgument(
        "monitor_video_dir",
        default_value="",
        description="Dump each camera's frames to <dir>/<camera_name>_frames/ for the full "
        "replay run. Only takes effect when monitor_enable is true. Empty = disabled.",
    )

    completion_signal_path_arg = DeclareLaunchArgument(
        "completion_signal_path",
        default_value="",
        description="Path to write a JSON sentinel to on episode completion/interruption/"
        "homing failure (see dataset_gt_replayer_node.py's _write_signal). Empty = disabled. "
        "Used by scripts/gt_replay_human_eval.py to detect episode completion.",
    )

    home_before_replay_arg = DeclareLaunchArgument(
        "home_before_replay",
        default_value="true",
        description="EE mode only: command the robot to this episode's frame-0 recorded "
        "pose and wait for confirmed arrival before GT playback begins.",
    )

    home_atol_pos_m_arg = DeclareLaunchArgument(
        "home_atol_pos_m", default_value="0.05",
        description="Homing arrival position tolerance (m) — coarser than "
        "GtReplayVerifierNode's trajectory-match tolerance; this just checks 'did we arrive'. "
        "Well above anvil_eval's real-hardware pass/fail threshold (0.02m) — observed live, "
        "across several attempts, plateauing anywhere up to ~0.039m without fully closing the "
        "gap within a realistic homing_timeout_sec, so this sets comfortable margin above the "
        "worst observed value rather than sitting right at the edge of it.",
    )

    home_atol_rot_deg_arg = DeclareLaunchArgument(
        "home_atol_rot_deg", default_value="10.0",
        description="Homing arrival orientation tolerance (deg) — same reasoning as "
        "home_atol_pos_m, well above anvil_eval's 5.0deg real-hardware threshold.",
    )

    homing_timeout_sec_arg = DeclareLaunchArgument(
        "homing_timeout_sec", default_value="30.0",
        description="Give up homing (write homing_failed, shut down) after this many seconds.",
    )

    home_max_pos_delta_m_arg = DeclareLaunchArgument(
        "home_max_pos_delta_m", default_value="0.01",
        description="Max homing approach speed, position: metres of motion per publish tick "
        "(inference_node's action_limiter — joint-space delta limiting — is not applied in "
        "EE mode, so this is the equivalent safety ramp for the one-shot homing command).",
    )

    home_max_rot_delta_deg_arg = DeclareLaunchArgument(
        "home_max_rot_delta_deg", default_value="2.0",
        description="Max homing approach speed, orientation: degrees of rotation per publish tick.",
    )

    mock_ee_pose_echo_arg = DeclareLaunchArgument(
        "mock_ee_pose_echo",
        default_value="false",
        description="Fake-hardware-only: true iff /ee_pose_{arm} is the mock's MockEEPose "
        "echo (sequence-guarded) rather than real hardware's plain CommandedEEPose (no "
        "guard). See inference.launch.py's arg of the same name.",
    )

    replayer_node = Node(
        package="lerobot_control",
        executable="dataset_gt_replayer_node",
        name="dataset_gt_replayer",
        output="screen",
        parameters=[
            {
                "dataset": LaunchConfiguration("dataset"),
                "episode": LaunchConfiguration("episode"),
                "loop": LaunchConfiguration("loop"),
                "hold_last": LaunchConfiguration("hold_last"),
                "dry_run": LaunchConfiguration("dry_run"),
                "config_file": LaunchConfiguration("config_file"),
                "device": LaunchConfiguration("device"),
                "control_frequency": LaunchConfiguration("control_frequency"),
                "debug": LaunchConfiguration("debug"),
                "debug_image_dir": LaunchConfiguration("debug_image_dir"),
                "monitor_enable": LaunchConfiguration("monitor_enable"),
                "monitor_video_dir": LaunchConfiguration("monitor_video_dir"),
                "completion_signal_path": LaunchConfiguration("completion_signal_path"),
                "home_before_replay": LaunchConfiguration("home_before_replay"),
                "home_atol_pos_m": LaunchConfiguration("home_atol_pos_m"),
                "home_atol_rot_deg": LaunchConfiguration("home_atol_rot_deg"),
                "homing_timeout_sec": LaunchConfiguration("homing_timeout_sec"),
                "home_max_pos_delta_m": LaunchConfiguration("home_max_pos_delta_m"),
                "home_max_rot_delta_deg": LaunchConfiguration("home_max_rot_delta_deg"),
                "mock_ee_pose_echo": LaunchConfiguration("mock_ee_pose_echo"),
            }
        ],
    )

    return LaunchDescription(
        [
            dataset_arg,
            episode_arg,
            loop_arg,
            hold_last_arg,
            dry_run_arg,
            config_file_arg,
            device_arg,
            control_freq_arg,
            debug_arg,
            debug_image_dir_arg,
            monitor_enable_arg,
            monitor_video_dir_arg,
            completion_signal_path_arg,
            home_before_replay_arg,
            home_atol_pos_m_arg,
            home_atol_rot_deg_arg,
            homing_timeout_sec_arg,
            home_max_pos_delta_m_arg,
            home_max_rot_delta_deg_arg,
            mock_ee_pose_echo_arg,
            replayer_node,
        ]
    )
