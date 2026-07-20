"""Launch file for LeRobot inference node.

Configuration is loaded from a YAML config file that specifies:
- Camera topic mapping (ROS topic -> ML model camera name)
- Joint state mapping (filter and reorder joints)
- Arm configuration (command topics and action indices)

See configs/lerobot_control/inference_default.yaml for the default configuration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description"""

    # Arguments
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="", description="Path to trained model checkpoint (REQUIRED)"
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file", default_value="", description="Path to inference config YAML file"
    )

    device_arg = DeclareLaunchArgument(
        "device", default_value="cuda", description="Inference device (cuda or cpu)"
    )

    control_freq_arg = DeclareLaunchArgument(
        "control_frequency", default_value="30.0", description="Control loop frequency (Hz)"
    )

    deterministic_arg = DeclareLaunchArgument(
        "deterministic", default_value="false", description="Enable deterministic mode"
    )

    deterministic_seed_arg = DeclareLaunchArgument(
        "deterministic_seed", default_value="42", description="Random seed for deterministic mode"
    )

    echo_topic_only_arg = DeclareLaunchArgument(
        "echo_topic_only",
        default_value="false",
        description="Echo-topic-only mode: subscribe + log FPS, no model loaded or actions published",
    )

    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        description="Enable debug metrics: action smoothness, queue depth stats, Action FPS",
    )

    debug_image_dir_arg = DeclareLaunchArgument(
        "debug_image_dir",
        default_value="",
        description="Save pre-model input frames to this directory (one sub-dir per camera). Empty = disabled.",
    )

    monitor_enable_arg = DeclareLaunchArgument(
        "monitor_enable",
        default_value="false",
        description="Publish /monitor/obs_state, /monitor/raw_output, /monitor/control_cmd for inference_monitor_node",
    )

    monitor_video_dir_arg = DeclareLaunchArgument(
        "monitor_video_dir",
        default_value="",
        description="Dump each camera's frames to <dir>/<camera_name>_frames/ for the "
        "full episode; run_inference.sh converts them to <dir>/<camera_name>.mp4 with "
        "ffmpeg after shutdown. Only takes effect when monitor_enable is true. "
        "Empty = disabled.",
    )

    mock_ee_pose_echo_arg = DeclareLaunchArgument(
        "mock_ee_pose_echo",
        default_value="false",
        description="Fake-hardware-only: true iff /ee_pose_{arm} is the mock's MockEEPose "
        "echo (sequence-guarded against stale echoes) rather than real hardware's plain "
        "CommandedEEPose (no guard at all — see MockEEPose.msg/ee_obs_sequence_guard.py). "
        "Set by the deployment, not per-model config: docker-compose.fake-hardware.yml "
        "sets this true wherever mock-robot is co-launched; real hardware leaves it false.",
    )

    # Node
    inference_node = Node(
        package="lerobot_control",
        executable="inference_node",
        name="lerobot_inference",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "config_file": LaunchConfiguration("config_file"),
                "control_frequency": LaunchConfiguration("control_frequency"),
                "device": LaunchConfiguration("device"),
                "deterministic": LaunchConfiguration("deterministic"),
                "deterministic_seed": LaunchConfiguration("deterministic_seed"),
                "echo_topic_only": LaunchConfiguration("echo_topic_only"),
                "debug": LaunchConfiguration("debug"),
                "debug_image_dir": LaunchConfiguration("debug_image_dir"),
                "monitor_enable": LaunchConfiguration("monitor_enable"),
                "monitor_video_dir": LaunchConfiguration("monitor_video_dir"),
                "mock_ee_pose_echo": LaunchConfiguration("mock_ee_pose_echo"),
            }
        ],
    )

    return LaunchDescription(
        [
            model_path_arg,
            config_file_arg,
            device_arg,
            control_freq_arg,
            deterministic_arg,
            deterministic_seed_arg,
            echo_topic_only_arg,
            debug_arg,
            debug_image_dir_arg,
            monitor_enable_arg,
            monitor_video_dir_arg,
            mock_ee_pose_echo_arg,
            inference_node,
        ]
    )
