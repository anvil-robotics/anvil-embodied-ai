"""Launch file for Neuracore local inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_file_arg = DeclareLaunchArgument(
        "model_file",
        default_value="",
        description="Path to local .nc.zip (takes precedence over train_run_name)",
    )
    train_run_name_arg = DeclareLaunchArgument(
        "train_run_name",
        default_value="",
        description="Neuracore training run name (downloaded on first use)",
    )
    robot_name_arg = DeclareLaunchArgument(
        "robot_name",
        default_value="anvil_openarm",
        description="Neuracore robot name — must match the trained embodiment",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="",
        description="Optional URDF path passed to nc.connect_robot",
    )
    inference_rate_arg = DeclareLaunchArgument(
        "inference_rate_hz",
        default_value="30.0",
        description="Control loop rate (Hz)",
    )
    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        description="If true, log predicted actions instead of publishing commands",
    )
    max_joint_delta_arg = DeclareLaunchArgument(
        "max_joint_delta",
        default_value="0.05",
        description="Per-tick arm joint target is clamped to current ± this (radians)",
    )
    predictions_log_arg = DeclareLaunchArgument(
        "predictions_log",
        default_value="",
        description="If set, write per-tick predictions to this CSV path (raw policy targets + current state)",
    )

    inference_node = Node(
        package="neuracore_control",
        executable="inference_node",
        name="neuracore_inference",
        output="screen",
        parameters=[
            {
                "model_file": LaunchConfiguration("model_file"),
                "train_run_name": LaunchConfiguration("train_run_name"),
                "robot_name": LaunchConfiguration("robot_name"),
                "urdf_path": LaunchConfiguration("urdf_path"),
                "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
                "debug": LaunchConfiguration("debug"),
                "max_joint_delta": LaunchConfiguration("max_joint_delta"),
                "predictions_log": LaunchConfiguration("predictions_log"),
            }
        ],
    )

    return LaunchDescription(
        [
            model_file_arg,
            train_run_name_arg,
            robot_name_arg,
            urdf_path_arg,
            inference_rate_arg,
            debug_arg,
            max_joint_delta_arg,
            predictions_log_arg,
            inference_node,
        ]
    )
