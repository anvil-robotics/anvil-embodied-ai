"""Launch file for LeRobot inference node.

Configuration is loaded from a YAML config file that specifies:
- Camera topic mapping (ROS topic -> ML model camera name)
- Joint state mapping (filter and reorder joints)
- Arm configuration (command topics and action indices)

See configs/lerobot_control/inference_default.yaml for the default configuration.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description"""

    # Arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to trained model checkpoint (REQUIRED)'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to inference config YAML file'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Inference device (cuda or cpu)'
    )

    control_freq_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='30.0',
        description='Control loop frequency (Hz)'
    )

    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='act',
        description='Model type (act, diffusion, or smolvla)'
    )

    max_position_delta_arg = DeclareLaunchArgument(
        'max_position_delta',
        default_value='0.01',
        description='Max position delta per step in radians (safety limit)'
    )

    use_amp_arg = DeclareLaunchArgument(
        'use_amp',
        default_value='false',
        description='Automatic Mixed Precision'
    )

    clip_actions_arg = DeclareLaunchArgument(
        'clip_actions',
        default_value='true',
        description='Clip actions to training bounds'
    )

    throttle_action_logs_arg = DeclareLaunchArgument(
        'throttle_action_logs',
        default_value='true',
        description='Throttle action logging'
    )

    throttle_duration_arg = DeclareLaunchArgument(
        'throttle_duration',
        default_value='5.0',
        description='Log throttle duration in seconds'
    )

    # Node
    inference_node = Node(
        package='lerobot_control',
        executable='inference_node',
        name='lerobot_inference',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'config_file': LaunchConfiguration('config_file'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'device': LaunchConfiguration('device'),
            'model_type': LaunchConfiguration('model_type'),
            'max_position_delta': LaunchConfiguration('max_position_delta'),
            'use_amp': LaunchConfiguration('use_amp'),
            'clip_actions': LaunchConfiguration('clip_actions'),
            'throttle_action_logs': LaunchConfiguration('throttle_action_logs'),
            'throttle_duration': LaunchConfiguration('throttle_duration'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        config_file_arg,
        device_arg,
        control_freq_arg,
        model_type_arg,
        max_position_delta_arg,
        use_amp_arg,
        clip_actions_arg,
        throttle_action_logs_arg,
        throttle_duration_arg,
        inference_node,
    ])
