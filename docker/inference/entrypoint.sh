#!/bin/bash
set -e

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash

exec "$@"
