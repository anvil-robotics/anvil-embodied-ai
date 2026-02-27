#!/bin/bash
set -e

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash

# --- CycloneDDS / distributed discovery ---
# If ROS_DOMAIN_ID is set and non-empty, configure for distributed mode.
# Otherwise, fall back to localhost-only communication.
if [ -n "${ROS_DOMAIN_ID}" ]; then
    export ROS_DOMAIN_ID
    echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} — distributed mode"
else
    export ROS_LOCALHOST_ONLY=1
    echo "[entrypoint] ROS_DOMAIN_ID not set — localhost-only mode"
fi

# Export CycloneDDS middleware if configured
if [ -n "${RMW_IMPLEMENTATION}" ]; then
    export RMW_IMPLEMENTATION
    echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
fi

if [ -n "${CYCLONEDDS_URI}" ]; then
    export CYCLONEDDS_URI
    echo "[entrypoint] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
fi

exec "$@"
