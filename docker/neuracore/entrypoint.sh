#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash

# --- CycloneDDS / distributed discovery ---
if [ -n "${ROS_DOMAIN_ID}" ]; then
    export ROS_DOMAIN_ID
    echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} — distributed mode"
else
    export ROS_LOCALHOST_ONLY=1
    echo "[entrypoint] ROS_DOMAIN_ID not set — localhost-only mode"
fi

if [ -n "${RMW_IMPLEMENTATION}" ]; then
    export RMW_IMPLEMENTATION
    echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
fi

if [ -n "${CYCLONEDDS_URI}" ]; then
    export CYCLONEDDS_URI
    echo "[entrypoint] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
fi

# Skip neuracore's P2P live-data listeners — we're the consumer of local
# sync points, not a cross-machine webrtc participant.
export NEURACORE_CONSUME_LIVE_DATA="${NEURACORE_CONSUME_LIVE_DATA:-0}"

if [ -z "${NEURACORE_API_KEY}" ]; then
    echo "[entrypoint] WARNING: NEURACORE_API_KEY not set — inference will fail to init" >&2
fi

exec "$@"
