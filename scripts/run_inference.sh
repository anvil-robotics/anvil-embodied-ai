#!/usr/bin/env bash
# Entry point for ALL runtime inference scenarios.
# Handles directory ownership for monitor output and auto-plots CSV on exit.
#
# Usage:
#   ./scripts/run_inference.sh [OPTIONS] [COMPOSE_ARGS...]
#
# Options:
#   --fake-hardware   Use docker-compose.fake-hardware.yml (DDS bridge test, no real robot)
#   --monitor         Enable monitor profile; for production also sets MONITOR_ENABLE=true,
#                     pre-creates MONITOR_OUTPUT_DIR as current user, and plots CSV on exit
#   -h, --help        Show this message
#
# All other arguments (e.g. up --build, down, logs) are passed directly to docker compose.
#
# Environment variables:
#   MONITOR_OUTPUT_DIR   Host dir for monitor CSV/PNG (default: ./monitor_output)
#   MODEL_PATH           Path to model checkpoint (required for production inference)
#   CONFIG_FILE          Path to inference config YAML (default: ./configs/lerobot_control/inference_default.yaml)
#   IMAGE_TAG            Docker image tag (default: latest)
#   ROS_DOMAIN_ID        ROS domain ID
#   HF_CACHE             HuggingFace cache dir (needed for VLA models)
#
# Examples:
#   # Production inference (real robot), no monitor
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh up --build
#
#   # Production inference with real-time monitor + auto-plot
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --monitor up --build
#
#   # Fake-hardware DDS test (FPS monitor only, no GPU)
#   ./scripts/run_inference.sh --fake-hardware --monitor up --build
#
#   # Fake-hardware full inference pipeline
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --fake-hardware up --build --profile inference

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
FAKE_HARDWARE=false
MONITOR_REQUESTED=false
PASSTHROUGH=()

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

# Parse our flags; collect everything else for docker compose
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fake-hardware)
            FAKE_HARDWARE=true
            COMPOSE_FILE="${REPO_ROOT}/docker-compose.fake-hardware.yml"
            shift
            ;;
        --monitor)
            MONITOR_REQUESTED=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

# Always inject --profile monitor when requested
if [[ "$MONITOR_REQUESTED" == true ]]; then
    PASSTHROUGH=("--profile" "monitor" "${PASSTHROUGH[@]}")
fi

# Production-only: MONITOR_ENABLE env var triggers inference_monitor_node inside the container;
# also pre-create the output dir as current user so Docker can't claim root ownership.
REAL_MONITOR=false
if [[ "$MONITOR_REQUESTED" == true && "$FAKE_HARDWARE" == false ]]; then
    REAL_MONITOR=true
    export MONITOR_ENABLE=true
    MONITOR_DIR="${MONITOR_OUTPUT_DIR:-${REPO_ROOT}/monitor_output}"
    export MONITOR_OUTPUT_DIR="$MONITOR_DIR"
    mkdir -p "$MONITOR_DIR"
    echo "[run_inference] Monitor enabled → output: $MONITOR_DIR"
fi

echo "[run_inference] compose: $(basename "$COMPOSE_FILE") | args: ${PASSTHROUGH[*]:-<none>}"

# Run docker compose and capture exit code (don't let set -e abort before plotting)
set +e
docker compose -f "$COMPOSE_FILE" "${PASSTHROUGH[@]}"
COMPOSE_EXIT=$?
set -e

# Auto-plot monitor CSV after production monitor run
if [[ "$REAL_MONITOR" == true ]]; then
    CSV="${MONITOR_DIR}/inference_data.csv"
    PNG="${MONITOR_DIR}/inference_report.png"
    if [[ -f "$CSV" ]]; then
        echo "[run_inference] Plotting monitor data: $CSV"
        uv run python "${REPO_ROOT}/scripts/plot_monitor_csv.py" "$CSV" -o "$PNG" \
            && echo "[run_inference] Report saved: $PNG" \
            || echo "[run_inference] WARNING: plot_monitor_csv.py failed (exit $?)"
    else
        echo "[run_inference] WARNING: monitor CSV not found at $CSV"
    fi
fi

exit "$COMPOSE_EXIT"
