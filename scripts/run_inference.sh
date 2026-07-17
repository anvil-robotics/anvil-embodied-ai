#!/usr/bin/env bash
# Entry point for ALL runtime inference scenarios.
# Handles directory ownership for monitor output and auto-plots CSV on exit.
#
# Usage:
#   ./scripts/run_inference.sh [OPTIONS] [COMPOSE_ARGS...]
#
# Options:
#   --fake-hardware    Use docker-compose.fake-hardware.yml (DDS bridge test, no real robot)
#   --monitor-enable          Enable monitor profile; for production also sets MONITOR_ENABLE=true,
#                      pre-creates MONITOR_OUTPUT_DIR as current user, and plots CSV on exit
#   --echo-topic-only  Subscribe + log FPS without loading a model (sets ECHO_TOPIC_ONLY=true);
#                      useful to verify DDS connectivity on the GPU PC without a checkpoint
#   --debug            Enable debug metrics: action smoothness, queue depth, Action FPS (sets DEBUG=true)
#   -h, --help         Show this message
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
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --monitor-enable up --build
#
#   # Fake-hardware DDS test (FPS monitor only, no GPU)
#   ./scripts/run_inference.sh --fake-hardware --monitor-enable up --build
#
#   # Verify DDS connectivity without a model (no MODEL_PATH needed)
#   ./scripts/run_inference.sh --echo-topic-only up --build
#
#   # Fake-hardware full inference pipeline
#   # NOTE: --profile is a docker compose global flag and must come BEFORE the subcommand.
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --fake-hardware --profile inference up --build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
FAKE_HARDWARE=false
MONITOR_REQUESTED=false
ECHO_TOPIC_ONLY_REQUESTED=false
DEBUG_REQUESTED=false
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
        --monitor-enable)
            MONITOR_REQUESTED=true
            shift
            ;;
        --echo-topic-only)
            ECHO_TOPIC_ONLY_REQUESTED=true
            shift
            ;;
        --debug)
            DEBUG_REQUESTED=true
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

# ECHO_TOPIC_ONLY: subscribe + log FPS without loading a model (DDS connectivity check)
if [[ "$ECHO_TOPIC_ONLY_REQUESTED" == true ]]; then
    export ECHO_TOPIC_ONLY=true
fi

# DEBUG: enable extra metrics inside the container
if [[ "$DEBUG_REQUESTED" == true ]]; then
    export DEBUG=true
fi

# Auto-detect ACTION_TYPE from model checkpoint's anvil_config.json.
# Checks pretrained_model/ subdirectory first, then checkpoint root, then HF snapshots/.
# Always overrides any existing ACTION_TYPE value.
if [[ -n "${MODEL_PATH:-}" ]]; then
    _model_root="${MODEL_PATH}"
    _anvil_config=""
    # 1. pretrained_model/anvil_config.json (most common)
    if [[ -f "${_model_root}/pretrained_model/anvil_config.json" ]]; then
        _anvil_config="${_model_root}/pretrained_model/anvil_config.json"
    # 2. root-level anvil_config.json
    elif [[ -f "${_model_root}/anvil_config.json" ]]; then
        _anvil_config="${_model_root}/anvil_config.json"
    # 3. HF cache snapshot: snapshots/<hash>/anvil_config.json
    elif [[ -d "${_model_root}/snapshots" ]]; then
        _anvil_config=$(find "${_model_root}/snapshots" -maxdepth 2 -name "anvil_config.json" | sort -r | head -1)
    fi

    if [[ -n "${_anvil_config}" && -f "${_anvil_config}" ]]; then
        _detected=$(python3 -c "import json; d=json.load(open('${_anvil_config}')); print(d.get('action_type','joint_abs'))" 2>/dev/null || true)
        if [[ -n "${_detected}" ]]; then
            export ACTION_TYPE="${_detected}"
            echo "[run_inference] ACTION_TYPE=${ACTION_TYPE} (auto-detected from $(basename $(dirname ${_anvil_config})))"
        fi
    fi
fi

# Auto-detect JOINT_NAMES from CONFIG_FILE for joint_abs models (needed by monitor node).
# For EE models (ee_abs/ee_rel), joint_names is empty — monitor uses dim indices instead.
if [[ "${ACTION_TYPE:-joint_abs}" == "joint_abs" && -n "${CONFIG_FILE:-}" && -f "${CONFIG_FILE}" ]]; then
    _joint_names=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG_FILE}'))
order = cfg.get('joint_names', {}).get('model_joint_order', [])
print(','.join(order))
" 2>/dev/null || true)
    if [[ -n "${_joint_names}" ]]; then
        export JOINT_NAMES="${_joint_names}"
        echo "[run_inference] JOINT_NAMES=${JOINT_NAMES}"
    fi
fi

# Production-only: MONITOR_ENABLE env var triggers inference_monitor_node inside the container;
# also pre-create the output dir (and the videos/ subdir the container writes
# per-camera frames into) as current user so Docker can't claim root ownership —
# otherwise the ffmpeg encode step below (running as the host user) can't create
# <camera>.mp4 inside a root-owned videos/ directory.
REAL_MONITOR=false
if [[ "$MONITOR_REQUESTED" == true && "$FAKE_HARDWARE" == false ]]; then
    REAL_MONITOR=true
    export MONITOR_ENABLE=true
    MONITOR_DIR="${MONITOR_OUTPUT_DIR:-${REPO_ROOT}/monitor_output}"
    export MONITOR_OUTPUT_DIR="$MONITOR_DIR"
    mkdir -p "$MONITOR_DIR/videos"
    echo "[run_inference] Monitor enabled → output: $MONITOR_DIR"
fi

echo "[run_inference] compose: $(basename "$COMPOSE_FILE") | args: ${PASSTHROUGH[*]:-<none>}"

# Run docker compose and capture exit code (don't let set -e abort before plotting)
set +e
docker compose -f "$COMPOSE_FILE" "${PASSTHROUGH[@]}" --remove-orphans
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

    # Batch-convert per-camera frame sequences to mp4 now that the container has
    # fully exited — deliberately done here (host side, after shutdown) rather
    # than as a cv2.VideoWriter inside the container: an mp4 container's index is
    # only written on a clean release(), so any abrupt kill (SIGKILL, or a slow
    # shutdown racing the container's stop_grace_period) produced a corrupted,
    # unplayable file in practice. Individual JPEG frames can't be corrupted this
    # way — a kill just means the tail frames are missing.
    VIDEO_DIR="${MONITOR_DIR}/videos"
    if [[ -d "$VIDEO_DIR" ]] && command -v ffmpeg >/dev/null 2>&1; then
        shopt -s nullglob
        for frames_dir in "$VIDEO_DIR"/*_frames; do
            camera_name="$(basename "$frames_dir" _frames)"
            mp4_out="${VIDEO_DIR}/${camera_name}.mp4"
            n_frames=$(find "$frames_dir" -name "frame_*.jpg" | wc -l)
            if [[ "$n_frames" -eq 0 ]]; then
                echo "[run_inference] WARNING: no frames found in $frames_dir, skipping"
                continue
            fi
            echo "[run_inference] Encoding $camera_name ($n_frames frames) -> $mp4_out"
            if ffmpeg -y -loglevel error \
                -framerate "${CONTROL_FREQ:-30.0}" \
                -i "${frames_dir}/frame_%06d.jpg" \
                -c:v libx264 -pix_fmt yuv420p \
                "$mp4_out"; then
                rm -rf "$frames_dir"
                echo "[run_inference] Video saved: $mp4_out"
            else
                echo "[run_inference] WARNING: ffmpeg failed for $camera_name (exit $?) — frames kept at $frames_dir"
            fi
        done
        shopt -u nullglob
    elif [[ -d "$VIDEO_DIR" ]]; then
        echo "[run_inference] WARNING: ffmpeg not found on host — frame sequences left at $VIDEO_DIR"
    fi
fi

exit "$COMPOSE_EXIT"
