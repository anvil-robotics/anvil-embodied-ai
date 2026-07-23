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
#   --preflight-only  Validate real-robot arm routing, then exit without running Compose
#   -h, --help         Show this message
#
# All other arguments (e.g. up --build, down, logs) are passed directly to docker compose.
#
# Environment variables:
#   MONITOR_OUTPUT_DIR   Host dir for monitor CSV/PNG (default: ./monitor_output)
#   MODEL_PATH           Path to model checkpoint (required for production inference)
#   CONFIG_FILE          Path to inference config YAML (default: ./configs/lerobot_control/inference_default.yaml)
#   INFERENCE_ARM        Required for real-robot startup: left | right | bimanual
#   MAX_RUN_SECONDS      Maximum active inference duration after model load (0 = unlimited)
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
PREFLIGHT_ONLY=false
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
        --preflight-only)
            PREFLIGHT_ONLY=true
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

read_dotenv_value() {
    sed -n "s/^${1}=//p" "${REPO_ROOT}/.env" | tail -n 1
}

if [[ -f "${REPO_ROOT}/.env" ]]; then
    MODEL_PATH="${MODEL_PATH:-$(read_dotenv_value MODEL_PATH)}"
    CONFIG_FILE="${CONFIG_FILE:-$(read_dotenv_value CONFIG_FILE)}"
    INFERENCE_ARM="${INFERENCE_ARM:-$(read_dotenv_value INFERENCE_ARM)}"
fi

# Fail closed on real-robot arm routing before a command-publishing service starts.
START_REQUESTED="${PREFLIGHT_ONLY}"
for _arg in "${PASSTHROUGH[@]}"; do
    case "${_arg}" in
        up|start|restart|run) START_REQUESTED=true; break ;;
    esac
done

require_config_line() {
    if ! grep -Fqx "$1" "${CONFIG_PATH}"; then
        echo "[run_inference] ERROR: ${CONFIG_PATH} is missing expected line: $1" >&2
        return 1
    fi
}

validate_configured_arm() {
    local arm="$1" short ros_prefix command_topic
    case "${arm}" in
        left)
            short=l
            ros_prefix=follower_l
            command_topic=/follower_l_forward_position_controller/commands
            ;;
        right)
            short=r
            ros_prefix=follower_r
            command_topic=/follower_r_forward_position_controller/commands
            ;;
    esac
    require_config_line "    ${short}: ${arm}"
    require_config_line "  ${arm}:"
    require_config_line "    ros_prefix: \"${ros_prefix}\""
    require_config_line "    command_topic: \"${command_topic}\""
}

if [[ "${START_REQUESTED}" == true && "${FAKE_HARDWARE}" == false && "${ECHO_TOPIC_ONLY_REQUESTED}" == false ]]; then
    CONFIG_PATH="${CONFIG_FILE:-${REPO_ROOT}/configs/lerobot_control/inference_default.yaml}"
    if [[ "${CONFIG_PATH}" != /* ]]; then
        CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH#./}"
    fi
    case "${INFERENCE_ARM:-}" in
        left)
            validate_configured_arm left
            if grep -Fqx "    r: right" "${CONFIG_PATH}" || grep -Fqx "  right:" "${CONFIG_PATH}"; then
                echo "[run_inference] ERROR: config also routes the right arm" >&2
                exit 2
            fi
            ;;
        right)
            validate_configured_arm right
            if grep -Fqx "    l: left" "${CONFIG_PATH}" || grep -Fqx "  left:" "${CONFIG_PATH}"; then
                echo "[run_inference] ERROR: config also routes the left arm" >&2
                exit 2
            fi
            ;;
        bimanual)
            validate_configured_arm left
            validate_configured_arm right
            ;;
        *)
            echo "[run_inference] ERROR: set INFERENCE_ARM to left, right, or bimanual before real-robot startup" >&2
            exit 2
            ;;
    esac
    echo "[run_inference] arm preflight passed: ${INFERENCE_ARM} (${CONFIG_PATH})"
    if [[ "${PREFLIGHT_ONLY}" == true ]]; then
        exit 0
    fi
fi

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
        _detected=$(python3 -c "import json; d=json.load(open('${_anvil_config}')); print(d.get('action_type','absolute'))" 2>/dev/null || true)
        if [[ -n "${_detected}" ]]; then
            export ACTION_TYPE="${_detected}"
            echo "[run_inference] ACTION_TYPE=${ACTION_TYPE} (auto-detected from $(basename $(dirname ${_anvil_config})))"
        fi
    fi
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
fi

exit "$COMPOSE_EXIT"
