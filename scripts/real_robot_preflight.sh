#!/usr/bin/env bash
# Prepare and check this checkout before first real OpenArm data collection.
#
# Usage:
#   ./scripts/real_robot_preflight.sh [--full] [--run-fake-hardware] [--no-local-files]
#
# Default mode is intentionally lightweight: it checks host tools, config files,
# Docker compose rendering, and creates ignored local directories/files that are
# useful on first robot day. Use --full when you are ready to let uv resolve and
# install the Python workspace dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL=false
RUN_FAKE_HARDWARE=false
WRITE_LOCAL_FILES=true

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            FULL=true
            shift
            ;;
        --run-fake-hardware)
            RUN_FAKE_HARDWARE=true
            shift
            ;;
        --no-local-files)
            WRITE_LOCAL_FILES=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$REPO_ROOT"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

ok() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "[OK] $*"
}

warn() {
    WARN_COUNT=$((WARN_COUNT + 1))
    echo "[WARN] $*"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[FAIL] $*"
}

check_cmd() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        ok "$name found: $(command -v "$name")"
    else
        fail "$name is not installed or not on PATH"
    fi
}

check_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        ok "found $path"
    else
        fail "missing $path"
    fi
}

check_dir() {
    local path="$1"
    if [[ -d "$path" ]]; then
        ok "found $path/"
    else
        fail "missing $path/"
    fi
}

echo "[preflight] repo: $REPO_ROOT"

if [[ "$WRITE_LOCAL_FILES" == true ]]; then
    mkdir -p data/raw data/datasets model_zoo monitor_output debug_images eval_results
    ok "local output directories are present"

    if [[ ! -f .python-version ]]; then
        printf "3.12\n" > .python-version
        ok "created .python-version pinned to Python 3.12"
    else
        ok ".python-version already exists: $(tr -d '\n' < .python-version)"
    fi

    if [[ ! -f .env ]]; then
        cp .env.example .env
        ok "created .env from .env.example"
        warn "edit .env so ROS_DOMAIN_ID matches the robot/devbox before real DDS checks"
    else
        ok ".env already exists"
    fi
fi

check_cmd uv
check_cmd docker

if command -v uv >/dev/null 2>&1; then
    uv --version || warn "uv is installed but did not print a version"
    if uv python find 3.12 >/dev/null 2>&1; then
        ok "uv can find Python 3.12"
    else
        warn "uv cannot find Python 3.12 yet; run: uv python install 3.12"
    fi
fi

if command -v docker >/dev/null 2>&1; then
    docker --version || warn "docker is installed but did not print a version"
    docker compose version || warn "docker compose is unavailable"
    if docker info >/dev/null 2>&1; then
        ok "Docker daemon is reachable"
    else
        warn "Docker daemon is not reachable; start Docker before inference or ROS2 replay eval"
    fi
fi

check_file pyproject.toml
check_file .env.example
check_file scripts/run_inference.sh
check_file scripts/mcap_timing_report.py
check_file docker-compose.yml
check_file docker-compose.fake-hardware.yml
check_file docker-compose.eval.yml
check_file configs/lerobot_control/inference_default.yaml
check_file configs/lerobot_control/inference_eval.yaml
check_file configs/mcap_converter/openarm_bimanual.yaml
check_file configs/mcap_converter/openarm_bimanual_quest.yaml
check_file configs/mcap_converter/openarm_single_quest.yaml
check_file configs/mcap_converter/openarm_single_quest_afo.yaml
check_file configs/cyclonedds/gpu_pc.xml
check_file configs/cyclonedds/robot_pc.xml
check_file configs/cyclonedds/test_bridge.xml

check_dir packages/mcap_converter
check_dir packages/anvil_trainer
check_dir packages/anvil_eval
check_dir packages/anvil_eval_ros
check_dir ros2/src/lerobot_control

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if MODEL_PATH=/dev/null CONFIG_FILE=./configs/lerobot_control/inference_default.yaml HF_CACHE=.docker_empty_hf_cache docker compose -f docker-compose.yml config >/dev/null; then
        ok "production docker compose config renders"
    else
        fail "production docker compose config does not render"
    fi

    if CONFIG_FILE=./configs/lerobot_control/inference_default.yaml docker compose -f docker-compose.fake-hardware.yml --profile monitor config >/dev/null; then
        ok "fake-hardware docker compose config renders"
    else
        fail "fake-hardware docker compose config does not render"
    fi

    if MODEL_PATH=/dev/null \
        MCAP_ROOT=./data/raw \
        OUTPUT_DIR=./eval_results/preflight \
        EVAL_PLAN_FILE=/dev/null \
        CONFIG_FILE=./configs/lerobot_control/inference_eval.yaml \
        HF_CACHE=.docker_empty_hf_cache \
        docker compose -f docker-compose.eval.yml config >/dev/null; then
        ok "ROS2 eval docker compose config renders"
    else
        fail "ROS2 eval docker compose config does not render"
    fi
fi

if [[ "$FULL" == true ]]; then
    echo "[preflight] running uv sync --all-packages"
    if uv sync --all-packages; then
        ok "uv workspace sync completed"
    else
        fail "uv workspace sync failed"
    fi

    for cli in mcap-inspect mcap-convert dataset-validate mcap-to-video anvil-trainer anvil-eval anvil-eval-ros; do
        if uv run "$cli" --help >/dev/null; then
            ok "CLI works: $cli --help"
        else
            fail "CLI failed: $cli --help"
        fi
    done

    if uv run python scripts/mcap_timing_report.py --help >/dev/null; then
        ok "CLI works: scripts/mcap_timing_report.py --help"
    else
        fail "CLI failed: scripts/mcap_timing_report.py --help"
    fi
else
    warn "skipped uv dependency install; run this script with --full before training/eval"
fi

if [[ "$RUN_FAKE_HARDWARE" == true ]]; then
    echo "[preflight] starting fake-hardware monitor smoke test for up to 90 seconds"
    if ! command -v timeout >/dev/null 2>&1; then
        warn "timeout command is unavailable; skipping --run-fake-hardware smoke test"
    elif timeout 90s ./scripts/run_inference.sh --fake-hardware --monitor up --build; then
        ok "fake-hardware monitor smoke test completed"
    else
        warn "fake-hardware monitor smoke test did not complete cleanly; inspect Docker logs"
    fi
fi

echo
echo "[preflight] summary: ${PASS_COUNT} ok, ${WARN_COUNT} warn, ${FAIL_COUNT} fail"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi
