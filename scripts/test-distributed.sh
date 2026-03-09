#!/usr/bin/env bash
#
# test-distributed.sh — Validate CycloneDDS distributed inference (GPU PC ↔ Robot PC)
#
# Run this on the GPU PC after the Robot PC's anvil-workcell stack is up.
# It builds/starts the inference container and checks CycloneDDS can discover
# the Robot PC's ROS2 nodes and topics.
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker)
#   - .env configured (copy from .env.example)
#   - Robot PC running with matching ROS_DOMAIN_ID and CycloneDDS peer config
#
# Usage:
#   ./scripts/test-distributed.sh              # full build + test
#   ./scripts/test-distributed.sh --no-build   # skip build, test running container
#   ./scripts/test-distributed.sh --diag-only  # diagnostics on running container only
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
ENV_FILE="${REPO_ROOT}/.env"
CONTAINER_NAME="lerobot-inference"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; }
warn() { echo -e "${YELLOW}WARN${NC}: $1"; }
info() { echo -e "INFO: $1"; }

ERRORS=0

# ---------- Parse args ----------
DO_BUILD=true
DIAG_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --no-build)  DO_BUILD=false ;;
    --diag-only) DIAG_ONLY=true; DO_BUILD=false ;;
    -h|--help)
      echo "Usage: $0 [--no-build] [--diag-only]"
      echo ""
      echo "  --no-build   Skip docker compose build, test running container"
      echo "  --diag-only  Only run diagnostics on an already-running container"
      exit 0
      ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

echo "=========================================="
echo "  CycloneDDS Distributed Inference Test"
echo "=========================================="
echo ""

# ---------- Step 1: Prerequisites ----------
info "Checking prerequisites..."

if ! command -v docker &>/dev/null; then
  fail "docker not found"; exit 1
fi
pass "docker found"

if ! docker info 2>/dev/null | grep -qi "nvidia\|gpu"; then
  warn "NVIDIA runtime not detected in 'docker info' — GPU may not be available"
else
  pass "NVIDIA runtime detected"
fi

if [ ! -f "$ENV_FILE" ]; then
  fail ".env file not found at ${ENV_FILE}"
  info "Create it with: cp .env.example .env"
  exit 1
fi
pass ".env file exists"

# Source .env for display and validation
set -a
source "$ENV_FILE"
set +a

echo ""
info "Configuration from .env:"
info "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<not set>}"
info "  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<not set>}"
info "  CYCLONEDDS_URI=${CYCLONEDDS_URI:-<not set>}"
info "  MODEL_PATH=${MODEL_PATH:-<not set>}"
info "  CONFIG_FILE=${CONFIG_FILE:-<not set>}"
echo ""

if [ -z "${ROS_DOMAIN_ID:-}" ]; then
  fail "ROS_DOMAIN_ID not set in .env — required for distributed mode"
  exit 1
fi
pass "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"

if [ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]; then
  fail "RMW_IMPLEMENTATION should be 'rmw_cyclonedds_cpp', got '${RMW_IMPLEMENTATION:-}'"
  ERRORS=$((ERRORS + 1))
else
  pass "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
fi

# ---------- Step 2: Extract Robot PC IP from CycloneDDS config ----------
# The CYCLONEDDS_URI points to an XML with Peer addresses
CYCLONEDDS_HOST_PATH="${REPO_ROOT}/configs/cyclonedds/gpu_pc.xml"
if [ -f "$CYCLONEDDS_HOST_PATH" ]; then
  ROBOT_IP=$(grep -oP 'address="\K[^"]+' "$CYCLONEDDS_HOST_PATH" | head -1)
  if [ -n "$ROBOT_IP" ]; then
    info "Robot PC peer address from config: ${ROBOT_IP}"
    echo ""
    info "Pinging Robot PC (${ROBOT_IP})..."
    if ping -c 2 -W 2 "$ROBOT_IP" &>/dev/null; then
      pass "Robot PC ${ROBOT_IP} is reachable"
    else
      fail "Robot PC ${ROBOT_IP} is NOT reachable"
      ERRORS=$((ERRORS + 1))
    fi
  fi
else
  warn "CycloneDDS host config not found at ${CYCLONEDDS_HOST_PATH} — skipping ping check"
fi

echo ""

# ---------- Step 3: Build + Start (unless --no-build or --diag-only) ----------
if [ "$DIAG_ONLY" = false ]; then
  if [ "$DO_BUILD" = true ]; then
    info "Building inference container..."
    docker compose -f "$COMPOSE_FILE" build 2>&1 | tail -5
    echo ""
  fi

  info "Starting inference container (detached)..."
  docker compose -f "$COMPOSE_FILE" up -d 2>&1
  echo ""

  info "Waiting for container to be ready (15s)..."
  sleep 15
fi

# ---------- Step 4: Container diagnostics ----------
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  fail "Container '${CONTAINER_NAME}' is not running"
  info "Check logs with: docker compose -f ${COMPOSE_FILE} logs"
  exit 1
fi
pass "Container '${CONTAINER_NAME}' is running"
echo ""

info "--- Container environment ---"
docker exec "$CONTAINER_NAME" bash -c 'echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"; echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"; echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"; echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-<unset>}"'
echo ""

# Check RMW inside container
CONTAINER_RMW=$(docker exec "$CONTAINER_NAME" bash -c 'echo ${RMW_IMPLEMENTATION:-}')
if [ "$CONTAINER_RMW" = "rmw_cyclonedds_cpp" ]; then
  pass "Container RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
else
  fail "Container RMW_IMPLEMENTATION='${CONTAINER_RMW}' (expected rmw_cyclonedds_cpp)"
  ERRORS=$((ERRORS + 1))
fi

# Check CYCLONEDDS_URI file exists inside container
CONTAINER_URI=$(docker exec "$CONTAINER_NAME" bash -c 'echo ${CYCLONEDDS_URI:-}')
if [ -n "$CONTAINER_URI" ]; then
  URI_PATH="${CONTAINER_URI#file://}"
  if docker exec "$CONTAINER_NAME" test -f "$URI_PATH"; then
    pass "CycloneDDS config exists at ${URI_PATH} inside container"
  else
    fail "CycloneDDS config NOT found at ${URI_PATH} inside container"
    ERRORS=$((ERRORS + 1))
  fi
fi

# Check ROS_LOCALHOST_ONLY is NOT set (distributed mode)
CONTAINER_LOCALHOST=$(docker exec "$CONTAINER_NAME" bash -c 'echo ${ROS_LOCALHOST_ONLY:-}')
if [ -z "$CONTAINER_LOCALHOST" ] || [ "$CONTAINER_LOCALHOST" = "0" ]; then
  pass "ROS_LOCALHOST_ONLY is not set (distributed mode active)"
else
  fail "ROS_LOCALHOST_ONLY=${CONTAINER_LOCALHOST} — blocks cross-machine discovery"
  ERRORS=$((ERRORS + 1))
fi

echo ""

# ---------- Step 5: ROS2 discovery check ----------
info "--- Topic discovery (from inside container, 30s timeout) ---"
TOPICS=""
for i in $(seq 1 30); do
  TOPICS=$(docker exec "$CONTAINER_NAME" ros2 topic list 2>/dev/null || true)
  TOPIC_COUNT=$(echo "$TOPICS" | grep -c "^/" || true)
  if [ "$TOPIC_COUNT" -gt 2 ]; then
    break
  fi
  sleep 1
done

echo "$TOPICS"
echo ""

if echo "$TOPICS" | grep -q "/joint_states"; then
  pass "Discovered /joint_states from Robot PC"
else
  fail "Could not discover /joint_states — Robot PC may not be publishing or CycloneDDS peer config may be wrong"
  ERRORS=$((ERRORS + 1))
fi

info "--- Node discovery ---"
NODES=$(docker exec "$CONTAINER_NAME" ros2 node list 2>/dev/null || true)
echo "$NODES"
echo ""

# ---------- Step 6: Quick data check ----------
if echo "$TOPICS" | grep -q "/joint_states"; then
  info "--- Sampling /joint_states (1 message, 5s timeout) ---"
  if docker exec "$CONTAINER_NAME" timeout 5 ros2 topic echo /joint_states --once 2>/dev/null; then
    pass "Received live /joint_states data from Robot PC"
  else
    fail "Timed out waiting for /joint_states data"
    ERRORS=$((ERRORS + 1))
  fi
fi

# ---------- Summary ----------
echo ""
echo "=========================================="
if [ "$ERRORS" -eq 0 ]; then
  echo -e "  ${GREEN}All checks passed!${NC}"
  echo "  CycloneDDS distributed inference is working."
else
  echo -e "  ${RED}${ERRORS} check(s) failed.${NC}"
  echo "  Review the output above and check:"
  echo "    - Robot PC is running with matching ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
  echo "    - CycloneDDS peer IPs in configs/cyclonedds/gpu_pc.xml"
  echo "    - Network allows UDP multicast between GPU PC and Robot PC"
fi
echo "=========================================="
exit "$ERRORS"
