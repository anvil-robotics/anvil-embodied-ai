#!/usr/bin/env bash
#
# build-docker.sh - Build the inference Docker image
#
# This script builds the LeRobot inference Docker image using the
# Dockerfile at docker/inference/Dockerfile.
#
# Usage:
#   ./build-docker.sh              # Build with default tag (lerobot-inference:latest)
#   ./build-docker.sh v1.0.0       # Build with custom tag (lerobot-inference:v1.0.0)
#   ./build-docker.sh --help       # Show this help message
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
IMAGE_NAME="lerobot-inference"
DEFAULT_TAG="latest"
DOCKERFILE_PATH="${REPO_ROOT}/docker/inference/Dockerfile"

# Display usage
show_usage() {
    echo "Usage: ./build-docker.sh [TAG] [OPTIONS]"
    echo ""
    echo "Build the LeRobot inference Docker image."
    echo ""
    echo "Arguments:"
    echo "  TAG              Docker image tag (default: latest)"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo "  --no-cache       Build without using cache"
    echo ""
    echo "Examples:"
    echo "  ./build-docker.sh                # Build lerobot-inference:latest"
    echo "  ./build-docker.sh v1.0.0         # Build lerobot-inference:v1.0.0"
    echo "  ./build-docker.sh dev --no-cache # Build without cache"
    echo ""
}

# Parse arguments
TAG="${DEFAULT_TAG}"
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -*)
            echo "Error: Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            TAG="$1"
            shift
            ;;
    esac
done

# Verify Dockerfile exists
if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "Error: Dockerfile not found at ${DOCKERFILE_PATH}"
    exit 1
fi

echo "=========================================="
echo "  Anvil Embodied AI - Docker Build"
echo "=========================================="
echo ""
echo "  Image:      ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile: ${DOCKERFILE_PATH}"
echo "  Context:    ${REPO_ROOT}"
echo ""

# Build the Docker image
echo "Building Docker image..."
echo ""

cd "${REPO_ROOT}"

docker build \
    ${NO_CACHE} \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${DOCKERFILE_PATH}" \
    .

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo ""
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run the container:"
echo "  docker run --rm -it ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run with GPU support:"
echo "  docker run --rm -it --gpus all ${IMAGE_NAME}:${TAG}"
echo ""


# This worked for me: docker compose -f docker/inference/docker-compose.yml build --no-cache
