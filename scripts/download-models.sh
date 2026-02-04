#!/usr/bin/env bash
#
# download-models.sh - Download models from HuggingFace Hub
#
# This script downloads model checkpoints from the HuggingFace Hub
# to the local model_zoo directory.
#
# Usage:
#   ./download-models.sh <repo-id> <local-name>
#   ./download-models.sh lerobot/act_aloha_sim_transfer_cube_human act_aloha
#
# The model will be downloaded to: model_zoo/<local-name>/
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default model storage directory
MODEL_ZOO_DIR="${REPO_ROOT}/model_zoo"

# Display usage
show_usage() {
    echo "Usage: ./download-models.sh <repo-id> <local-name>"
    echo ""
    echo "Download models from HuggingFace Hub to the local model_zoo."
    echo ""
    echo "Arguments:"
    echo "  repo-id      HuggingFace repository ID (e.g., lerobot/act_aloha)"
    echo "  local-name   Local directory name for the model"
    echo ""
    echo "Options:"
    echo "  --help, -h   Show this help message"
    echo "  --list       List currently downloaded models"
    echo ""
    echo "Examples:"
    echo "  ./download-models.sh lerobot/act_aloha_sim_transfer_cube_human act_aloha"
    echo "  ./download-models.sh cadene/lerobot_pusht_diffusion pusht_diffusion"
    echo ""
    echo "Models will be saved to: ${MODEL_ZOO_DIR}/<local-name>/"
    echo ""
}

# List downloaded models
list_models() {
    echo "Downloaded models in ${MODEL_ZOO_DIR}:"
    echo ""
    if [ -d "${MODEL_ZOO_DIR}" ]; then
        for dir in "${MODEL_ZOO_DIR}"/*/; do
            if [ -d "$dir" ]; then
                model_name=$(basename "$dir")
                # Skip hidden directories and .gitkeep
                if [[ ! "$model_name" == .* ]]; then
                    echo "  - ${model_name}"
                fi
            fi
        done
    else
        echo "  (no models downloaded yet)"
    fi
    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

case $1 in
    --help|-h)
        show_usage
        exit 0
        ;;
    --list)
        list_models
        exit 0
        ;;
esac

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing arguments."
    echo ""
    show_usage
    exit 1
fi

REPO_ID="$1"
LOCAL_NAME="$2"
OUTPUT_DIR="${MODEL_ZOO_DIR}/${LOCAL_NAME}"

echo "=========================================="
echo "  Anvil Embodied AI - Model Download"
echo "=========================================="
echo ""
echo "  Repository:  ${REPO_ID}"
echo "  Local name:  ${LOCAL_NAME}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

# Create model_zoo directory if it doesn't exist
mkdir -p "${MODEL_ZOO_DIR}"

# Check if model already exists
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; then
    echo "Warning: Directory ${OUTPUT_DIR} already exists and is not empty."
    read -p "Do you want to overwrite? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "${OUTPUT_DIR}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Downloading model from HuggingFace Hub..."
echo ""

# Try using huggingface-cli first, fall back to Python if not available
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "${REPO_ID}" --local-dir "${OUTPUT_DIR}"
else
    echo "huggingface-cli not found, using Python huggingface_hub..."
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    local_dir='${OUTPUT_DIR}',
    local_dir_use_symlinks=False
)
print('Download complete!')
"
fi

echo ""
echo "=========================================="
echo "  Download complete!"
echo "=========================================="
echo ""
echo "  Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To use this model, reference it in your config as:"
echo "  model_path: model_zoo/${LOCAL_NAME}"
echo ""
