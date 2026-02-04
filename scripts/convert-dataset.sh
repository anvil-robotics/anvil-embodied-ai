#!/usr/bin/env bash
#
# convert-dataset.sh - Convert MCAP recordings to LeRobot dataset format
#
# This script demonstrates how to convert MCAP files (ROS2 bag format)
# to the LeRobot HuggingFace dataset format using the mcap_converter package.
#
# Usage:
#   ./convert-dataset.sh <input-dir> <output-dir> [config]
#
# Arguments:
#   input-dir   Directory containing MCAP files to convert
#   output-dir  Output directory for the LeRobot dataset
#   config      (Optional) Path to converter config YAML file
#               Default: configs/mcap_converter/openarm_bimanual.yaml
#
# Example:
#   ./convert-dataset.sh ./data/raw/episode_001 ./data/datasets/my_dataset
#   ./convert-dataset.sh ./recordings ./datasets/custom configs/custom.yaml
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
DEFAULT_CONFIG="${REPO_ROOT}/configs/mcap_converter/openarm_bimanual.yaml"

# Display usage
show_usage() {
    echo "Usage: ./convert-dataset.sh <input-dir> <output-dir> [config]"
    echo ""
    echo "Convert MCAP recordings to LeRobot dataset format."
    echo ""
    echo "Arguments:"
    echo "  input-dir    Directory containing MCAP files to convert"
    echo "  output-dir   Output directory for the LeRobot dataset"
    echo "  config       (Optional) Path to converter config YAML file"
    echo "               Default: configs/mcap_converter/openarm_bimanual.yaml"
    echo ""
    echo "Options:"
    echo "  --help, -h   Show this help message"
    echo "  --dry-run    Show what would be done without executing"
    echo ""
    echo "Examples:"
    echo "  ./convert-dataset.sh ./data/raw/session_01 ./data/datasets/my_dataset"
    echo "  ./convert-dataset.sh ./recordings ./datasets/custom ./configs/custom.yaml"
    echo ""
    echo "The converter will:"
    echo "  1. Read MCAP files from the input directory"
    echo "  2. Extract images, joint states, and actions based on config"
    echo "  3. Create a LeRobot-compatible HuggingFace dataset"
    echo ""
    echo "Note: Ensure the mcap_converter package is installed:"
    echo "  uv sync"
    echo ""
}

# Parse arguments
DRY_RUN=false

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

case $1 in
    --help|-h)
        show_usage
        exit 0
        ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
esac

# Validate required arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments."
    echo ""
    show_usage
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CONFIG="${3:-${DEFAULT_CONFIG}}"

# Convert to absolute paths if relative
if [[ ! "$INPUT_DIR" = /* ]]; then
    INPUT_DIR="$(pwd)/${INPUT_DIR}"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$(pwd)/${OUTPUT_DIR}"
fi
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG="$(pwd)/${CONFIG}"
fi

echo "=========================================="
echo "  Anvil Embodied AI - Dataset Conversion"
echo "=========================================="
echo ""
echo "  Input directory:  ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Config file:      ${CONFIG}"
echo ""

# Validate input directory
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Check for MCAP files
MCAP_COUNT=$(find "${INPUT_DIR}" -name "*.mcap" 2>/dev/null | wc -l)
if [ "${MCAP_COUNT}" -eq 0 ]; then
    echo "Warning: No .mcap files found in ${INPUT_DIR}"
    echo "Looking for MCAP files recursively..."
    MCAP_COUNT=$(find "${INPUT_DIR}" -name "*.mcap" -type f 2>/dev/null | wc -l)
    if [ "${MCAP_COUNT}" -eq 0 ]; then
        echo "Error: No .mcap files found in input directory or subdirectories."
        exit 1
    fi
fi
echo "  Found ${MCAP_COUNT} MCAP file(s)"

# Check config file
if [ ! -f "${CONFIG}" ]; then
    echo ""
    echo "Warning: Config file not found: ${CONFIG}"
    echo ""
    echo "Please create a config file or use an existing one."
    echo "Example config structure:"
    echo ""
    echo "  # configs/mcap_converter/openarm_bimanual.yaml"
    echo "  robot_type: openarm_bimanual"
    echo "  fps: 30"
    echo "  "
    echo "  topics:"
    echo "    images:"
    echo "      - /camera/front/image_raw"
    echo "      - /camera/wrist/image_raw"
    echo "    joint_states: /joint_states"
    echo "    actions: /robot/commands"
    echo "  "
    echo "  output:"
    echo "    format: lerobot"
    echo "    compress_images: true"
    echo ""
    exit 1
fi

echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    echo ""
    echo "  cd ${REPO_ROOT}"
    echo "  uv run python -m mcap_converter.cli convert \\"
    echo "      --input ${INPUT_DIR} \\"
    echo "      --output ${OUTPUT_DIR} \\"
    echo "      --config ${CONFIG}"
    echo ""
    exit 0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the conversion
echo "Starting conversion..."
echo ""

cd "${REPO_ROOT}"

# Run the mcap_converter CLI
# Note: Adjust the module path based on actual package structure
uv run python -m mcap_converter.cli convert \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --config "${CONFIG}"

echo ""
echo "=========================================="
echo "  Conversion complete!"
echo "=========================================="
echo ""
echo "  Output saved to: ${OUTPUT_DIR}"
echo ""
echo "To load the dataset in Python:"
echo ""
echo "  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset"
echo "  dataset = LeRobotDataset('${OUTPUT_DIR}')"
echo ""
