#!/usr/bin/env bash
#
# test.sh - Run pytest for the workspace packages
#
# This script runs pytest on the packages/ directory with verbose output.
# Additional arguments are passed through to pytest.
#
# Usage:
#   ./test.sh                    # Run all tests
#   ./test.sh -k "test_foo"      # Run tests matching "test_foo"
#   ./test.sh --cov              # Run with coverage
#   ./test.sh packages/mcap_converter  # Run tests for specific package
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Display usage if --help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: ./test.sh [pytest-options]"
    echo ""
    echo "Run pytest for the anvil-embodied-ai workspace packages."
    echo ""
    echo "Examples:"
    echo "  ./test.sh                         # Run all tests"
    echo "  ./test.sh -k 'test_converter'     # Run tests matching pattern"
    echo "  ./test.sh --cov                   # Run with coverage report"
    echo "  ./test.sh -x                      # Stop on first failure"
    echo "  ./test.sh packages/mcap_converter # Test specific package"
    echo ""
    echo "All arguments are passed directly to pytest."
    exit 0
fi

echo "=========================================="
echo "  Anvil Embodied AI - Test Suite"
echo "=========================================="
echo ""

# Default test path if no arguments provided that look like paths
DEFAULT_PATH="packages/"

# Check if any argument looks like a path (contains / or starts with packages)
HAS_PATH_ARG=false
for arg in "$@"; do
    if [[ "$arg" == *"/"* ]] || [[ "$arg" == "packages"* ]]; then
        HAS_PATH_ARG=true
        break
    fi
done

# Run pytest with arguments
# If no path-like argument is given, use default packages/ path
if [ $# -eq 0 ]; then
    echo "Running: uv run pytest ${DEFAULT_PATH} -v"
    echo ""
    uv run pytest "${DEFAULT_PATH}" -v
elif [ "$HAS_PATH_ARG" = true ]; then
    echo "Running: uv run pytest $*"
    echo ""
    uv run pytest "$@"
else
    echo "Running: uv run pytest ${DEFAULT_PATH} -v $*"
    echo ""
    uv run pytest "${DEFAULT_PATH}" -v "$@"
fi

echo ""
echo "=========================================="
echo "  Tests completed!"
echo "=========================================="
