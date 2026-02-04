#!/usr/bin/env bash
#
# setup-dev.sh - Development environment setup script
#
# This script sets up the development environment for anvil-embodied-ai:
# - Checks for and installs uv if needed
# - Creates virtual environment and installs workspace packages
# - Installs pre-commit hooks
#
# Usage: ./setup-dev.sh
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================="
echo "  Anvil Embodied AI - Development Setup"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check and install uv
# -----------------------------------------------------------------------------
echo "[1/3] Checking for uv package manager..."

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo "  ✓ uv is already installed: ${UV_VERSION}"
else
    echo "  → uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for the current session
    export PATH="${HOME}/.local/bin:${PATH}"

    if command -v uv &> /dev/null; then
        echo "  ✓ uv installed successfully: $(uv --version)"
    else
        echo "  ✗ Failed to install uv. Please install manually:"
        echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: Create venv and install workspace packages
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Setting up virtual environment and installing packages..."

cd "${REPO_ROOT}"
echo "  → Running 'uv sync' in ${REPO_ROOT}..."
uv sync

echo "  ✓ Virtual environment created and packages installed"

# -----------------------------------------------------------------------------
# Step 3: Install pre-commit hooks
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Installing pre-commit hooks..."

if [ -f "${REPO_ROOT}/.pre-commit-config.yaml" ]; then
    uv run pre-commit install
    echo "  ✓ Pre-commit hooks installed"
else
    echo "  ⚠ No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# -----------------------------------------------------------------------------
# Success message
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Development environment setup complete!"
echo "=========================================="
echo ""
echo "You can now:"
echo "  - Run tests:   ./scripts/test.sh"
echo "  - Run linter:  ./scripts/lint.sh"
echo "  - Activate venv: source .venv/bin/activate"
echo ""
