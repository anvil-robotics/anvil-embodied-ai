#!/usr/bin/env bash
#
# lint.sh - Run code linting and formatting checks
#
# This script runs ruff for both linting and format checking.
# Exit code is non-zero if any check fails.
#
# Usage: ./lint.sh [--fix]
#
# Options:
#   --fix    Auto-fix linting issues and format code
#

set -e

# Get the repository root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Track overall exit status
EXIT_CODE=0

# Check for --fix flag
FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
fi

echo "=========================================="
echo "  Anvil Embodied AI - Lint & Format Check"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Run ruff check (linting)
# -----------------------------------------------------------------------------
echo "[1/2] Running ruff check (linting)..."
echo ""

if [ "$FIX_MODE" = true ]; then
    if uv run ruff check . --fix; then
        echo ""
        echo "  ✓ Linting passed (with auto-fix applied)"
    else
        echo ""
        echo "  ✗ Linting issues found (some may have been auto-fixed)"
        EXIT_CODE=1
    fi
else
    if uv run ruff check .; then
        echo ""
        echo "  ✓ Linting passed"
    else
        echo ""
        echo "  ✗ Linting issues found. Run './scripts/lint.sh --fix' to auto-fix."
        EXIT_CODE=1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Step 2: Run ruff format (formatting)
# -----------------------------------------------------------------------------
echo "[2/2] Running ruff format check..."
echo ""

if [ "$FIX_MODE" = true ]; then
    if uv run ruff format .; then
        echo ""
        echo "  ✓ Formatting applied"
    else
        echo ""
        echo "  ✗ Formatting failed"
        EXIT_CODE=1
    fi
else
    if uv run ruff format --check .; then
        echo ""
        echo "  ✓ Formatting check passed"
    else
        echo ""
        echo "  ✗ Formatting issues found. Run './scripts/lint.sh --fix' to auto-format."
        EXIT_CODE=1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  All checks passed!"
else
    echo "  Some checks failed. See above for details."
fi
echo "=========================================="

exit $EXIT_CODE
