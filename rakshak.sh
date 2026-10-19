#!/usr/bin/env zsh
# rakshak.sh — One-click launcher for Rakshak SAR simulation
# Usage: ./rakshak.sh [any args forwarded to app.py, e.g. --fullscreen --no-priority]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Prefer venv python, fall back to python3
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
    PYTHON="$VENV_PY"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: No Python 3 found. Install it or run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "▶ Launching Rakshak with $PYTHON"
cd "$SCRIPT_DIR"
"$PYTHON" app.py "$@"
