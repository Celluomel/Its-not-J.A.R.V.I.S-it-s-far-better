#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x "venv/bin/python" ]]; then
  BODY_PYTHON="venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  BODY_PYTHON=".venv/bin/python"
else
  echo "[ERROR] No Python environment found. Run setup.sh first."
  exit 1
fi
echo "Starting standalone Lumina Body Runtime..."
exec "$BODY_PYTHON" -m cognition.body_runtime
