#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x "body_venv/bin/python" ]]; then
  echo "[*] Creating the independent Body virtual environment..."
  python3 -m venv body_venv
fi
BODY_PYTHON="body_venv/bin/python"
"$BODY_PYTHON" -m pip install --upgrade pip -q
"$BODY_PYTHON" -m pip install -r body_requirements.txt -q
echo "Starting standalone PandoraBOX Body Runtime..."
exec "$BODY_PYTHON" -m body_runtime_host
