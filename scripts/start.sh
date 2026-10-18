#!/bin/sh
# After completing README setup, start the local API (Ctrl+C for graceful shutdown).
set -eu
cd "$(dirname "$0")/.."
if [ ! -f ".venv/bin/python" ]; then
  echo 'Python environment missing. Follow README.md Quick start first.' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-graceful-shutdown 30
