#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec uvicorn src.app:app --reload --host 0.0.0.0 --port 8000


