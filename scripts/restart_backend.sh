#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pkill -f "uvicorn src.app:app" || true
mkdir -p logs
nohup uvicorn src.app:app --host 0.0.0.0 --port 8000 > uvicorn.out 2>&1 &

# Wait until backend is up (max 30s)
for i in $(seq 1 30); do
  if curl -sS http://127.0.0.1:8000/health >/dev/null; then
    echo "HEALTH: {\"status\":\"ok\"}"
    exit 0
  fi
  sleep 1
done

echo "Backend failed to start. Log tail:" >&2
tail -n 120 uvicorn.out || true
exit 1


