#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Backend
pkill -f "uvicorn src.app:app" || true
nohup uvicorn src.app:app --host 0.0.0.0 --port 8000 > uvicorn.out 2>&1 &
sleep 1
echo "HEALTH: $(curl -sS http://127.0.0.1:8000/health || true)"

# Frontend (5174)
pushd frontend >/dev/null
pkill -f vite || true
npm install --no-audit --no-fund
nohup npm run dev -- --host 0.0.0.0 --port 5174 > ../vite.out 2>&1 &
popd >/dev/null
sleep 2

echo "FRONT: $(curl -sSI http://127.0.0.1:5174/ 2>/dev/null | head -n 1)"
echo "Started. Backend http://localhost:8000  Frontend http://localhost:5174"


