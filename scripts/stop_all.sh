#!/usr/bin/env bash
set -euo pipefail

pkill -f "uvicorn src.app:app" || true
pkill -f vite || true
echo "Stopped backend/frontend."


