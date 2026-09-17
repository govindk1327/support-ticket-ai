#!/usr/bin/env bash
# Start the API and the UI together.
#
#   ./start.sh
#
# Resolves its own directory, so it works from any working directory.
# Ctrl-C stops both processes.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: $PYTHON not found"; exit 1; }

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "error: Python 3.10 or newer is required (found: $("$PYTHON" --version 2>&1))"
  echo "       recreate the venv with a newer interpreter, e.g. python3.12 -m venv .venv"
  exit 1
fi

if [ ! -f "data/support_tickets.csv" ]; then
  echo "error: data/support_tickets.csv is missing"
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "note: no .env found. /health, /meta and /anomalies will work;"
  echo "      /query needs a provider. Copy .env.example to .env to set one up."
fi

cleanup() {
  echo ""
  echo "Shutting down..."
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "API -> http://${API_HOST}:${API_PORT}    (docs at /docs)"
"$PYTHON" -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" &

# Give the API a moment so the UI's first health call succeeds.
sleep 3

echo "UI  -> http://localhost:${UI_PORT}"
API_URL="http://${API_HOST}:${API_PORT}" \
  "$PYTHON" -m streamlit run ui/app.py \
    --server.port "$UI_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false &

wait
