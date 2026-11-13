#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run tools/macos/setup.sh first." >&2
  exit 1
fi
PORT="${FLOWHMR_PORT:-8080}"
URL="http://127.0.0.1:${PORT}/"

(
  for _ in {1..120}; do
    if curl -fsS "$URL" >/dev/null 2>&1; then
      open "$URL"
      exit 0
    fi
    sleep 1
  done
) &

exec .venv/bin/python app.py --port "$PORT" "$@"
