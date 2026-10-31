#!/usr/bin/env bash
# SpectatorAI dashboard: one command, works offline once set up.
# Windows (Git Bash) delegates to run.bat; macOS/Linux sets up a venv the same way.
set -euo pipefail
cd "$(dirname "$0")"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) exec cmd //c run.bat ;;
esac

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[run] first run: setting up (needs internet once)"
  python3 -m venv .venv
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  $PY -m pip install -q -r backend/requirements.txt
  $PY -m pip install -q --no-deps git+https://github.com/warmshao/WiLoR-mini
  $PY -m pip install -q --no-build-isolation --no-deps git+https://github.com/mattloper/chumpy
  $PY code/tools/patch_chumpy.py
  $PY -c "import sys; sys.path.insert(0,'code'); from ps4v4.models.wilor_loader import load_wilor; load_wilor('cpu')"
  $PY -c "from transformers import CLIPModel, CLIPProcessor as P; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); P.from_pretrained('openai/clip-vit-base-patch32')"
fi
if [ ! -f frontend/dist/index.html ]; then
  (cd frontend && npm ci && npm run build)
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PORT=${PORT:-8000}
echo "[run] SpectatorAI on http://127.0.0.1:$PORT  (Ctrl+C to stop)"
( sleep 2; (xdg-open "http://127.0.0.1:$PORT" || open "http://127.0.0.1:$PORT") >/dev/null 2>&1 || true ) &
exec $PY -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT"
