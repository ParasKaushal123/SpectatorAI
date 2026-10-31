@echo off
rem SpectatorAI dashboard: one command, works offline once setup.bat has run.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo [run] first run: setting up, needs internet once
  call setup.bat || exit /b 1
)
if not exist frontend\dist\index.html (
  pushd frontend & call npm run build & popd
)

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PORT=8000
echo [run] SpectatorAI on http://127.0.0.1:%PORT%  - Ctrl+C to stop
start "" http://127.0.0.1:%PORT%
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port %PORT%
