@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  call setup.bat
  if errorlevel 1 exit /b 1
)
if not exist dashboard\frontend\dist\index.html (
  pushd dashboard\frontend
  call npm ci
  if errorlevel 1 (popd & exit /b 1)
  call npm run build
  if errorlevel 1 (popd & exit /b 1)
  popd
)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PORT=8000
.venv\Scripts\python.exe -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:%PORT%/openapi.json',timeout=2)); assert d.get('info',{}).get('title')=='SpectatorAI'" >nul 2>&1
if not errorlevel 1 (
  start "" http://127.0.0.1:%PORT%
  exit /b 0
)
echo SpectatorAI: http://127.0.0.1:%PORT%
start "" http://127.0.0.1:%PORT%
.venv\Scripts\python.exe -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port %PORT%
