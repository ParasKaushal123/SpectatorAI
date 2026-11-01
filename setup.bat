@echo off
rem One-time setup (needs internet). After this, run.bat works offline.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo [setup] creating Python 3.12 venv
  py -3.12 -m venv .venv || python -m venv .venv || goto :fail
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install -q --upgrade pip || goto :fail
echo [setup] CPU PyTorch
%PY% -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :fail
echo [setup] pipeline + API packages
%PY% -m pip install -q -r dashboard\backend\requirements.txt || goto :fail
%PY% -m pip install -q --no-deps git+https://github.com/warmshao/WiLoR-mini || goto :fail
%PY% -m pip install -q --no-build-isolation --no-deps git+https://github.com/mattloper/chumpy || goto :fail
%PY% code\tools\patch_chumpy.py
echo [setup] downloading model weights (WiLoR, CLIP) once
%PY% -c "import sys; sys.path.insert(0,'code'); from ps4v4.models.wilor_loader import load_wilor; load_wilor('cpu'); print('WiLoR OK')" || goto :fail
%PY% -c "from transformers import CLIPModel, CLIPProcessor as P; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); P.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP OK')" || goto :fail

echo [setup] frontend
pushd dashboard\frontend
call npm ci || (popd & goto :fail)
call npm run build || (popd & goto :fail)
popd
echo [setup] done. Start with run.bat
exit /b 0

:fail
echo [setup] FAILED - see the message above
exit /b 1
