# One-time setup of PS4 v4 on Windows (RTX 50-series GPU).
# Run from the ps4v4 folder in PowerShell:   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
# Needs: Python 3.10-3.12 on PATH (py launcher), git, and (optional) ffmpeg for the final video encode.

$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) {
    try { py -3.11 -m venv .venv } catch { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"

function Run([string[]]$cmd) {
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: python $($cmd -join ' ')" }
}

Run @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")   # setuptools: needed to build chumpy

# PyTorch built for CUDA 12.8 (required for RTX 50-series), installed first so nothing replaces it
Run @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")

# core + WiLoR dependencies
Run @("-m", "pip", "install", "numpy", "scipy", "opencv-python", "pandas", "pyyaml", "ultralytics", "smplx==0.1.28",
      "timm", "einops", "roma", "scikit-image", "huggingface_hub")
Run @("-m", "pip", "install", "--no-deps", "git+https://github.com/warmshao/WiLoR-mini")
Run @("-m", "pip", "install", "--no-build-isolation", "--no-deps", "git+https://github.com/mattloper/chumpy")
Run @("tools\patch_chumpy.py")
Run @("-c", "import chumpy; print('chumpy OK')")

# optional: GeoCalib (better focal length). The pipeline falls back to a FOV guess if this fails.
& $py -m pip install git+https://github.com/cvg/GeoCalib
if ($LASTEXITCODE -ne 0) { Write-Warning "GeoCalib not installed; using the FOV fallback" }

Run @("-c", "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')")
Write-Host "Setup done. Run:  .\.venv\Scripts\python.exe run.py --clip <video>"
