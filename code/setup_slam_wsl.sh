#!/usr/bin/env bash
# One-time setup for the PS5 world view (SLAM with NVIDIA ViPE) inside WSL2 Ubuntu on Windows.
# Run from this code/ folder inside Ubuntu:   bash setup_slam_wsl.sh
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential ffmpeg git wget

# CUDA compiler (nvcc) for WSL; the GPU driver itself comes from the normal Windows NVIDIA driver
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
  sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

python3 -m venv ~/ps5env
source ~/ps5env/bin/activate
pip install -U pip setuptools wheel ninja
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128      # RTX 50-series needs cu128
pip install numpy scipy opencv-python pandas pyyaml rerun-sdk

# compile ViPE's CUDA kernels only for this GPU (much faster)
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; c=torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')")
echo "GPU arch: $TORCH_CUDA_ARCH_LIST"
pip install --no-build-isolation nvidia-vipe
pip install git+https://github.com/cvg/GeoCalib || echo "GeoCalib not installed: gravity will come from the map only"

python -c "import torch, vipe; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('ViPE OK')"
python tests/test_world_synthetic.py | tail -1
echo "Setup done. Next time:  source ~/ps5env/bin/activate"
