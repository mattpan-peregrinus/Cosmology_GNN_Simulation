#!/bin/bash

# Setup script for conda environment
conda create -n test_env python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate test_env

# Install PyTorch with CUDA 12.6 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Install torch-geometric and dependencies
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.6.1 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# Install remaining Python packages
pip install aiohappyeyeballs aiohttp aiosignal attrs brotli certifi charset-normalizer \
  contourpy cycler filelock fonttools frozenlist fsspec h5py idna jinja2 kiwisolver \
  markupsafe matplotlib mkl-service mkl_fft mkl_random mpmath multidict networkx \
  numpy nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 packaging pillow propcache psutil \
  pyparsing PyQt6 PyQt6_sip python-dateutil requests scipy sip six sympy tornado tqdm \
  triton typing_extensions unicodedata2 urllib3 yarl
