FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"

# PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 marks the system Python as
# "externally managed" (PEP 668) and refuses bare `pip install`. There's no
# real "system" to protect inside an isolated container, so this is safe here.

# TORCH_CUDA_ARCH_LIST covers 3090/3090Ti (8.6), 4090 (8.9), and 5090 (12.0,
# Blackwell) -- so this same image works if you add cheaper GPU workers later.
# 5090 support specifically REQUIRES this; older prebuilt wheels/extensions
# compiled without sm_120 fail at runtime with "no kernel image is available
# for execution on the device".

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip python3-venv \
        git wget curl build-essential ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# PyTorch built against CUDA 12.8 -- the only wheel line with Blackwell (5090)
# kernels. Do not swap this for a cu121/cu124 index url.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# --- Hunyuan3D-2 (shape + texture pipelines) ---
RUN git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.git /app/Hunyuan3D-2
WORKDIR /app/Hunyuan3D-2
RUN pip install --no-cache-dir -r requirements.txt

# Compiled CUDA extensions the texture pipeline needs (rasterizer + renderer).
# TORCH_CUDA_ARCH_LIST above makes these build with sm_120 (5090) kernels included.
RUN pip install --no-cache-dir -e hy3dgen/texgen/custom_rasterizer \
    && pip install --no-cache-dir ./hy3dgen/texgen/differentiable_renderer

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow \
    diffusers accelerate huggingface_hub

# Model weights are NOT baked in at build time -- tens of GB of checkpoints
# blew the disk budget on every build environment tried (local Docker
# Desktop, GitHub Actions runners). server.py's from_pretrained() calls
# download straight from Hugging Face into HF_HOME on first container start
# instead. Trade-off: first boot after deploy takes a few extra minutes;
# every boot after that is fast since HF_HOME should live on a persistent
# volume, not the ephemeral container filesystem.
COPY server.py /app/server.py

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
