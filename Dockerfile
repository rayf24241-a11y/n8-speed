FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    U2NET_HOME=/app/hf-cache/u2net

# U2NET_HOME: confirmed live -- the rembg package (hy3dgen.rembg's
# BackgroundRemover wraps it) defaults to ~/.u2net, which is NOT on the
# persistent volume the way HF_HOME is. Every single pod restart was
# re-downloading its 176MB u2net.onnx model from GitHub releases from
# scratch, and GitHub's release-asset CDN throughput is inconsistent --
# one real restart took 7+ minutes on this alone, dropping to under
# 100KB/s at points, purely from bad luck on an external host neither we
# nor RunPod control. Redirecting it onto the same persistent volume as
# everything else means this only ever happens once, not on every restart.

# PIP_BREAK_SYSTEM_PACKAGES: harmless here (Ubuntu 22.04's pip predates the
# PEP 668 "externally managed" restriction), kept in case that ever changes.

# CUDA 12.4, not 12.8: originally targeted the RTX 5090 (Blackwell, needs
# cuda>=12.8), but we're actually deploying on RTX 4090s on RunPod, and a
# 12.8-based image got flat-out rejected by real hosts there --
# "nvidia-container-cli: requirement error: unsatisfied condition:
# cuda>=12.8, please update your driver" -- because plenty of hosts in the
# fleet only have drivers supporting up to 12.4. 12.4 covers 4090 (8.9) and
# 3090/3090Ti (8.6) fully and runs on far more real hosts. If a 5090 worker
# gets added later, that's a separate image built with a 12.8 base + arch
# 12.0 added back in -- don't try to make one image serve both, that's what
# just broke.

# Ubuntu 22.04, not 24.04: nvidia/cuda has no 12.4.x tag published for
# 24.04 at all (confirmed against Docker Hub's tag list -- 24.04 starts at
# 12.6). 22.04 ships Python 3.10 by default, which is why this uses plain
# `python3` packages instead of a 3.12 PPA -- nothing here needs 3.12
# specifically.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        git wget curl build-essential ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- Hunyuan3D-2 (shape + texture pipelines) ---
RUN git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.git /app/Hunyuan3D-2
WORKDIR /app/Hunyuan3D-2

# The paint pipeline's own code calls DiffusionPipeline.from_pretrained()
# internally without trust_remote_code=True, so diffusers refuses to
# execute hunyuanpaint's custom pipeline.py -- confirmed on a real deploy:
# "ValueError: ... contains custom code in pipeline.py which must be
# executed ... Pass trust_remote_code=True".
#
# Inserting the kwarg right after "from_pretrained(" (an earlier version of
# this patch) broke on a real deploy: that particular call's first argument
# is positional, so prepending a keyword arg produced
# "SyntaxError: positional argument follows keyword argument". Appending
# right before the call's own closing ")" instead is safe regardless of
# what positional args come first, since a trailing keyword arg is always
# valid Python. Matches multiview_utils.py's confirmed exact call tail
# (torch_dtype=torch.float16)) and applies repo-wide in case other call
# sites share it -- trust_remote_code is a no-op for pipelines that don't
# need it, so over-matching is harmless.
RUN grep -rl "DiffusionPipeline.from_pretrained(" hy3dgen/ | \
    xargs -r sed -i 's/torch_dtype=torch\.float16)/torch_dtype=torch.float16, trust_remote_code=True)/g'

RUN pip install --no-cache-dir -r requirements.txt

# Compiled CUDA extensions the texture pipeline needs (rasterizer + renderer).
# TORCH_CUDA_ARCH_LIST above makes these build with 4090/3090 kernels included.
RUN pip install --no-cache-dir -e hy3dgen/texgen/custom_rasterizer \
    && pip install --no-cache-dir ./hy3dgen/texgen/differentiable_renderer

# requirements.txt only installs hy3dgen's *dependencies*, not the hy3dgen
# package itself -- confirmed on a real deploy: "ModuleNotFoundError: No
# module named 'hy3dgen'" the moment server.py tried to import it. Put the
# repo root on PYTHONPATH instead of relying on the repo having proper
# setup.py packaging (it may not).
ENV PYTHONPATH=/app/Hunyuan3D-2:${PYTHONPATH}

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow \
    diffusers accelerate huggingface_hub fast-simplification

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
