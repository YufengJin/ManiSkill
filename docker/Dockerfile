# =========================================================
# ManiSkill Dockerfile (Micromamba-based, GPU-ready, Vulkan)
# =========================================================

FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ---------------------------------------------------------
# System dependencies (Vulkan, rendering, SSL) + debug tools
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libegl1 \
    libosmesa6 \
    libvulkan1 \
    vulkan-tools \
    libjpeg-dev \
    libpng-dev \
    ffmpeg \
    lsof \
    psmisc \
    tmux \
    vim \
    x11-apps \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Micromamba
# ---------------------------------------------------------
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o /usr/bin/micromamba \
    && chmod +x /usr/bin/micromamba

# ---------------------------------------------------------
# Create conda environment (Python 3.10 for ManiSkill 3.x)
# ---------------------------------------------------------
RUN micromamba create -y -n maniskill python=3.10 -c conda-forge \
    && micromamba clean -a -y

# ---------------------------------------------------------
# Configure shell hook for micromamba
# ---------------------------------------------------------
RUN /bin/bash -c "micromamba shell init -s bash" \
    && echo "micromamba activate maniskill" >> /root/.bashrc

# ---------------------------------------------------------
# Upgrade pip; install setuptools<75 for pkg_resources (SAPIEN requires it; removed in setuptools 82+)
# ---------------------------------------------------------
RUN micromamba run -n maniskill python -m pip install --upgrade pip \
    && micromamba run -n maniskill python -m pip install "setuptools<75"

# ---------------------------------------------------------
# Install PyTorch and ManiSkill
# ---------------------------------------------------------
RUN micromamba run -n maniskill python -m pip install torch
RUN micromamba run -n maniskill python -m pip install --upgrade mani_skill==3.0.0b22

# ---------------------------------------------------------
# Vulkan ICD config (required for GPU rendering in container)
# https://github.com/haosulab/ManiSkill/issues/9
# ---------------------------------------------------------
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# ---------------------------------------------------------
# Download PhysX GPU binary via SAPIEN (for GPU sim)
# ---------------------------------------------------------
RUN micromamba run -n maniskill python -c "import sapien.physx as physx; physx.enable_gpu()" || true

# ---------------------------------------------------------
# Set up workspace directory (mani_skill installed in entrypoint for in-container dev)
# ---------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
