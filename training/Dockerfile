# =============================================================================
# ai-toolkit training image for FLUX.2 Klein 4B — GDS -> SEM edit LoRA
#
# Target host: Ubuntu 24.04, NVIDIA driver 560.35.05 (CUDA 12.6 max)
#   -> we MUST use cu126 torch wheels, NOT the cu130 wheels the upstream
#      ai-toolkit Dockerfile uses (those need driver >= 580).
#
# Build ON THE ONLINE MACHINE (no GPU needed to build):
#   docker build -t gds2sem-train:v1 -f training/Dockerfile .
#   docker save gds2sem-train:v1 | gzip > gds2sem-train_v1.tar.gz
# On the offline machine:
#   docker load < gds2sem-train_v1.tar.gz
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# RTX 6000 Ada = sm_89
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN apt-get update && apt-get install --no-install-recommends -y \
    git curl wget rsync unzip \
    python3.12 python3-pip python3-dev python3-venv \
    python3-opencv ffmpeg build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ---- torch first, pinned to the versions ai-toolkit's manager/spec.py pins,
# ---- but from the cu126 index (driver 560-compatible)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app/ai-toolkit

# Pin a known-good commit if you want reproducible rebuilds:
#   ARG GIT_COMMIT=<sha>   and add `git checkout ${GIT_COMMIT}`
ARG GIT_COMMIT=main
RUN git clone https://github.com/ostris/ai-toolkit.git /app/ai-toolkit && \
    git checkout ${GIT_COMMIT}

# Constraints stop pip from silently backtracking torch to an older version
# while resolving requirements.txt (see warning in ai-toolkit manager/spec.py).
RUN printf 'torch==2.13.0+cu126\ntorchvision==0.28.0+cu126\ntorchaudio==2.11.0\n' \
      > /tmp/torch-constraints.txt && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -c /tmp/torch-constraints.txt \
      -r requirements.txt && \
    pip install --no-cache-dir --break-system-packages setuptools==69.5.1

# Accelerators, cu126 builds (same versions as upstream, different CUDA tag).
# torchcodec 0.15 overrides the older pin in requirements (audio/video only,
# but keeps imports happy). flash-attn / natten are optional speedups.
RUN pip install --no-cache-dir --break-system-packages -U \
      -c /tmp/torch-constraints.txt \
      torchcodec==0.15.0 && \
    (pip install --no-cache-dir --break-system-packages \
      natten==0.21.7+torch2130cu126 --find-links https://whl.natten.org || \
      echo "WARN: natten install failed (optional)") && \
    (pip install --no-cache-dir --break-system-packages \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu126torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl || \
      echo "WARN: flash-attn install failed (optional)")

# Sanity check: correct torch build survived dependency resolution
RUN python - <<'EOF'
import torch, torchvision
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert "+cu126" in torch.__version__, torch.__version__
print("torch OK:", torch.__version__, "| torchvision:", torchvision.__version__)
EOF

# ---- offline runtime behavior ------------------------------------------------
# NO Hugging Face usage at runtime: all weights are plain local files mounted
# at /models/flux2-klein and wired in by run_offline.py. The offline env vars
# are a tripwire — any code path that still tried to reach the hub fails fast
# instead of hanging. (The transformers/huggingface_hub *libraries* remain
# installed as ai-toolkit dependencies; they just never touch the network or
# a cache.)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV DO_NOT_TRACK=1
ENV DISABLE_TELEMETRY=1
ENV FLUX2_MODEL_DIR=/models/flux2-klein

# Local-weights launcher (patches ai-toolkit's hardcoded text-encoder hub id
# to the mounted local directory, then runs run.py).
COPY scripts/run_offline.py /app/run_offline.py

VOLUME ["/models/flux2-klein", "/workspace"]

WORKDIR /app/ai-toolkit
# default: run whatever config is mounted at /workspace/config.yaml
CMD ["python", "/app/run_offline.py", "/workspace/config.yaml"]
