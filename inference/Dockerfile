# =============================================================================
# ComfyUI inference image — FLUX.2 Klein 4B + GDS->SEM LoRA
# Same host constraint as training: driver 560.35.05 -> cu126 torch wheels.
#
# Build online:  docker build -t gds2sem-comfy:v1 -f inference/Dockerfile inference
# Export:        docker save gds2sem-comfy:v1 | gzip > gds2sem-comfy_v1.tar.gz
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    git curl python3.12 python3-pip python3-dev python3-venv ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --break-system-packages \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu126

WORKDIR /comfy
# FLUX.2 Klein nodes (Flux2Scheduler, EmptyFlux2LatentImage, ReferenceLatent)
# need a recent ComfyUI. Pin a release tag for reproducibility if desired:
#   ARG COMFY_REF=v0.3.x
ARG COMFY_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && git checkout ${COMFY_REF}

RUN printf 'torch==2.13.0+cu126\ntorchvision==0.28.0+cu126\ntorchaudio==2.11.0\n' \
      > /tmp/torch-constraints.txt && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -c /tmp/torch-constraints.txt \
      -r /comfy/ComfyUI/requirements.txt

RUN python - <<'EOF'
import torch
assert "+cu126" in torch.__version__, torch.__version__
print("torch OK:", torch.__version__)
EOF

# Model dirs are bind-mounted at runtime (see run_comfyui.sh)
VOLUME ["/comfy/ComfyUI/models", "/comfy/ComfyUI/input", "/comfy/ComfyUI/output", "/comfy/ComfyUI/user"]

EXPOSE 8188
WORKDIR /comfy/ComfyUI
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
