# =============================================================================
# sem-trojan-detect — detection + MCP server image (self-contained).
#
# Host constraint (matches the rest of the fleet): NVIDIA driver 560.35.05
# -> CUDA 12.6 max, so cu126 wheels. RTX 6000 Ada = sm_89.
#
# Build ONLINE from the repo root, export, load OFFLINE:
#   ./offline_prep/build_and_export_images.sh ./transfer
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.12 python3-pip python3-dev python3-venv \
    python3-opencv ffmpeg libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# torch is needed ONLY by the optional YOLO backend; the golden-model
# detector is pure numpy/opencv and runs without it.
ARG WITH_TORCH=1
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install --no-cache-dir --break-system-packages \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126 ; \
    fi

COPY docker/requirements.txt /tmp/req.txt
RUN printf 'torch==2.13.0+cu126\ntorchvision==0.28.0+cu126\n' > /tmp/tc.txt && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -c /tmp/tc.txt -r /tmp/req.txt

RUN python - <<'EOF'
import cv2, numpy, PIL, mcp
try:
    import torch
    assert "+cu126" in torch.__version__, torch.__version__
    print("deps OK (with torch):", torch.__version__, "cv2", cv2.__version__)
except ImportError:
    print("deps OK (golden backend only, no torch); cv2", cv2.__version__)
EOF

# app code — this repo only; no dependency on the gds2sem source tree
COPY trojanlib /app/trojanlib
COPY scripts   /app/scripts
COPY mcp       /app/mcp

ENV PYTHONPATH=/app
# offline tripwires (nothing here needs the network except the gds2sem
# client, which talks to a service on your own LAN)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 \
    DO_NOT_TRACK=1 MPLBACKEND=Agg
ENV TROJAN_DATA_ROOT=/data TROJAN_OUT_ROOT=/data/runs
# where the gds2sem generator's ComfyUI service lives, if you use it
ENV GDS2SEM_SERVER=http://host.docker.internal:8188
# Claude access for the optional analyst summary, via your Open WebUI
# instance. Supply the token at run time (-e OPENWEBUI_API_KEY=...) or mount
# a config file at /root/.config/sem-trojan-detect/config.json — never bake
# a key into the image.
ENV OPENWEBUI_URL=http://host.docker.internal:3000
ENV SEM_TROJAN_CONFIG=/data/.openwebui.json
VOLUME ["/data"]

# default: MCP server over HTTP. Override the CMD to run the CLI instead
# (see run_screen.sh).
ENV MCP_HTTP=1 MCP_HOST=0.0.0.0 MCP_PORT=8130
EXPOSE 8130
WORKDIR /app
CMD ["python", "/app/mcp/server.py"]
