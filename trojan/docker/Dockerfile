# =============================================================================
# Hardware-trojan detection image (golden-model detector + optional YOLO)
# + MCP server for LibreChat.
#
# Same host constraint as training/inference: driver 560.35.05 -> cu126.
# Build ONLINE, export, load OFFLINE (see offline_prep/build_and_export_images.sh
# or the commands in trojan/README section).
#
#   docker build -t gds2sem-trojan:v1 -f trojan/docker/Dockerfile .
#   docker save gds2sem-trojan:v1 | gzip > gds2sem-trojan_v1.tar.gz
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# RTX 6000 Ada = sm_89
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.12 python3-pip python3-dev python3-venv \
    python3-opencv ffmpeg libgl1 git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# torch cu126 (only needed for the optional YOLO backend; the golden-model
# detector is pure numpy/opencv and would run without it)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# App deps. ultralytics pulls YOLO; mcp is the LibreChat server SDK.
COPY trojan/docker/requirements.txt /tmp/req.txt
RUN printf 'torch==2.13.0+cu126\ntorchvision==0.28.0+cu126\n' > /tmp/tc.txt && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -c /tmp/tc.txt -r /tmp/req.txt

RUN python - <<'EOF'
import torch, cv2, numpy, PIL, mcp
assert "+cu126" in torch.__version__, torch.__version__
print("deps OK:", torch.__version__, "cv2", cv2.__version__)
EOF

# app code (scripts + eval helpers + mcp server)
COPY eval /app/eval
COPY trojan/scripts /app/trojan/scripts
COPY trojan/mcp /app/trojan/mcp

ENV PYTHONPATH=/app/eval:/app/trojan/scripts
# offline tripwires (YOLO/torch never need the network at inference)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 \
    DO_NOT_TRACK=1 MPLBACKEND=Agg
# data mount + default output location, and where the MCP server is rooted
ENV TROJAN_DATA_ROOT=/data TROJAN_OUT_ROOT=/data/runs
VOLUME ["/data"]

# run the MCP server as an HTTP service by default (a separate GPU can be
# pinned with --gpus at `docker run`). LibreChat connects over the network.
ENV MCP_HTTP=1 MCP_HOST=0.0.0.0 MCP_PORT=8130
EXPOSE 8130
WORKDIR /app
CMD ["python", "/app/trojan/mcp/server.py"]
