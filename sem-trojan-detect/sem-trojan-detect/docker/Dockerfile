# =============================================================================
# sem-trojan-detect — detection + MCP server image (self-contained).
#
# Host constraint (matches the rest of the fleet): NVIDIA driver 560.35.05
# -> CUDA 12.6 max, so cu126 wheels. RTX 6000 Ada = sm_89.
#
# Build ONLINE from the repo root, export, load OFFLINE:
#   ./offline_prep/build_and_export_images.sh ./transfer
#
# Everything Python lives in a virtualenv at /opt/venv. That is deliberate:
# Ubuntu 24.04 ships an "externally managed" system Python whose numpy and
# opencv are owned by dpkg, and pip cannot upgrade those (no RECORD file) —
# which is exactly the "cannot install numpy X, 1.26.4 is already installed"
# build failure. A venv gives pip its own site-packages, so no apt package is
# ever fought over and `python` unambiguously means the venv interpreter.
#
# Build args:
#   WITH_TORCH=0   skip torch + ultralytics — golden backend only, ~4 GB
#   WITH_CV2=0     skip OpenCV — pure-numpy fallbacks carry everything
#   PIP_INDEX      point at an internal PyPI mirror
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9"

# NOTE: no python3-opencv / python3-numpy here. Installing those from apt is
# what collides with pip. libglib2.0-0 is the only shared lib the headless
# OpenCV wheel needs.
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.12 python3-venv python3-dev \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- the virtualenv owns every Python package from here on ------------------
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ARG PIP_INDEX=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX}

# ---- core: numpy + Pillow. That really is all the detector needs. -----------
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- optional extras --------------------------------------------------------
# OpenCV only accelerates connected components / morphology / blur; imagelib
# falls back to pure numpy when it is missing.
ARG WITH_CV2=1
RUN if [ "$WITH_CV2" = "1" ]; then \
      pip install --no-cache-dir "opencv-python-headless>=4.8,<5" ; \
    fi

# MCP server (LibreChat + `screen.py remote`).
ARG WITH_MCP=1
RUN if [ "$WITH_MCP" = "1" ]; then pip install --no-cache-dir "mcp>=1.0"; fi

# torch + ultralytics are needed ONLY by the optional YOLO backend.
ARG WITH_TORCH=1
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install --no-cache-dir \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126 && \
      pip install --no-cache-dir "ultralytics>=8.1,<9" ; \
    fi

# ---- build-time verification ------------------------------------------------
# Hard-fails on the things that are actually required; reports the optional
# ones rather than pretending they are mandatory.
RUN python - <<'EOF'
import sys
import numpy, PIL
print(f"python {sys.version.split()[0]}  ({sys.executable})")
print(f"  numpy  {numpy.__version__}  <- {numpy.__file__}")
print(f"  pillow {PIL.__version__}")
for name in ("cv2", "mcp", "torch", "ultralytics"):
    try:
        m = __import__(name)
        print(f"  {name:<11} {getattr(m, '__version__', 'present')}")
    except ImportError:
        print(f"  {name:<11} absent (optional)")
EOF

# ---- app code — this repo only, no gds2sem source dependency ----------------
COPY trojanlib /app/trojanlib
COPY scripts   /app/scripts
COPY mcp       /app/mcp

# Import the package as it will actually be imported at runtime.
RUN cd /app && python -c "import trojanlib; print('trojanlib', trojanlib.__version__, 'imports OK')"

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
# offline tripwires (nothing here needs the network except the gds2sem
# client, which talks to a service on your own LAN)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 \
    DO_NOT_TRACK=1 MPLBACKEND=Agg
ENV TROJAN_DATA_ROOT=/data TROJAN_OUT_ROOT=/data/runs
# where the gds2sem generator's ComfyUI service lives, if you use it
ENV GDS2SEM_SERVER=http://host.docker.internal:8188
# Claude access for the optional analyst summary, via your Open WebUI
# instance. Supply the token at run time (-e OPENWEBUI_API_KEY=...) or mount
# a config file — never bake a key into the image.
ENV OPENWEBUI_URL=http://host.docker.internal:3000
ENV SEM_TROJAN_CONFIG=/data/.openwebui.json
VOLUME ["/data"]

# default: MCP server over HTTP. Override the CMD to run the CLI instead
# (see run_screen.sh), or `docker run ... IMAGE doctor` to diagnose.
ENV MCP_HTTP=1 MCP_HOST=0.0.0.0 MCP_PORT=8130
EXPOSE 8130
WORKDIR /app
CMD ["python", "/app/mcp/server.py"]
