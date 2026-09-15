#!/usr/bin/env bash
# Run on the ONLINE machine (docker installed; no GPU needed).
# Builds this repo's image and exports it for the transfer share.
#
#   ./offline_prep/build_and_export_images.sh ./transfer
#
# Build args, all optional:
#   WITH_TORCH=0   golden backend only, no torch/YOLO  (~4 GB instead of ~8 GB)
#   WITH_CV2=0     skip OpenCV — pure-numpy fallbacks carry everything
#   WITH_MCP=0     skip the MCP server deps
#   PIP_INDEX=...  build against an internal PyPI mirror
#
# The image installs every Python package into a virtualenv at /opt/venv, so
# pip never fights Ubuntu's dpkg-managed numpy/opencv.
set -euo pipefail

ROOT="${1:-./transfer}"
WITH_TORCH="${WITH_TORCH:-1}"
WITH_CV2="${WITH_CV2:-1}"
WITH_MCP="${WITH_MCP:-1}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/images"

echo "== Building sem-trojan-detect image (torch cu126, WITH_TORCH=${WITH_TORCH}) =="
docker build -t sem-trojan-detect:v1 \
  --build-arg "WITH_TORCH=${WITH_TORCH}" \
  --build-arg "WITH_CV2=${WITH_CV2}" \
  --build-arg "WITH_MCP=${WITH_MCP}" \
  --build-arg "PIP_INDEX=${PIP_INDEX}" \
  -f "${HERE}/docker/Dockerfile" "${HERE}"

echo "== Exporting =="
docker save sem-trojan-detect:v1 | gzip > "${ROOT}/images/sem-trojan-detect_v1.tar.gz"

cat <<EOF

Done.

Copy '${ROOT}' to the transfer share, then on the offline machine:
  docker load < images/sem-trojan-detect_v1.tar.gz
  ./offline_prep/verify_setup.sh ./transfer

If you plan to train the optional YOLO backend, also stage a base
checkpoint (e.g. yolo11s.pt) into '${ROOT}/models/' while you are online.
EOF
