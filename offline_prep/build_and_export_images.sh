#!/usr/bin/env bash
# Run on the ONLINE machine (docker installed; no GPU required).
# Builds both docker images and exports them as tarballs for the transfer share.
set -euo pipefail

ROOT="${1:-./transfer}"
mkdir -p "${ROOT}/images"

HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== Building training image (ai-toolkit, torch cu126) =="
docker build -t gds2sem-train:v1 -f "${HERE}/training/Dockerfile" "${HERE}/training"

echo "== Building inference image (ComfyUI, torch cu126) =="
docker build -t gds2sem-comfy:v1 -f "${HERE}/inference/Dockerfile" "${HERE}/inference"

echo "== Building trojan detection + MCP image (torch cu126) =="
# built from the repo root: its Dockerfile COPYs eval/ and trojan/
docker build -t gds2sem-trojan:v1 -f "${HERE}/trojan/docker/Dockerfile" "${HERE}"

echo "== Exporting =="
docker save gds2sem-train:v1  | gzip > "${ROOT}/images/gds2sem-train_v1.tar.gz"
docker save gds2sem-comfy:v1  | gzip > "${ROOT}/images/gds2sem-comfy_v1.tar.gz"
docker save gds2sem-trojan:v1 | gzip > "${ROOT}/images/gds2sem-trojan_v1.tar.gz"

echo "Done. On the offline machine:"
echo "  docker load < images/gds2sem-train_v1.tar.gz"
echo "  docker load < images/gds2sem-comfy_v1.tar.gz"
echo "  docker load < images/gds2sem-trojan_v1.tar.gz"
