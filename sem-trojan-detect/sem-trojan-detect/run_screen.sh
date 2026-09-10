#!/usr/bin/env bash
# One-shot containerized screening — no LibreChat, no long-running service.
# Mounts $DATA at /data and runs scripts/screen.py inside the image.
#
#   DATA=/srv/trojan_data ./run_screen.sh detect --root /data/incoming/lot42 \
#       --out /data/runs/lot42_D --report
#   DATA=/srv/trojan_data ./run_screen.sh demo --root /data/gds_2_sem \
#       --out /data/runs/demo
#
# All path arguments must be /data/... paths (inside the mount).
# GPU is only needed for the YOLO backend; the golden backend is CPU-only.
set -euo pipefail

DATA="${DATA:-$PWD/trojan_data}"
IMAGE="${IMAGE:-sem-trojan-detect:v1}"
GPU="${GPU:-}"                       # e.g. GPU=1 to pin a card
GDS2SEM_SERVER="${GDS2SEM_SERVER:-http://host.docker.internal:8188}"
# Claude via your Open WebUI instance (optional: only for --summarize
# and the summarize_run tool). Export these before running, or drop a
# config file at $DATA/.openwebui.json via `screen.py llm login`.
OPENWEBUI_URL="${OPENWEBUI_URL:-http://host.docker.internal:3000}"
OPENWEBUI_API_KEY="${OPENWEBUI_API_KEY:-}"

mkdir -p "${DATA}/runs"
gpu_args=(); [ -n "${GPU}" ] && gpu_args=(--gpus "device=${GPU}")

docker run --rm -it \
  "${gpu_args[@]}" \
  --add-host=host.docker.internal:host-gateway \
  -v "${DATA}:/data" \
  -e TROJAN_DATA_ROOT=/data -e TROJAN_OUT_ROOT=/data/runs \
  -e GDS2SEM_SERVER="${GDS2SEM_SERVER}" \
  -e OPENWEBUI_URL="${OPENWEBUI_URL}" \
  -e OPENWEBUI_API_KEY="${OPENWEBUI_API_KEY}" \
  "${IMAGE}" \
  python /app/scripts/screen.py "$@"
