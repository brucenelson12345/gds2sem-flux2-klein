#!/usr/bin/env bash
# One-shot containerized screening from the command line — no LibreChat, no
# long-running service. Mounts $DATA at /data and runs screen.py inside the
# gds2sem-trojan image.
#
#   DATA=/srv/trojan_data ./run_screen.sh detect --root /data/incoming/lot42 \
#       --out /data/runs/lot42_D --report
#   DATA=/srv/trojan_data ./run_screen.sh demo --root /data/gds_2_sem \
#       --out /data/runs/demo
#
# All path arguments you pass must be /data/... paths (inside the mount).
set -euo pipefail

GPU="${GPU:-1}"
DATA="${DATA:-$PWD/trojan_data}"
IMAGE="${IMAGE:-gds2sem-trojan:v1}"

mkdir -p "${DATA}"

docker run --rm -it \
  --gpus "device=${GPU}" \
  -v "${DATA}:/data" \
  "${IMAGE}" \
  python /app/trojan/scripts/screen.py "$@"
