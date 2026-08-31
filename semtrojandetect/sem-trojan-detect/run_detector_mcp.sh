#!/usr/bin/env bash
# Run the MCP server as an HTTP service on the offline host. LibreChat (or
# scripts/screen.py remote) connects to it over the network.
#
#   DATA=/srv/trojan_data GPU=1 PORT=8130 ./run_detector_mcp.sh
#
#   $DATA  the only tree the agent can read/write. A remote requester who
#          "sends a directory of images" drops it under $DATA (e.g.
#          $DATA/incoming/lot42/{A,B,C}); D output lands in $DATA/runs/.
set -euo pipefail

DATA="${DATA:-$PWD/trojan_data}"
PORT="${PORT:-8130}"
IMAGE="${IMAGE:-sem-trojan-detect:v1}"
GPU="${GPU:-}"
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
  -p "${PORT}:8130" \
  -v "${DATA}:/data" \
  -e TROJAN_DATA_ROOT=/data -e TROJAN_OUT_ROOT=/data/runs \
  -e GDS2SEM_SERVER="${GDS2SEM_SERVER}" \
  -e OPENWEBUI_URL="${OPENWEBUI_URL}" \
  -e OPENWEBUI_API_KEY="${OPENWEBUI_API_KEY}" \
  -e MCP_HTTP=1 -e MCP_PORT=8130 \
  "${IMAGE}"
