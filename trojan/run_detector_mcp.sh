#!/usr/bin/env bash
# Run the trojan-detection MCP server on the OFFLINE machine as an HTTP
# service, on its own GPU. LibreChat connects to it over the network
# (see trojan/librechat/librechat.yaml).
#
#   DATA=/srv/trojan_data GPU=1 PORT=8130 ./run_detector_mcp.sh
#
#   $DATA  host directory that holds your input sets and receives D/ outputs.
#          The agent can only read/write under here. Inbound requests that
#          "send a directory of images" should drop them somewhere under $DATA.
set -euo pipefail

GPU="${GPU:-1}"                     # separate GPU from image generation
DATA="${DATA:-$PWD/trojan_data}"
PORT="${PORT:-8130}"
IMAGE="${IMAGE:-gds2sem-trojan:v1}"

mkdir -p "${DATA}/runs"

docker run --rm -it \
  --gpus "device=${GPU}" \
  -p "${PORT}:8130" \
  -v "${DATA}:/data" \
  -e TROJAN_DATA_ROOT=/data \
  -e TROJAN_OUT_ROOT=/data/runs \
  -e MCP_HTTP=1 -e MCP_PORT=8130 \
  "${IMAGE}"
