#!/usr/bin/env bash
# Launch ComfyUI on the OFFLINE machine.
#
# Expects on the host:
#   $COMFY_MODELS/diffusion_models/flux-2-klein-4b-fp8.safetensors
#   $COMFY_MODELS/diffusion_models/flux-2-klein-base-4b-fp8.safetensors
#   $COMFY_MODELS/text_encoders/qwen_3_4b.safetensors
#   $COMFY_MODELS/vae/flux2-vae.safetensors
#   $COMFY_MODELS/loras/gds2sem_klein4b_v1.safetensors   <- your trained LoRA
#
# Then open http://<host>:8188 and load
#   inference/workflows/gds2sem_klein4b_base_lora.json      (quality)
#   inference/workflows/gds2sem_klein4b_distilled_lora.json (speed)
set -euo pipefail

GPU="${GPU:-1}"                       # keep GPU 0 free for training
COMFY_MODELS="${COMFY_MODELS:-$PWD/transfer/comfy_models}"
DATA="${DATA:-$PWD/comfy_data}"       # input/output/user dirs live here
IMAGE="${IMAGE:-gds2sem-comfy:v1}"

mkdir -p "${DATA}"/{input,output,user}

docker run --rm -it \
  --gpus "device=${GPU}" \
  -p 8188:8188 \
  -v "${COMFY_MODELS}/diffusion_models:/comfy/ComfyUI/models/diffusion_models" \
  -v "${COMFY_MODELS}/text_encoders:/comfy/ComfyUI/models/text_encoders" \
  -v "${COMFY_MODELS}/vae:/comfy/ComfyUI/models/vae" \
  -v "${COMFY_MODELS}/loras:/comfy/ComfyUI/models/loras" \
  -v "${DATA}/input:/comfy/ComfyUI/input" \
  -v "${DATA}/output:/comfy/ComfyUI/output" \
  -v "${DATA}/user:/comfy/ComfyUI/user" \
  "${IMAGE}"
