#!/usr/bin/env bash
# Run on the ONLINE machine. Downloads every model weight needed for both
# training and ComfyUI inference into ./transfer/, ready to copy to the
# offline machine. Total ~30 GB.
#
#   pip install -U huggingface_hub
set -euo pipefail

ROOT="${1:-./transfer}"
HF_CACHE="${ROOT}/hf_cache"
COMFY_MODELS="${ROOT}/comfy_models"
mkdir -p "${HF_CACHE}" "${COMFY_MODELS}"/{diffusion_models,text_encoders,vae,loras}

export HF_HOME="${HF_CACHE}"

echo "== Training weights (into HF cache; mounted read-only into the training container) =="
# FLUX.2 Klein 4B base transformer (undistilled -> the one you train on), ~7.8 GB
hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors
# FLUX.2 VAE (ai-toolkit resolves the VAE from this repo), ~0.3 GB
hf download ai-toolkit/flux2_vae ae.safetensors
# Text encoder — ai-toolkit's flux2_klein_4b arch loads Qwen/Qwen3-4B from the
# HF cache (repo id hardcoded in the arch), ~8 GB
hf download Qwen/Qwen3-4B --exclude "*.gguf" "*.onnx*"

echo "== ComfyUI inference weights =="
dl () { # url -> dest
  echo "downloading $2"
  curl -L --fail --retry 3 -o "$2" "$1"
}
# Distilled Klein 4B (4-step, CFG 1) — fast inference
dl "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors" \
   "${COMFY_MODELS}/diffusion_models/flux-2-klein-4b-fp8.safetensors"
# Base Klein 4B fp8 (20-step, CFG ~4) — matches what the LoRA was trained on
dl "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors" \
   "${COMFY_MODELS}/diffusion_models/flux-2-klein-base-4b-fp8.safetensors"
# Qwen3-4B text encoder in ComfyUI single-file form
dl "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
   "${COMFY_MODELS}/text_encoders/qwen_3_4b.safetensors"
# FLUX.2 VAE in ComfyUI single-file form
dl "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
   "${COMFY_MODELS}/vae/flux2-vae.safetensors"

echo
echo "Done. Copy '${ROOT}' to the transfer share."
echo "  hf_cache/      -> mount into the training container at /huggingface"
echo "  comfy_models/  -> mount into the ComfyUI container at /comfy/ComfyUI/models/... (see inference/run_comfyui.sh)"
echo "  loras/         -> drop your trained LoRA .safetensors here later"
