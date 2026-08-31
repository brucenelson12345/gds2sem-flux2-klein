#!/usr/bin/env bash
# Verify the transfer directory holds every model file the workflow needs,
# as plain local files (no Hugging Face cache, no hub access anywhere).
#
#   ./verify_models.sh [transfer-dir]     (default ./transfer)
#
# Expected layout:
#   transfer/
#   ├── models/flux2-klein/                  TRAINING weights
#   │   ├── flux-2-klein-base-4b.safetensors   transformer, undistilled base (~7.8 GB)
#   │   ├── ae.safetensors                     FLUX.2 VAE, BFL single-file format
#   │   │                                      (flux2-vae.safetensors also accepted)
#   │   └── qwen3-4b/                          text encoder — FULL transformers
#   │       ├── config.json                    format directory (config, tokenizer,
#   │       ├── tokenizer.json ...             weight shards + index). The ComfyUI
#   │       └── model*.safetensors             single-file TE does NOT work here.
#   └── comfy_models/                        INFERENCE (ComfyUI) weights
#       ├── diffusion_models/flux-2-klein-4b-fp8.safetensors        (distilled)
#       ├── diffusion_models/flux-2-klein-base-4b-fp8.safetensors   (base)
#       ├── text_encoders/qwen_3_4b.safetensors
#       ├── vae/flux2-vae.safetensors
#       └── loras/                           (trained LoRAs land here)
set -u

ROOT="${1:-./transfer}"
ok=0; bad=0

check_file () {  # path [human note]
  if [ -f "$1" ]; then
    sz=$(du -h "$1" | cut -f1)
    echo "  OK      $1  (${sz})"
    ok=$((ok+1))
  else
    echo "  MISSING $1${2:+   <- $2}"
    bad=$((bad+1))
  fi
}
check_any () {  # note path1 path2
  if [ -f "$2" ]; then check_file "$2"
  elif [ -f "$3" ]; then check_file "$3"
  else
    echo "  MISSING $2 (or $3)   <- $1"
    bad=$((bad+1))
  fi
}
check_glob () {  # dir pattern note
  if compgen -G "$1/$2" > /dev/null 2>&1; then
    echo "  OK      $1/$2"
    ok=$((ok+1))
  else
    echo "  MISSING $1/$2   <- $3"
    bad=$((bad+1))
  fi
}

T="${ROOT}/models/flux2-klein"
echo "== Training weights (${T}) =="
check_file "${T}/flux-2-klein-base-4b.safetensors" "undistilled base transformer"
check_any  "FLUX.2 VAE (BFL single-file format)" \
           "${T}/ae.safetensors" "${T}/flux2-vae.safetensors"
check_file "${T}/qwen3-4b/config.json" "full transformers-format Qwen3-4B dir"
check_glob "${T}/qwen3-4b" "tokenizer*" "Qwen3-4B tokenizer files"
check_glob "${T}/qwen3-4b" "*.safetensors" "Qwen3-4B weight shard(s)"

C="${ROOT}/comfy_models"
echo "== ComfyUI weights (${C}) =="
check_file "${C}/diffusion_models/flux-2-klein-4b-fp8.safetensors" "distilled (4-step) model"
check_file "${C}/diffusion_models/flux-2-klein-base-4b-fp8.safetensors" "base (20-step) model"
check_file "${C}/text_encoders/qwen_3_4b.safetensors" "ComfyUI single-file TE"
check_file "${C}/vae/flux2-vae.safetensors" "ComfyUI VAE"
[ -d "${C}/loras" ] && echo "  OK      ${C}/loras/" || {
  echo "  note    creating ${C}/loras/"; mkdir -p "${C}/loras"; }

echo
echo "${ok} present, ${bad} missing."
if [ "${bad}" -gt 0 ]; then
  echo "Fix the missing entries before training/inference. If your files use"
  echo "different names, rename or symlink them into this layout."
  exit 1
fi
echo "All model files in place — no network access will be needed."
