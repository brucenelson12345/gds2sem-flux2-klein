#!/usr/bin/env bash
# Launch LoRA training on the OFFLINE machine, directly against the
# gds_2_sem dataset layout (A/{train,val} = GDS, B/{train,val} = SEM),
# using ONLY plain local model files — no Hugging Face, no HF cache.
#
# Expects on the host:
#   $DATASET                 your gds_2_sem directory (default ./gds_2_sem)
#   $WORKSPACE/config.yaml   copy of training/config/gds2sem_klein4b.yaml,
#                            sample prompts patched by setup_dataset.py
#   $MODELS                  training weights directory containing:
#                              flux-2-klein-base-4b.safetensors
#                              ae.safetensors (or flux2-vae.safetensors)
#                              qwen3-4b/   (full transformers-format dir)
#                            default: ./transfer/models/flux2-klein
#                            (check it first with offline_prep/verify_models.sh)
# Output (checkpoints + samples) lands in $WORKSPACE/output/gds2sem_klein4b_v1/
set -euo pipefail

GPU="${GPU:-0}"                      # GPU=1 ./run_training.sh for the 2nd card
DATASET="${DATASET:-$PWD/gds_2_sem}"
WORKSPACE="${WORKSPACE:-$PWD/workspace}"
MODELS="${MODELS:-$PWD/transfer/models/flux2-klein}"
IMAGE="${IMAGE:-gds2sem-train:v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for d in "${DATASET}/A/train" "${DATASET}/B/train" "${DATASET}/A/val"; do
  [ -d "$d" ] || { echo "ERROR: missing ${d}"; exit 1; }
done
[ -f "${WORKSPACE}/config.yaml" ] || { echo "ERROR: missing ${WORKSPACE}/config.yaml"; exit 1; }
[ -f "${MODELS}/flux-2-klein-base-4b.safetensors" ] || {
  echo "ERROR: missing ${MODELS}/flux-2-klein-base-4b.safetensors"; exit 1; }

docker run --rm -it \
  --gpus "device=${GPU}" \
  --shm-size=16g \
  -v "${WORKSPACE}:/workspace" \
  -v "${DATASET}:/workspace/gds_2_sem" \
  -v "${MODELS}:/models/flux2-klein:ro" \
  -v "${HERE}/scripts/run_offline.py:/app/run_offline.py:ro" \
  "${IMAGE}" \
  python /app/run_offline.py /workspace/config.yaml
