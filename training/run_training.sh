#!/usr/bin/env bash
# Launch LoRA training on the OFFLINE machine, directly against the
# gds_2_sem dataset layout (A/{train,val} = GDS, B/{train,val} = SEM).
#
# Expects on the host:
#   $DATASET                        your gds_2_sem directory (default ./gds_2_sem)
#   $WORKSPACE/config.yaml          copy of training/config/gds2sem_klein4b.yaml,
#                                   sample prompts patched by setup_dataset.py
#   $HF_CACHE                       pre-populated Hugging Face cache
#                                   (offline_prep/download_models.sh)
# Output (checkpoints + samples) lands in $WORKSPACE/output/gds2sem_klein4b_v1/
set -euo pipefail

GPU="${GPU:-0}"                      # GPU=1 ./run_training.sh for the 2nd card
DATASET="${DATASET:-$PWD/gds_2_sem}"
WORKSPACE="${WORKSPACE:-$PWD/workspace}"
HF_CACHE="${HF_CACHE:-$PWD/hf_cache}"
IMAGE="${IMAGE:-gds2sem-train:v1}"

for d in "${DATASET}/A/train" "${DATASET}/B/train" "${DATASET}/A/val"; do
  [ -d "$d" ] || { echo "ERROR: missing ${d}"; exit 1; }
done
[ -f "${WORKSPACE}/config.yaml" ] || { echo "ERROR: missing ${WORKSPACE}/config.yaml"; exit 1; }

docker run --rm -it \
  --gpus "device=${GPU}" \
  --shm-size=16g \
  -v "${WORKSPACE}:/workspace" \
  -v "${DATASET}:/workspace/gds_2_sem" \
  -v "${HF_CACHE}:/huggingface:ro" \
  -e HF_HOME=/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${IMAGE}" \
  python run.py /workspace/config.yaml
