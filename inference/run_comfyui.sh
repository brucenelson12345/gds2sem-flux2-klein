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
# On start this seeds $DATA/user/default/workflows/ with the repo's workflow
# JSONs, so they appear in ComfyUI's Workflows sidebar and persist across
# restarts. The bundled startup extension then opens one automatically —
# pick which with DEFAULT_WORKFLOW.
#
# Then open http://<host>:8188.
set -euo pipefail

GPU="${GPU:-1}"                       # keep GPU 0 free for training
COMFY_MODELS="${COMFY_MODELS:-$PWD/transfer/comfy_models}"
DATA="${DATA:-$PWD/comfy_data}"       # input/output/user dirs live here
IMAGE="${IMAGE:-gds2sem-comfy:v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Which workflow the startup extension opens. Must be a filename that exists
# in user/default/workflows/ (seeded below).
DEFAULT_WORKFLOW="${DEFAULT_WORKFLOW:-gds2sem_klein4b_base_lora.json}"
# new-session = only when the browser has no restored workflow (safe default)
# always      = every page load, overriding whatever was open
# off         = leave ComfyUI's stock default graph alone
DEFAULT_WORKFLOW_MODE="${DEFAULT_WORKFLOW_MODE:-new-session}"
# Set FORCE_SEED=1 to overwrite workflow JSONs already in user/default/workflows
FORCE_SEED="${FORCE_SEED:-0}"

mkdir -p "${DATA}"/{input,output,user} "${DATA}/user/default/workflows"

# ---- seed the workflows the UI browses -------------------------------------
for f in "${HERE}"/workflows/*.json; do
  [ -e "$f" ] || continue
  dst="${DATA}/user/default/workflows/$(basename "$f")"
  if [ "${FORCE_SEED}" = "1" ] || [ ! -f "$dst" ]; then
    cp "$f" "$dst"
    echo "seeded workflow: $(basename "$f")"
  fi
done

# ---- pin the startup extension's settings ----------------------------------
# ComfyUI reads frontend settings from user/default/comfy.settings.json. We
# merge our two keys in without disturbing anything else the user has set.
SETTINGS="${DATA}/user/default/comfy.settings.json"
python3 - "$SETTINGS" "$DEFAULT_WORKFLOW" "$DEFAULT_WORKFLOW_MODE" <<'PY' || \
  echo "note: could not pre-set workflow settings (python3 missing?) — set them in the UI"
import json, sys, os
path, wf, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    cfg = json.load(open(path))
    if not isinstance(cfg, dict):
        cfg = {}
except Exception:
    cfg = {}
cfg["gds2sem.DefaultWorkflow.File"] = wf
cfg["gds2sem.DefaultWorkflow.Mode"] = mode
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(cfg, open(path, "w"), indent=2)
print(f"default workflow: {wf}  (mode: {mode})")
PY

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
