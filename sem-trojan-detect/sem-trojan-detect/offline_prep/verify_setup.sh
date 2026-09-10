#!/usr/bin/env bash
# Verify the offline machine has everything sem-trojan-detect needs.
#
#   ./offline_prep/verify_setup.sh [transfer-dir]     (default ./transfer)
#
# Checks: the docker image is loaded, the image's deps import, the data root
# exists, and (optionally) that YOLO base weights and the gds2sem generator
# service are present. The golden-model backend needs NO model weights at
# all — only the image.
set -u

ROOT="${1:-./transfer}"
IMAGE="${IMAGE:-sem-trojan-detect:v1}"
DATA="${DATA:-./trojan_data}"
GDS2SEM_SERVER="${GDS2SEM_SERVER:-http://localhost:8188}"
ok=0; bad=0; warn=0

pass () { echo "  OK      $1"; ok=$((ok+1)); }
fail () { echo "  MISSING $1"; bad=$((bad+1)); }
note () { echo "  note    $1"; warn=$((warn+1)); }

echo "== Docker image =="
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  pass "${IMAGE} loaded ($(docker image inspect -f '{{.Size}}' "${IMAGE}" | awk '{printf "%.1f GB", $1/1e9}'))"
else
  fail "${IMAGE} not loaded — run: docker load < ${ROOT}/images/sem-trojan-detect_v1.tar.gz"
fi

echo "== In-image dependencies =="
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if docker run --rm "${IMAGE}" python -c "
import cv2, numpy, PIL, mcp, trojanlib
print('trojanlib', trojanlib.__version__)
" 2>/dev/null; then
    pass "trojanlib + cv2 + numpy + PIL + mcp import cleanly"
  else
    fail "imports failed inside the image"
  fi
  if docker run --rm "${IMAGE}" python -c "import ultralytics" 2>/dev/null; then
    pass "ultralytics present (YOLO backend available)"
  else
    note "ultralytics absent — golden backend only (fine unless you want YOLO)"
  fi
fi

echo "== GPU access =="
if docker run --rm --gpus all "${IMAGE}" nvidia-smi -L >/dev/null 2>&1; then
  pass "docker can see the GPUs (nvidia-container-toolkit working)"
else
  note "no GPU visible to docker — the golden backend is CPU-only anyway"
fi

echo "== Data root =="
if [ -d "${DATA}" ]; then pass "${DATA}"; else
  note "creating ${DATA}"; mkdir -p "${DATA}/runs"; fi

echo "== Optional: YOLO base weights =="
if compgen -G "${ROOT}/models/*.pt" >/dev/null 2>&1; then
  pass "$(ls "${ROOT}"/models/*.pt | tr '\n' ' ')"
else
  note "no .pt in ${ROOT}/models/ — only needed to TRAIN the YOLO backend"
fi

echo "== Optional: gds2sem generator service =="
if curl -sf -m 3 "${GDS2SEM_SERVER}/system_stats" >/dev/null 2>&1; then
  pass "gds2sem ComfyUI reachable at ${GDS2SEM_SERVER}"
else
  note "gds2sem not reachable at ${GDS2SEM_SERVER} — only needed for
          'screen.py generate' / demo auto-generation. Start it from the
          gds2sem repo: inference/run_comfyui.sh"
fi

echo
echo "${ok} ok, ${bad} missing, ${warn} optional/notes."
if [ "${bad}" -gt 0 ]; then
  echo "Fix the missing items above before screening."
  exit 1
fi
echo "Ready. Try:  DATA=${DATA} ./run_screen.sh detect --root /data/<yourset> --out /data/runs/out --report"
