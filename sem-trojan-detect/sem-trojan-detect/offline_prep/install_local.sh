#!/usr/bin/env bash
# Install sem-trojan-detect directly on a host, no docker — for running
# scripts/screen.py and scripts/screen_matcher.py natively.
#
#   ./offline_prep/install_local.sh
#   ./offline_prep/install_local.sh --index-url https://pypi.internal/simple
#   ./offline_prep/install_local.sh --with-mcp --with-cv2
#
# Everything goes into a virtualenv at ./.venv. That is the point: Ubuntu's
# system Python is "externally managed" and its numpy/opencv belong to dpkg,
# so `pip install numpy==X` there fails with "cannot uninstall numpy 1.26.4".
# A venv gives pip its own site-packages and nothing is fought over.
#
# Only numpy and Pillow are installed by default — that is genuinely all the
# detector needs, and it works across numpy 1.26 … 2.4, so whatever version
# your mirror carries is fine.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HERE/.venv}"
INDEX_ARGS=()
EXTRAS=()
PY="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --index-url)        INDEX_ARGS+=(--index-url "$2"); shift 2 ;;
    --extra-index-url)  INDEX_ARGS+=(--extra-index-url "$2"); shift 2 ;;
    --trusted-host)     INDEX_ARGS+=(--trusted-host "$2"); shift 2 ;;
    --find-links)       INDEX_ARGS+=(--find-links "$2"); shift 2 ;;
    --with-cv2)         EXTRAS+=("opencv-python-headless>=4.8,<5"); shift ;;
    --with-mcp)         EXTRAS+=("mcp>=1.0"); shift ;;
    --with-yolo)        EXTRAS+=("ultralytics>=8.1,<9"); shift ;;
    --venv)             VENV="$2"; shift 2 ;;
    -h|--help)          sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

echo "== creating virtualenv at ${VENV} =="
"${PY}" -m venv "${VENV}" || {
  echo
  echo "venv creation failed. On Debian/Ubuntu install the venv module first:"
  echo "  sudo apt-get install python3-venv"
  exit 1
}

PIP="${VENV}/bin/pip"
"${PIP}" install --upgrade "${INDEX_ARGS[@]}" pip setuptools wheel >/dev/null

echo "== installing core (numpy, Pillow) =="
"${PIP}" install "${INDEX_ARGS[@]}" -r "${HERE}/docker/requirements.txt"

if [ ${#EXTRAS[@]} -gt 0 ]; then
  echo "== installing extras: ${EXTRAS[*]} =="
  # Extras are genuinely optional: if the mirror lacks one, say so and carry on
  # rather than failing the whole install.
  for pkg in "${EXTRAS[@]}"; do
    "${PIP}" install "${INDEX_ARGS[@]}" "${pkg}" \
      || echo "  WARNING: could not install ${pkg} — continuing without it"
  done
fi

echo
echo "== verifying =="
"${VENV}/bin/python" "${HERE}/scripts/doctor.py" || true

cat <<EOF

Done. Use the venv's python for everything:

  ${VENV}/bin/python scripts/screen_matcher.py --root /data/lot42 --out /data/runs/M --report
  ${VENV}/bin/python scripts/screen.py detect --root /data/lot42 --out /data/runs/D --report

Or activate it once per shell:

  source ${VENV}/bin/activate
EOF
