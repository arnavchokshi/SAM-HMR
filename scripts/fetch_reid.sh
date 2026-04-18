#!/usr/bin/env bash
# Fetch OSNet ReID weights for Tracker A (BoT-SORT + ReID + pose-OKS cascade).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/pretrain/reid"
mkdir -p "$DEST"

OSNET="$DEST/osnet_x0_25_msmt17.pt"

URLS=(
  "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt"
  "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.6/osnet_x0_25_msmt17.pt"
  "https://huggingface.co/spaces/Xenova/yolo_tracking/resolve/main/weights/osnet_x0_25_msmt17.pt"
)

dl() {
  local out="$1" ; shift
  for url in "$@"; do
    echo "  -> trying $url"
    if curl -fL --retry 3 --connect-timeout 20 -o "$out.tmp" "$url"; then
      mv "$out.tmp" "$out"
      echo "  OK: $out"
      return 0
    fi
    rm -f "$out.tmp"
  done
  return 1
}

if [ ! -s "$OSNET" ]; then
  dl "$OSNET" "${URLS[@]}" || { echo "FAIL: OSNet ReID weights"; exit 81; }
else
  echo "OSNet weights already present: $OSNET"
fi
