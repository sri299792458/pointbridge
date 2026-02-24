#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/2] Checking PointBridge third-party submodules..."
if [ ! -f "third_party/segment-anything-2-real-time/checkpoints/download_ckpts.sh" ]; then
  echo "SAM2 checkpoint downloader not found."
  echo "Initialize submodules first:"
  echo "  git submodule update --init --recursive"
  exit 1
fi

echo "[2/2] Downloading SAM2 checkpoints via PointBridge submodule script..."
(
  cd third_party/segment-anything-2-real-time/checkpoints
  bash download_ckpts.sh
)

cat <<'EOM'

Download complete.

PointBridge real-world pipeline also uses:
  - Molmo (downloaded automatically by Transformers when running model server)
  - FoundationStereo (see docs/install.md for docker + pretrained plan setup)

This script does NOT download policy checkpoints (snapshot.pt).
You still need your own PointBridge policy checkpoint for deployment.

EOM
