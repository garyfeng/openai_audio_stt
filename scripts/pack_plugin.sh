#!/usr/bin/env bash
set -euo pipefail

PLUGIN_NAME=openai_audio_stt
OUT_DIR=dist
ZIP="$OUT_DIR/${PLUGIN_NAME}.zip"

mkdir -p "$OUT_DIR"
rm -f "$ZIP"

zip -r "$ZIP" \
  manifest.yaml \
  main.py \
  provider/ \
  tools/ \
  README.md \
  PRIVACY.md \
  icon.svg \
  -x "*/__pycache__/*" "*.DS_Store" ".git/*" "dist/*" "scripts/*" ".env*"

echo "Created: $ZIP"
