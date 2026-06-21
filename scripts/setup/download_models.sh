#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-syon-7b}"
DEST="${2:-models/pretrained}"
mkdir -p "$DEST"
echo "huggingface-cli download syon-ai/$MODEL --local-dir $DEST/$MODEL"