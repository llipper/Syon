#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-syon-7b}"
DEST="${2:-./models}"

mkdir -p "$DEST"

case "$MODEL" in
  syon-7b)  FILE="syon-7b.gguf" ;;
  syon-13b) FILE="syon-13b.gguf" ;;
  syon-70b) FILE="syon-70b.gguf" ;;
  *) echo "Modelo desconhecido: $MODEL"; exit 1 ;;
esac

echo "Download: https://huggingface.co/syon-ai/$MODEL -> $DEST/$FILE"
echo "Configure HF_TOKEN se necessário e use huggingface-cli download syon-ai/$MODEL $FILE --local-dir $DEST"