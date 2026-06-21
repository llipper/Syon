#!/bin/sh
set -e

MODEL_PATH="/models/syon-7b.gguf"
API_PORT="8000"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

export SYON_API_PORT="$API_PORT"
export SYON_DEFAULT_MODEL_PATH="$MODEL_PATH"

echo "[Syon] Starting API on port ${API_PORT}"
echo "[Syon] Model path: ${MODEL_PATH}"

exec syon-api