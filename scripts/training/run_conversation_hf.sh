#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/syon3_conversation_hf.yaml}"
RESUME="${2:-}"
IMPORT="${3:-}"

if [[ "${IMPORT}" == "import" ]]; then
  echo "[1/2] Importando conversação PT local..."
  python scripts/data/import_conversation_datasets.py \
    --aira-limit 50000 \
    --ultrachat-limit 100000
fi

ARGS=(-m training.hf.conversation_trainer --config "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

echo "[Syon/HF] Treino de conversação..."
python "${ARGS[@]}"
echo "Modelo de chat: models/pretrained/syon-3-chat"