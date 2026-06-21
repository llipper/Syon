#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/syon3.yaml}"
PHASE="${2:-0}"
AUGMENT="${3:-3}"

echo "[Syon 3] Gerando curriculum..."
python scripts/data/build_syon3_curriculum.py --augment "${AUGMENT}"

ARGS=(-m training.syon3_trainer --config "${CONFIG}" --augment "${AUGMENT}")
if [[ "${PHASE}" != "0" ]]; then
  ARGS+=(--phase "${PHASE}")
fi
python "${ARGS[@]}"

echo "Concluído → training/logs/syon3_summary.json"