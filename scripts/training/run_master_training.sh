#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/master_full.yaml}"
PHASE="${2:-0}"
AUGMENT="${3:-50}"

echo "=== Syon Master Training ==="
python scripts/data/build_master_curriculum.py --augment "${AUGMENT}"

ARGS=(-m training.master_trainer --config "${CONFIG}" --augment "${AUGMENT}")
if [[ "${PHASE}" != "0" ]]; then
  ARGS+=(--phase "${PHASE}")
fi

python "${ARGS[@]}"
echo "Concluído → training/logs/master_summary.json"