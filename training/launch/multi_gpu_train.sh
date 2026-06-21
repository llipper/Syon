#!/usr/bin/env bash
# Treinamento multi-GPU com torchrun
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${1:-${PROJECT_ROOT}/training/configs/parallel_config.yaml}"
DATA_DIR="${2:-${PROJECT_ROOT}/data/raw}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export SYON_ENV="${SYON_ENV:-training}"

cd "${PROJECT_ROOT}"
torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m training.core.trainer \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  "${@:3}"