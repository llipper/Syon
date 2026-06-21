#!/usr/bin/env bash
# Treinamento em GPU única
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${1:-${PROJECT_ROOT}/training/configs/base_config.yaml}"
DATA_DIR="${2:-${PROJECT_ROOT}/data/raw}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SYON_ENV="${SYON_ENV:-training}"

cd "${PROJECT_ROOT}"
python -m training.core.trainer \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  "$@"