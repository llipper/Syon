#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/syon3.yaml}"
MIN_SAMPLES="${2:-10000}"
MODEL_NAME="${3:-syon-3}"

echo ""
echo "========================================"
echo "  SYON 3 — TREINAMENTO LOCAL"
echo "========================================"
echo ""

python -m training.pipeline --config "${CONFIG}" --min-samples "${MIN_SAMPLES}"

SRC="models/pretrained/syon-3"
if [[ ! -d "${SRC}" ]]; then SRC="training/checkpoints/phase4/best"; fi
python scripts/training/export_model.py --source "${SRC}" --name "${MODEL_NAME}"

python -m evaluation.benchmarks.benchmark_runner --output training/logs/syon3_eval.json || true

echo ""
echo "========================================"
echo "  SYON 3 FINALIZADO"
echo "  Modelo: models/pretrained/${MODEL_NAME}"
echo "  Logs:   training/logs/syon3_summary.json"
echo "========================================"
echo ""