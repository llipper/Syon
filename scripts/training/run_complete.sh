#!/usr/bin/env bash
# SYON — PIPELINE COMPLETO DE TREINAMENTO MASTER
set -euo pipefail

CONFIG="${1:-training/configs/master_complete.yaml}"
MIN_SAMPLES="${2:-10000}"
MODEL_NAME="${3:-syon-master-lora}"

echo "========================================"
echo "  SYON — TREINAMENTO MASTER COMPLETO"
echo "========================================"

echo "[1/6] Curriculum..."
python scripts/data/build_master_curriculum.py --min-samples "${MIN_SAMPLES}"

echo "[2/6] Processamento..."
python scripts/data/process_pipeline.py

echo "[3/6] Fase 1..."
python -m training.master_trainer --config "${CONFIG}" --phase 1 --augment 1

echo "[4/6] Fase 2..."
python -m training.master_trainer --config "${CONFIG}" --phase 2 --resume training/checkpoints/phase1/best --augment 1

echo "[5/6] Fase 3..."
python -m training.master_trainer --config "${CONFIG}" --phase 3 --resume training/checkpoints/phase2/best --augment 1

SRC="models/pretrained/syon-master-lora"
[[ -d "${SRC}" ]] || SRC="training/checkpoints/phase3/best"
echo "[6/6] Export..."
python scripts/training/export_model.py --source "${SRC}" --name "${MODEL_NAME}"

python -m evaluation.benchmarks.benchmark_runner --output training/logs/master_eval.json || true

echo "COMPLETO → models/pretrained/${MODEL_NAME}"