#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/training/parallel_sft.yaml}"
DATA_DIR="${2:-data/raw}"

python -m training.parallel_trainer --config "$CONFIG" --data-dir "$DATA_DIR"