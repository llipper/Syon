#!/usr/bin/env bash
set -euo pipefail

export SYON_API_HOST="${SYON_API_HOST:-0.0.0.0}"
export SYON_API_PORT="${SYON_API_PORT:-8000}"
export SYON_MODELS_DIR="${SYON_MODELS_DIR:-./models}"

pip install -e ".[dev]" -q
syon-api