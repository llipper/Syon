#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-}"

if [ -n "$OUTPUT" ]; then
  python -m eval.runner --output "$OUTPUT"
else
  python -m eval.runner
fi