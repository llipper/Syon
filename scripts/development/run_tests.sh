#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/ -v --cov=api --cov=inference --cov=security --cov=training --cov=evaluation